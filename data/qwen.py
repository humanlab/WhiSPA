#!/usr/bin/env python3
"""
Accelerate-powered, resumable Qwen embedding extractor.

- Reads an input JSONL dataset.
- Extracts text per sample using --extract_key (default: transcription).
- If --extract_key == affect, converts the JSON dict to a string via json_to_fenced_json_str
  using the same selection logic as in EmbeddingDataset from qwen_old.py.
- Computes embeddings with Qwen3-Embedding and saves per-sample .npy files in --embedding_dir.
- Writes per-rank temp JSONL files with the updated samples, where the new attribute key is the
  basename of --embedding_dir and its value is the saved .npy filepath.
- Merges all rank temp files (and optionally an existing output file) into the final --output_path,
  deduplicated by index, ensuring the final count equals the input dataset's number of lines.

This script is designed to be resumable and efficient for large datasets.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Ensure project root is importable for train.utils and data.utils
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.utils import last_token_pool, json_to_fenced_json_str


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


"""
Example usage:
accelerate launch --num_processes 8 --mixed_precision fp16 data/qwen.py \
    --model_id Qwen/Qwen3-Embedding-8B \
    --dataset_path /mnt/vast/data/speech/peoplespeech/test/dataset_clean.jsonl \
    --embedding_dir /mnt/vast/data/speech/peoplespeech/test/qwen_4096_affect \
    --output_path /mnt/vast/data/speech/peoplespeech/test/dataset_clean.qwen_4096_affect.jsonl \
    --extract_key affect \
    --batch_size 64

Note: This implementation uses last_token_pool as recommended for Qwen3-Embedding models.
The --extract_key parameter specifies which field to extract embeddings from (default: transcription).
The --instruction parameter is optional and should be used when you want task-specific embeddings.
"""


class EmbeddingDataset(Dataset):
    """Dataset that loads JSONL samples and extracts text to embed.

    - if sample['affect'] is a dict; find a key containing 'affect' or 'context',
      then convert that sub-object to a fenced JSON string.
    """

    def __init__(
        self,
        dataset_path: str,
        extract_key: str = 'transcription',
        skip_indices: Optional[Set[int]] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.extract_key = extract_key
        self.skip_indices = skip_indices or set()
        self.samples: List[Dict[str, Any]] = []
        self.total_in_file: int = 0

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.total_in_file += 1
                if 'index' not in sample:
                    # Keep non-indexed samples out; cannot safely dedupe/merge
                    continue

                if sample['index'] in self.skip_indices:
                    continue

                # Prepare text if possible; we also carry a flag if missing
                has_extract_key = self.extract_key in sample
                text: Optional[str] = None
                if has_extract_key:
                    if isinstance(sample[self.extract_key], dict):
                        for key in sample[self.extract_key].keys():
                            if 'affect' in key or 'context' in key:
                                text = json_to_fenced_json_str(sample[self.extract_key][key])
                                break
                        if text is None:
                            text = json_to_fenced_json_str(sample[self.extract_key])
                    else:
                        text = str(sample[self.extract_key])
                else:
                    raise ValueError(f"Invalid extract key: {self.extract_key}")

                self.samples.append({
                    'index': sample['index'],
                    'text': text,
                    'has_extract_key': has_extract_key,
                    'original_data': sample,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        'indices': [b['index'] for b in batch],
        'texts': [b['text'] for b in batch],
        'has_extract_key': [b['has_extract_key'] for b in batch],
        'original_data': [b['original_data'] for b in batch],
    }


class EmbeddingExtractor:
    def __init__(self, model_id: str, accelerator: Accelerator) -> None:
        self.model_id = model_id
        self.accelerator = accelerator

        # Tokenizer with left padding as recommended for Qwen3-Embedding
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if accelerator.mixed_precision == 'fp16' else (
            torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float32
        )

        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation='flash_attention_2'
            )
            logger.info("Using Flash Attention 2")
        except Exception:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            logger.info("Flash Attention 2 not available; using standard attention")

        self.model = model.eval()

    @torch.no_grad()
    def extract(self, texts: List[str], instruction: Optional[str]) -> np.ndarray:
        # Use max_length of 8192 per Qwen3-Embedding guidance
        max_length = 8192
        if instruction:
            formatted_texts = [f'Instruct: {instruction}\nQuery: {t}' for t in texts]
        else:
            formatted_texts = texts

        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings.float().cpu().numpy().astype(np.float32)


def count_lines(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            n += 1
    return n


def collect_expected_indices(path: str) -> Set[int]:
    indices: Set[int] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'index' in sample:
                indices.add(sample['index'])
    return indices


def _load_indices_from_jsonl(jsonl_path: str, attribute_name: str) -> Tuple[Set[int], int]:
    processed: Set[int] = set()
    total = 0
    if not os.path.exists(jsonl_path):
        return processed, total
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'index' in sample and attribute_name in sample and isinstance(sample[attribute_name], str):
                emb_path = sample[attribute_name]
                if os.path.exists(emb_path):
                    processed.add(sample['index'])
    return processed, total


def load_processed_from_output(output_path: str, attribute_name: str) -> Set[int]:
    processed, _ = _load_indices_from_jsonl(output_path, attribute_name)
    return processed


def load_processed_from_temp_prefix(output_path: str, attribute_name: str) -> Set[int]:
    processed: Set[int] = set()
    out_dir = str(Path(output_path).parent)
    out_stem = Path(output_path).name.replace(".jsonl", "")
    # Read both *.tmp and *.jsonl variants if present
    patterns = [
        f"{out_stem}.rank"  # prefix within directory
    ]
    for entry in os.listdir(out_dir):
        for pfx in patterns:
            if entry.startswith(pfx):
                path = os.path.join(out_dir, entry)
                if os.path.isfile(path):
                    p, _ = _load_indices_from_jsonl(path, attribute_name)
                    processed |= p
    return processed


def process_dataset(
    dataset_path: str,
    model_id: str,
    embedding_dir: str,
    output_path: str,
    batch_size: int,
    extract_key: str,
    instruction: Optional[str],
    mixed_precision: str,
    seed: int,
    overwrite_files: bool = False,
) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])

    # New attribute name derived from embedding_dir basename
    attribute_name = os.path.basename(os.path.normpath(embedding_dir))

    if accelerator.is_main_process:
        Path(embedding_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Attribute name: {attribute_name}")
        logger.info(f"Num processes: {accelerator.num_processes}")

    # Determine already processed indices from existing output and temp files (only count if .npy exists)
    if accelerator.is_main_process:
        if overwrite_files:
            logger.info("Force reprocess enabled, will overwrite all embeddings")
            already_processed_indices = set()
        else:
            already_from_output = load_processed_from_output(output_path, attribute_name)
            already_from_temp = load_processed_from_temp_prefix(output_path, attribute_name)
            already_processed_indices = already_from_output | already_from_temp
            logger.info(f"Found {len(already_from_output)} completed in existing output")
            logger.info(f"Found {len(already_from_temp)} completed in existing temp files")
    else:
        already_processed_indices = set()

    # Broadcast processed set to all ranks
    # Convert to list for torch.distributed broadcast via accelerate
    already_list = list(already_processed_indices)
    already_tensor = torch.tensor(already_list, dtype=torch.long, device=accelerator.device) if len(already_list) > 0 else torch.empty(0, dtype=torch.long, device=accelerator.device)
    # Gather from main and then broadcast using pad to equal sizes
    sizes = accelerator.gather(torch.tensor([already_tensor.numel()], device=accelerator.device))
    max_size = int(sizes.max().item())
    if already_tensor.numel() < max_size:
        pad = torch.full((max_size - already_tensor.numel(),), -1, dtype=torch.long, device=accelerator.device)
        already_tensor = torch.cat([already_tensor, pad], dim=0)
    gathered = accelerator.gather(already_tensor)
    # Filter out padding and build set
    alist: List[int] = [int(x.item()) for x in gathered if int(x.item()) >= 0]
    already_processed_indices = set(alist)

    # Build dataset excluding already completed indices
    dataset = EmbeddingDataset(
        dataset_path=dataset_path,
        extract_key=extract_key,
        skip_indices=already_processed_indices,
    )

    if accelerator.is_main_process:
        logger.info(f"Input file lines: {dataset.total_in_file}")
        logger.info(f"{len(already_processed_indices)} already processed")
        logger.info(f"To process after resume-skip: {len(dataset)}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    # Initialize model and prepare
    extractor = EmbeddingExtractor(model_id, accelerator)
    extractor.model = accelerator.prepare(extractor.model)

    # Prepare temp file for this rank (append for resumability)
    rank = accelerator.process_index
    out_dir = str(Path(output_path).parent)
    out_stem = Path(output_path).name.replace(".jsonl", f".rank{rank}.jsonl.tmp")
    temp_path = os.path.join(out_dir, out_stem)

    # Build a set of indices already written to this temp to avoid duplicates on resume
    written_in_this_temp, _ = _load_indices_from_jsonl(temp_path, attribute_name)
    if accelerator.is_main_process:
        logger.info(f"Rank {rank}: will append to temp {temp_path} (skipping {len(written_in_this_temp)} already in temp)")

    # Iterate and process
    processed_local = 0
    # Progress bar per-rank (only visible on local main process)
    total_batches = len(dataloader)
    pbar = tqdm(total=total_batches, desc=f"Rank {rank}: batches", disable=not accelerator.is_local_main_process)
    with open(temp_path, 'a', encoding='utf-8') as temp_f:
        for batch in dataloader:
            indices: List[int] = batch['indices']
            texts: List[Optional[str]] = batch['texts']
            has_keys: List[bool] = batch['has_extract_key']
            originals: List[Dict[str, Any]] = batch['original_data']

            # Skip anything already written to this rank's temp (resume safety)
            batch_results: List[Dict[str, Any]] = []

            # Collect items that need embedding computation
            indices_to_embed: List[int] = []
            texts_to_embed: List[str] = []
            originals_to_embed: List[Dict[str, Any]] = []

            for idx, text, has_key, orig in zip(indices, texts, has_keys, originals):
                if idx in written_in_this_temp and not overwrite_files:
                    continue

                if not has_key or text is None:
                    # No text available; write original to preserve count
                    batch_results.append(orig)
                    continue

                emb_filename = f"emb_{idx:09d}.npy"
                emb_filepath = os.path.join(embedding_dir, emb_filename)

                if os.path.exists(emb_filepath) and not overwrite_files:
                    # Reuse existing embedding file; just attach path
                    out_sample = dict(orig)
                    out_sample[attribute_name] = emb_filepath
                    batch_results.append(out_sample)
                else:
                    indices_to_embed.append(idx)
                    texts_to_embed.append(text)
                    originals_to_embed.append(orig)

            # Compute embeddings if needed
            if texts_to_embed:
                try:
                    embs = extractor.extract(texts_to_embed, instruction)
                except torch.cuda.OutOfMemoryError:
                    # Fallback to smaller sub-batches
                    torch.cuda.empty_cache()
                    embs_list: List[np.ndarray] = []
                    sub_bs = max(1, len(texts_to_embed) // 4)
                    for s in range(0, len(texts_to_embed), sub_bs):
                        sub_texts = texts_to_embed[s:s+sub_bs]
                        sub_embs = extractor.extract(sub_texts, instruction)
                        embs_list.append(sub_embs)
                    embs = np.concatenate(embs_list, axis=0)

                # Save and assemble outputs
                for i, idx in enumerate(indices_to_embed):
                    emb = embs[i]
                    emb_filename = f"emb_{idx:09d}.npy"
                    emb_filepath = os.path.join(embedding_dir, emb_filename)
                    np.save(emb_filepath, emb)
                    out_sample = dict(originals_to_embed[i])
                    out_sample[attribute_name] = emb_filepath
                    batch_results.append(out_sample)

            # Write batch results
            for sample in batch_results:
                temp_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                processed_local += 1

            if processed_local and processed_local % 500 == 0 and accelerator.is_local_main_process:
                temp_f.flush()
                logger.info(f"Rank {rank}: processed {processed_local} samples (temp flushed)")
            # Update batch progress
            pbar.update(1)

    pbar.close()

    if accelerator.is_main_process:
        logger.info(f"Rank {rank}: finished with {processed_local} newly written samples")

    accelerator.wait_for_everyone()

    # Merge (main process only)
    if accelerator.is_main_process:
        logger.info("Merging per-rank temp files into final output...")
        out_dir = Path(output_path).parent
        temp_prefix = Path(output_path).name.replace(".jsonl", ".rank")

        # Collect all candidates: existing output (if any) + all temp files
        merged_by_index: Dict[int, Dict[str, Any]] = {}

        # 1) Existing output (keep only valid entries with existing .npy)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'index' in sample:
                        if attribute_name in sample and isinstance(sample[attribute_name], str) and os.path.exists(sample[attribute_name]):
                            merged_by_index[sample['index']] = sample

        # 2) All temp files
        for entry in os.listdir(out_dir):
            if entry.startswith(temp_prefix):
                tpath = os.path.join(out_dir, entry)
                if not os.path.isfile(tpath):
                    continue
                with open(tpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            sample = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if 'index' in sample:
                            merged_by_index[sample['index']] = sample

        # Deterministic order by index
        final_samples = [merged_by_index[k] for k in sorted(merged_by_index.keys())]

        # Validate count equals input dataset lines (by indices present in input)
        expected_indices = collect_expected_indices(dataset_path)
        if len(final_samples) != len(expected_indices):
            missing = sorted(expected_indices - set(merged_by_index.keys()))[:20]
            extra = sorted(set(merged_by_index.keys()) - expected_indices)[:20]
            logger.error(
                f"Final sample count mismatch: got {len(final_samples)}, expected {len(expected_indices)}.\n"
                f"First missing indices: {missing}\nFirst extra indices: {extra}"
            )
            raise RuntimeError("Final output does not match input dataset size by index set")

        # Write final output
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in final_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Wrote {len(final_samples)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen embeddings with Accelerate (resumable, per-rank temp files)."
    )
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--embedding_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_key', type=str, default='transcription')
    parser.add_argument('--instruction', type=str, default='Represent this speech transcript for semantic similarity search')
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite_files', action='store_true', help='Force re-extraction of all embeddings, ignoring existing files')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    set_seed(args.seed)

    logger.info("Configuration:")
    logger.info(f"  model_id: {args.model_id}")
    logger.info(f"  dataset_path: {args.dataset_path}")
    logger.info(f"  embedding_dir: {args.embedding_dir}")
    logger.info(f"  output_path: {args.output_path}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  extract_key: {args.extract_key}")
    logger.info(f"  mixed_precision: {args.mixed_precision}")
    logger.info(f"  overwrite_files: {args.overwrite_files}")

    process_dataset(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        embedding_dir=args.embedding_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        extract_key=args.extract_key,
        instruction=args.instruction,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        overwrite_files=args.overwrite_files,
    )


if __name__ == '__main__':
    main()
