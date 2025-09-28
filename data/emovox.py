#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from tqdm import tqdm
from data.utils import fenced_json_str_to_json

load_dotenv()


"""
Example usage:
  accelerate launch --multi_gpu data/emovox.py \
    --input /mnt/vast/data/speech/peoplespeech/test/dataset_og.jsonl \
    --output /mnt/vast/data/speech/peoplespeech/test/dataset_clean.jsonl \
    --batch_size 32
"""


SYSTEM_PROMPT = """
You are a speech understanding model that analyzes audio recordings of human speech.

Your task is to:
1. Transcribe the utterance.
2. Infer the speaker’s overall affective state based on acoustic and paralinguistic cues throughout the utterance.
3. Return this affective context in a structured and interpretable format, designed for use by a large language model (LLM) to guide emotionally intelligent responses.

Only include emotional features that are **clearly evident** in the voice signal. If any element cannot be confidently inferred, return `"unsure"`.

Return the following JSON:

{
  "transcript": "<full_text_transcription>",
  "speaker_demographics": {
    "gender": {
      "label": "<male | female | unsure>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    },
    "age_group": {
      "label": "<child | teen | young_adult | adult | senior | unsure>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    },
    "language": {
      "label": "<language of the speaker>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    },
    "accent_region": {
      "label": "<e.g., 'American Southern', 'British RP', 'Indian English', or 'unsure'>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    },
    "speaking_style": {
      "label": "<formal | casual | hesitant | assertive | unsure>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    },
    "speech_pattern": {
      "label": "<clear | disfluent | monotonic | impaired | unsure>",
      "confidence": <float between 0 and 1 indicating confidence in the label>
    }
  },
  "affective_context": {
    "valence": <float between -1 and 1, or "unsure">,
    "arousal": <float between 0 and 1, or "unsure">,
    "emotions": [
      {
        "label": "<emotion name, e.g., 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral', 'trust', 'anticipation', 'empathy', 'interest', 'awe', 'frustration', 'sarcasm', 'confusion', 'resignation', 'pride', 'embarrassment', 'gratitude', 'shame', 'guilt', 'skepticism', 'jealousy'>",
        "score": <float between 0 and 1 indicating presence or intensity>,
        "confidence": <float between 0 and 1 indicating confidence in the score>,
        "description": "<A concise description of the presence or intensity of the emotion>"
      },
      ...
    ],
    "emotional_description": "<1–2 sentence natural language summary of how the speaker sounds and feels, e.g., 'The speaker sounds frustrated but is trying to stay composed.'>"
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fault-tolerant batched inference with Accelerate")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4)")
    return parser.parse_args()


def count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def try_parse_affect_from_text(text: str) -> Optional[Any]:
    fenced = fenced_json_str_to_json(text)
    if fenced is None:
        return None
    try:
        value = json.loads(fenced)
    except Exception:
        return None
    if isinstance(value, (dict, list)):
        return value
    return None


def build_worklist(input_path: str, show_progress: bool = False, total_hint: Optional[int] = None) -> Tuple[List[Tuple[int, Optional[Dict[str, Any]], bool]], int]:
    items: List[Tuple[int, Optional[Dict[str, Any]], bool]] = []
    total = 0
    iterator = open(input_path, "r", encoding="utf-8")
    pbar = tqdm(total=total_hint, disable=not show_progress, desc="scan", unit="lines")
    try:
        for idx, line in enumerate(iterator):
            total += 1
            line = line.rstrip("\n")
            try:
                sample = json.loads(line)
            except Exception:
                # Invalid JSON – copy through unchanged later
                items.append((idx, None, False))
                if show_progress:
                    pbar.update(1)
                continue
            needs = not isinstance(sample, dict) or ("affect" not in sample)
            items.append((idx, sample, needs))
            if show_progress:
                pbar.update(1)
    finally:
        if show_progress:
            pbar.close()
        iterator.close()
    return items, total


class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, shard_items: List[Tuple[int, Dict[str, Any]]]):
        self.shard_items = shard_items

    def __len__(self) -> int:
        return len(self.shard_items)

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        line_idx, sample = self.shard_items[idx]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": sample["audio"]},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]
        return line_idx, sample, conversation


def collate_batch(batch: List[Tuple[int, Dict[str, Any], Dict[str, Any]]]):
    line_indices = [b[0] for b in batch]
    samples = [b[1] for b in batch]
    conversations = [b[2] for b in batch]
    return line_indices, samples, conversations


def fsync_flush(fh) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def main():
    args = parse_args()
    accelerator = Accelerator()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    if is_main:
        accelerator.print(f"Launching Accelerate with world_size={world_size}")

    # Early exit if final output already complete
    input_line_count = count_lines(args.input)
    done_flag = False
    if os.path.exists(args.output):
        try:
            output_line_count = count_lines(args.output)
            if output_line_count == input_line_count:
                done_flag = True
        except Exception:
            done_flag = False
    obj_list = [done_flag]
    broadcast_object_list(obj_list)
    done_flag = obj_list[0]
    if done_flag:
        if is_main:
            accelerator.print("Output already complete; exiting.")
        return

    # Rank 0 builds worklist and broadcasts
    items: List[Tuple[int, Optional[Dict[str, Any]], bool]]
    if is_main:
        items, _ = build_worklist(args.input, show_progress=True, total_hint=input_line_count)
    else:
        items = []  # type: ignore
    obj_list = [items]
    broadcast_object_list(obj_list)
    items = obj_list[0]

    # Shard by line_idx across ranks, filter to needs_inference
    shard_all = [items[i] for i in range(rank, len(items), world_size)]
    shard_pairs: List[Tuple[int, Dict[str, Any]]] = [
        (line_idx, sample) for (line_idx, sample, needs) in shard_all if needs and isinstance(sample, dict)
    ]

    # Paths for per-rank temp and final
    rank_tmp = f"{args.output}.rank{rank}.tmp"

    # Load any existing per-rank results and mark done items (only those with valid affect)
    done_map: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(rank_tmp):
        try:
            with open(rank_tmp, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            continue
                        li = rec.get("line_idx", None)
                        sm = rec.get("sample", None)
                        if isinstance(li, int) and isinstance(sm, dict):
                            val = sm.get("affect", None)
                            if isinstance(val, (dict, list)):
                                done_map[li] = sm
                    except Exception:
                        continue
        except Exception:
            pass

    # Pending items exclude already-done ones
    pending_pairs: List[Tuple[int, Dict[str, Any]]] = [
        (li, sm) for (li, sm) in shard_pairs if li not in done_map
    ]

    # Ensure the per-rank temp file exists so users can see progress
    if not os.path.exists(rank_tmp):
        open(rank_tmp, "a", encoding="utf-8").close()

    if len(pending_pairs) == 0:
        accelerator.print(f"Rank {rank}: nothing to do (reusing {rank_tmp}).")
    else:
        # Load model/processor per-rank
        model_path = os.getenv("EMOVOX_DIR", None)
        if model_path is None:
            raise RuntimeError("EMOVOX_DIR environment variable not set")

        processor = AutoProcessor.from_pretrained(model_path)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        model.to(accelerator.device)
        model.eval()

        dataset = ShardDataset(pending_pairs)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=max(0, args.num_workers),
            collate_fn=collate_batch,
            pin_memory=True,
        )

        # Per-rank progress bar over batches
        pbar = tqdm(total=len(dataloader), desc=f"rank{rank} infer", position=rank + 1, leave=True)

        writes_since_flush = 0
        flush_every = 100
        with open(rank_tmp, "a", encoding="utf-8") as out_f:
            with torch.inference_mode():
                for batch_idx, (line_indices, samples, conversations) in enumerate(dataloader):
                    inputs = processor.apply_chat_template(conversations)
                    inputs = inputs.to(accelerator.device, dtype=torch.bfloat16)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        top_p=1.0,
                        repetition_penalty=1.0,
                    )

                    # Remove prompt tokens uniformly (keeps parity with original script)
                    try:
                        offset = inputs.input_ids.shape[1]
                        decoded = processor.batch_decode(
                            outputs[:, offset:], skip_special_tokens=True
                        )
                    except Exception:
                        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

                    for li, sample, text in zip(line_indices, samples, decoded):
                        affect = try_parse_affect_from_text(text)
                        if affect is not None:
                            sample["affect"] = affect
                        record = {"line_idx": int(li), "sample": sample}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        writes_since_flush += 1
                        if writes_since_flush % flush_every == 0:
                            fsync_flush(out_f)
                    pbar.update(1)
            fsync_flush(out_f)
        pbar.close()
        accelerator.print(f"Rank {rank}: appended results to {rank_tmp}")

    accelerator.wait_for_everyone()

    # Rank 0: merge and final verify
    if is_main:
        # Build map line_idx -> processed sample
        processed: Dict[int, Dict[str, Any]] = {}
        for r in range(world_size):
            r_path = f"{args.output}.rank{r}.tmp"
            if not os.path.exists(r_path):
                continue
            with open(r_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            continue
                        li = rec.get("line_idx", None)
                        sm = rec.get("sample", None)
                        if isinstance(li, int) and isinstance(sm, dict):
                            processed[li] = sm
                    except Exception:
                        continue

        # Step 7: merge in strict input order, writing to output, copying unchanged lines when needed
        writes_since_flush = 0
        flush_every = 1000
        with open(args.input, "r", encoding="utf-8") as in_f, open(args.output, "w", encoding="utf-8") as out_f:
            pbar_merge = tqdm(total=input_line_count, disable=not is_main, desc="merge", unit="lines", position=0)
            for idx, line in enumerate(in_f):
                if idx in processed:
                    out_f.write(json.dumps(processed[idx], ensure_ascii=False) + "\n")
                else:
                    # Copy original line unchanged (handles invalid JSON and pre-existing affect)
                    if not line.endswith("\n"):
                        line = line + "\n"
                    out_f.write(line)
                writes_since_flush += 1
                if writes_since_flush % flush_every == 0:
                    fsync_flush(out_f)
                pbar_merge.update(1)
            fsync_flush(out_f)
            pbar_merge.close()

        # Step 8: final verification with de-dup by "index" and atomic replace
        verified_tmp = f"{args.output}.tmp"
        verify_total = count_lines(args.output)

        # First pass: determine first occurrence positions and first valid-affect positions per index
        first_pos_by_index: Dict[Any, int] = {}
        first_valid_pos_by_index: Dict[Any, int] = {}
        with open(args.output, "r", encoding="utf-8") as src:
            for pos, line in enumerate(src):
                raw = line.rstrip("\n")
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if "index" not in obj:
                    continue
                idx_val = obj.get("index")
                if idx_val not in first_pos_by_index:
                    first_pos_by_index[idx_val] = pos
                if "affect" in obj:
                    val = obj["affect"]
                    valid = False
                    if isinstance(val, (dict, list)):
                        try:
                            _ = json.loads(json.dumps(val))
                            valid = True
                        except Exception:
                            valid = False
                    if valid and (idx_val not in first_valid_pos_by_index):
                        first_valid_pos_by_index[idx_val] = pos

        # Second pass: write output applying affect validation and index de-dup policy
        with open(args.output, "r", encoding="utf-8") as src, open(verified_tmp, "w", encoding="utf-8") as dst:
            pbar_verify = tqdm(total=verify_total, disable=not is_main, desc="verify", unit="lines", position=0)
            for pos, line in enumerate(src):
                raw = line.rstrip("\n")
                try:
                    obj = json.loads(raw)
                except Exception:
                    # Keep line unchanged if not valid JSON
                    dst.write(raw + "\n")
                    pbar_verify.update(1)
                    continue
                if isinstance(obj, dict):
                    # Validate affect JSON and remove if invalid
                    if "affect" in obj:
                        val = obj["affect"]
                        if isinstance(val, (dict, list)):
                            try:
                                _ = json.loads(json.dumps(val))
                            except Exception:
                                obj.pop("affect", None)
                        else:
                            obj.pop("affect", None)

                    # De-dup by index if present
                    if "index" in obj:
                        idx_val = obj.get("index")
                        keep_pos = first_valid_pos_by_index.get(idx_val)
                        if keep_pos is None:
                            keep_pos = first_pos_by_index.get(idx_val)
                        if keep_pos is not None and pos != keep_pos:
                            # Skip duplicate entries for this index
                            pbar_verify.update(1)
                            continue

                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar_verify.update(1)
            fsync_flush(dst)
            pbar_verify.close()

        # Atomically replace
        os.replace(verified_tmp, args.output)
        accelerator.print("Final output verified and written.")


if __name__ == "__main__":
    main()
