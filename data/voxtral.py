#!/usr/bin/env python3
"""
Extract Voxtral audio embeddings from a dataset and save them as .npy files.
Supports multi-GPU inference using HuggingFace Accelerate for efficient batch processing.
"""

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.voxtral.processing_voxtral import VoxtralProcessor
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, broadcast_object_list, gather_object

from model.whispa import WhiSPAModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


"""
Example usage:
accelerate launch --num_processes 8 --mixed_precision fp16 data/voxtral.py \
    --model_path /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --dataset_path /mnt/vast/data/speech/gigaspeech/data/data/dataset_clean.jsonl \
    --embedding_dir /mnt/vast/data/speech/gigaspeech/data/data/vox_3072_audio \
    --output_path /mnt/vast/data/speech/gigaspeech/data/data/dataset_clean.vox_3072_audio.jsonl \
    --batch_size 32 \
    --num_workers 128 \
    --overwrite_files \
&& accelerate launch --num_processes 8 --mixed_precision fp16 data/voxtral.py \
    --model_path /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Small-24B \
    --dataset_path /mnt/vast/data/speech/gigaspeech/data/data/dataset_clean.vox_3072_audio.jsonl \
    --embedding_dir /mnt/vast/data/speech/gigaspeech/data/data/vox_5120_audio \
    --output_path /mnt/vast/data/speech/gigaspeech/data/data/dataset_clean.vox_5120_audio.jsonl \
    --batch_size 32 \
    --num_workers 128 \
    --overwrite_files
"""


class AudioDataset(Dataset):
    """Custom dataset for loading audio file paths from JSONL file."""
    
    def __init__(self, dataset_path: str, processed_indices: set = None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the JSONL file containing the dataset
            processed_indices: Set of indices to skip (already processed)
        """
        self.samples = []
        processed_indices = processed_indices or set()
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                if 'index' in sample and 'audio' in sample:
                    if sample['index'] not in processed_indices:
                        # Check if audio file exists
                        if os.path.exists(sample['audio']):
                            self.samples.append(sample)
                        else:
                            logger.warning(f"Audio file not found: {sample['audio']}")
                else:
                    logger.warning(f"Skipping sample without required fields: {sample}")
        
        logger.info(f"Loaded {len(self.samples)} samples to process from {dataset_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'index': sample['index'],
            'audio': sample['audio'],
            'original_data': sample  # Keep all original fields
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching."""
    indices = [item['index'] for item in batch]
    audio_paths = [item['audio'] for item in batch]
    original_data = [item['original_data'] for item in batch]
    
    return {
        'indices': indices,
        'audios': audio_paths,
        'original_data': original_data
    }


class VoxtralEncoder:
    """Class to handle Voxtral audio encoding with Accelerate multi-GPU support."""
    
    def __init__(self, model_path: str, accelerator: Accelerator, language: str = "en", dtype: torch.dtype = None):
        """
        Initialize the Voxtral encoder.
        
        Args:
            model_path: Local filesystem path to the WhiSPA checkpoint directory
            accelerator: Accelerator instance for distributed processing
            language: Language code for encoding
            dtype: Model dtype (default: based on mixed precision setting)
        """
        self.model_path = model_path
        self.accelerator = accelerator
        self.language = language
        
        # Determine dtype based on mixed precision setting
        if dtype is None:
            if accelerator.mixed_precision == "fp16":
                dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        
        # Pre-cache Voxtral assets (processor and backbone) on main process so all ranks can load from cache
        try:
            cfg_path = os.path.join(model_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            backbone_model_id = cfg_dict.get("backbone_model_id")
            if not backbone_model_id:
                raise ValueError("`backbone_model_id` not found in config.json")
            if accelerator.is_main_process:
                logger.info(f"Pre-caching Voxtral assets for backbone: {backbone_model_id}")
                # Ensure processor and model weights are present in the local HF cache
                VoxtralProcessor.from_pretrained(backbone_model_id)
                VoxtralForConditionalGeneration.from_pretrained(backbone_model_id)
            accelerator.wait_for_everyone()
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(f"Could not pre-cache Voxtral assets: {e}")
            accelerator.wait_for_everyone()

        # Load WhiSPA model from local checkpoint
        logger.info(f"Loading WhiSPA model from local path: {model_path}")
        self.model = WhiSPAModel.from_pretrained_local(model_path).eval()
        self.model.to(device=accelerator.device, dtype=dtype)
        # Ensure inference stage
        if hasattr(self.model, 'config'):
            self.model.config.stage = 'inference'
            self.model.config.device = accelerator.device
        
        logger.info(f"Model loaded successfully on device: {accelerator.device}")
        logger.info(f"Using dtype: {dtype}")
    
    @torch.no_grad()
    def encode_batch(self, audio_paths: List[str]) -> np.ndarray:
        """
        Encode a batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            numpy array of embeddings (batch_size, hidden_dim)
        """
        # Returns pooled embeddings of shape (batch_size, hidden_dim)
        embeddings = self.model.encode(audio_paths, language=self.language)
        
        # Convert to CPU and numpy
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        return embeddings_np


def process_dataset(
    dataset_path: str,
    model_path: str,
    embedding_dir: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    language: str = "en",
    overwrite_files: bool = False,
    mixed_precision: str = "fp16",
    dtype: str = None
):
    """
    Process the entire dataset to extract Voxtral audio embeddings using Accelerate.
    Supports resuming from previous runs and multi-GPU processing.
    
    Args:
        dataset_path: Path to input JSONL file with audio paths
        model_path: Local filesystem path to the WhiSPA checkpoint directory
        embedding_dir: Directory to save embedding files
        output_path: Path to output JSONL file
        batch_size: Batch size per GPU for processing
        num_workers: Number of workers for data loading
        language: Language code for audio encoding
        overwrite_files: If True, ignore existing output and reprocess all samples
        mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
        dtype: Model dtype override ('float32', 'float16', 'bfloat16', or None for auto)
    """
    # Initialize accelerator with custom kwargs for better performance
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Only main process should handle file I/O for output
    if accelerator.is_main_process:
        # Create embedding directory if it doesn't exist
        Path(embedding_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the attribute name from the final directory name of embedding_dir
    attribute_name = os.path.basename(os.path.normpath(embedding_dir))
    
    if accelerator.is_main_process:
        logger.info(f"Using attribute name: {attribute_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Process index: {accelerator.process_index}")
        logger.info(f"Language: {language}")
    
    # Check for existing processed samples (only on main process)
    processed_indices = set()
    if accelerator.is_main_process:
        if overwrite_files:
            logger.info(f"Force reprocess enabled, will overwrite all embeddings")
            if os.path.exists(output_path):
                logger.info(f"Will overwrite existing output file: {output_path}")
        elif os.path.exists(output_path):
            logger.info(f"Found existing output file: {output_path}")
            logger.info("Reading already processed samples...")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if 'index' in sample and attribute_name in sample:
                            processed_indices.add(sample['index'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Found {len(processed_indices)} already processed samples")
    
    # Wait for main process to read the file
    accelerator.wait_for_everyone()
    
    # Broadcast processed indices to all processes
    if accelerator.is_main_process:
        processed_indices_list = list(processed_indices)
    else:
        processed_indices_list = []
    
    # Broadcast from main to all other processes
    objects = [processed_indices_list]
    broadcast_object_list(objects, from_process=0)
    processed_indices = set(objects[0])
        
    # Initialize dataset
    logger.info("Loading dataset...")
    dataset = AudioDataset(dataset_path, processed_indices)
    if len(dataset) == 0:
        if accelerator.is_main_process:
            logger.info("All samples have already been processed!")
        return
    if accelerator.is_main_process:
        logger.info(f"Processing {len(dataset)} remaining samples")
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    # Parse dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        None: None
    }
    model_dtype = dtype_map.get(dtype, None)
    
    # Initialize Voxtral encoder
    encoder = VoxtralEncoder(model_path, accelerator, language, dtype=model_dtype)
    
    dataloader = accelerator.prepare(dataloader)
    
    # Open output file for on-the-fly writing (only main process)
    output_file = None
    samples_written = 0
    existing_indices = set()
    
    if accelerator.is_main_process:
        # Determine write mode
        mode = 'w' if overwrite_files or not os.path.exists(output_path) else 'a'
        
        # Get existing indices if appending
        if mode == 'a' and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if 'index' in sample:
                            existing_indices.add(sample['index'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Found {len(existing_indices)} existing samples in output file")
        
        # Open file for writing
        output_file = open(output_path, mode, encoding='utf-8', buffering=1)  # Line buffering
        logger.info(f"Opened output file in '{mode}' mode: {output_path}")
    
    # Process batches
    progress_bar = tqdm(
        total=len(dataloader),
        desc=f"Encoding audio (GPU {accelerator.process_index})",
        disable=not accelerator.is_local_main_process
    )
        
    batch_results_buffer = []
    flush_interval = 5  # Flush every N batches
    
    for batch_idx, batch in enumerate(dataloader):
        indices = batch['indices']
        audio_paths = batch['audios']
        original_data = batch['original_data']
        
        # Encode audio batch - returns (batch_size, hidden_dim)
        embeddings = encoder.encode_batch(audio_paths)
        
        # Save embeddings and prepare output data
        batch_results = []
        for i, (idx, orig_data) in enumerate(zip(indices, original_data)):
            # Extract embedding for this sample
            emb = embeddings[i]
            
            # Create unique filename based on index
            emb_filename = f"vox_{idx:09d}.npy"
            emb_filepath = os.path.join(embedding_dir, emb_filename)
            
            # Check if embedding already exists (skip check if overwrite_files is True)
            if overwrite_files or not os.path.exists(emb_filepath):
                # Save embedding
                np.save(emb_filepath, emb)
            
            # Add embedding filepath to original data
            output_sample = orig_data.copy()
            output_sample[attribute_name] = emb_filepath
            batch_results.append(output_sample)
        
        batch_results_buffer.extend(batch_results)
        
        # Periodically gather and write results
        if (batch_idx + 1) % flush_interval == 0 or (batch_idx + 1) == len(dataloader):
            # Gather results from all processes
            all_batch_results = gather_object(batch_results_buffer)
            
            # Main process writes to file
            if accelerator.is_main_process and output_file:
                # Flatten results from all processes
                flattened_batch = []
                if all_batch_results and isinstance(all_batch_results[0], list):
                    for process_results in all_batch_results:
                        flattened_batch.extend(process_results)
                else:
                    flattened_batch = all_batch_results if all_batch_results else []
                
                # Sort by index to maintain order
                flattened_batch.sort(key=lambda x: x['index'])
                
                # Write results to file
                for sample in flattened_batch:
                    # Skip if already exists when appending
                    if sample['index'] not in existing_indices:
                        output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        existing_indices.add(sample['index'])
                        samples_written += 1
                
                # Flush to ensure data is written to disk
                output_file.flush()
                logger.debug(f"Wrote {len(flattened_batch)} samples to output (total: {samples_written})")
            
            # Clear buffer for next batch
            batch_results_buffer = []
            
            # Wait for main process to finish writing
            accelerator.wait_for_everyone()
        
        progress_bar.update(1)
        
        # Clear cache periodically
        if accelerator.is_local_main_process and torch.cuda.is_available():
            if progress_bar.n % 10 == 0:
                torch.cuda.empty_cache()
    
    progress_bar.close()
    
    # Final flush if there are remaining results
    if batch_results_buffer:
        all_batch_results = gather_object(batch_results_buffer)
        
        if accelerator.is_main_process and output_file:
            flattened_batch = []
            if all_batch_results and isinstance(all_batch_results[0], list):
                for process_results in all_batch_results:
                    flattened_batch.extend(process_results)
            else:
                flattened_batch = all_batch_results if all_batch_results else []
            
            flattened_batch.sort(key=lambda x: x['index'])
            
            for sample in flattened_batch:
                if sample['index'] not in existing_indices:
                    output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    existing_indices.add(sample['index'])
                    samples_written += 1
            
            output_file.flush()
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Close output file and log final statistics
    if accelerator.is_main_process:
        if output_file:
            output_file.close()
        
        logger.info(f"Successfully processed and wrote {samples_written} new samples")
        logger.info(f"Embeddings saved to: {embedding_dir}")
        logger.info(f"Output JSONL saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Voxtral audio embeddings from a dataset using multi-GPU acceleration"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local WhiSPA checkpoint directory"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the input JSONL file containing audio paths"
    )
    
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory to save embedding .npy files"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSONL file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for processing (default: 16)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for audio encoding (default: en)"
    )
    
    parser.add_argument(
        "--overwrite_files",
        action="store_true",
        help="Force reprocessing of all samples, ignoring existing output"
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (default: fp16)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16", None],
        help="Model dtype override (default: auto based on mixed_precision)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Log configuration (only from main process, handled by accelerate)
    logger.info("Configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Dataset path: {args.dataset_path}")
    logger.info(f"  Embedding directory: {args.embedding_dir}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Batch size per GPU: {args.batch_size}")
    logger.info(f"  Number of workers: {args.num_workers}")
    logger.info(f"  Language: {args.language}")
    logger.info(f"  Force reprocess: {args.overwrite_files}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Model dtype: {args.dtype if args.dtype else 'auto'}")
    
    if torch.cuda.is_available():
        logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("  Running on CPU")
    
    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        embedding_dir=args.embedding_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        language=args.language,
        overwrite_files=args.overwrite_files,
        mixed_precision=args.mixed_precision,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()