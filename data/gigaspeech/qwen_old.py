#!/usr/bin/env python3
"""
Extract Qwen embeddings from a dataset and save them as .npy files.
Supports multi-GPU inference using HuggingFace Accelerate for efficient batch processing.
Modified to use per-process temporary files to avoid distributed gathering timeouts.
"""

import os
import sys

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, broadcast_object_list
from train.utils import last_token_pool
import warnings
warnings.filterwarnings("ignore")
import tempfile
import glob
import time
import re
import signal
from contextlib import contextmanager

# Increase NCCL timeout to 10 minutes for large datasets
os.environ.setdefault('NCCL_TIMEOUT', '360')
# Enable NCCL async error handling
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
# Disable P2P for stability
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
# Set OMP threads to avoid oversubscription
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


@contextmanager
def timeout(seconds):
    """Context manager for setting a timeout on operations"""
    # Set the signal handler and a timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Restore the original handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


"""
Example usage:
accelerate launch --num_processes 8 --mixed_precision fp16 data/gigaspeech/qwen.py \
    --model_id Qwen/Qwen3-Embedding-8B \
    --dataset_path /mnt/vast/data/speech/gigaspeech/data/data/gigaspeech_clean.affect.jsonl \
    --embedding_dir /mnt/vast/data/speech/gigaspeech/data/data/qwen_4096_affect \
    --output_path /mnt/vast/data/speech/gigaspeech/data/data/gigaspeech_clean.qwen_4096_affect.jsonl \
    --extract_key affect \
    --batch_size 32 \
    --num_workers 4

Note: This implementation uses last_token_pool as recommended for Qwen3-Embedding models.
The --instruction parameter is optional and should be used when you want task-specific embeddings.
The --extract_key parameter specifies which field to extract embeddings from (default: transcription).
"""

class TextDataset(Dataset):
    """Custom dataset for loading text data from JSONL file."""
    
    def __init__(self, dataset_path: str, extract_key: str = 'transcription', processed_indices: set = None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the JSONL file containing the dataset
            extract_key: The JSON key to extract text from for embedding
            processed_indices: Set of indices to skip (already processed)
        """
        self.samples = []
        self.extract_key = extract_key
        processed_indices = processed_indices or set()
        
        total_samples = 0
        samples_with_key = 0
        samples_without_key = 0
        skipped_already_processed = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                total_samples += 1
                
                # Skip samples without required fields
                if 'index' not in sample:
                    continue
                    
                # Skip already processed samples
                if sample['index'] in processed_indices:
                    skipped_already_processed += 1
                    continue
                
                # Mark whether sample has the extract_key
                has_extract_key = extract_key in sample
                sample['_has_extract_key'] = has_extract_key
                
                if has_extract_key:
                    samples_with_key += 1
                else:
                    samples_without_key += 1
                    
                self.samples.append(sample)
        
        logger.info(f"Dataset loading summary:")
        logger.info(f"  Total samples in file: {total_samples}")
        logger.info(f"  Samples with '{extract_key}' field: {samples_with_key}")
        logger.info(f"  Samples without '{extract_key}' field: {samples_without_key}")
        logger.info(f"  Skipped (already processed): {skipped_already_processed}")
        logger.info(f"  Total samples to process: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {
            'index': sample['index'],
            'original_data': sample,  # Keep all original fields
            'has_extract_key': sample['_has_extract_key']
        }
        
        # Only include text if the sample has the extract_key
        if sample['_has_extract_key']:
            result['text'] = sample[self.extract_key]
        else:
            result['text'] = None  # No text to extract
            
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching."""
    indices = [item['index'] for item in batch]
    texts = [item['text'] for item in batch]
    original_data = [item['original_data'] for item in batch]
    has_extract_key = [item['has_extract_key'] for item in batch]
    
    return {
        'indices': indices,
        'texts': texts,
        'original_data': original_data,
        'has_extract_key': has_extract_key
    }


class EmbeddingExtractor:
    """Class to handle embedding extraction with Accelerate multi-GPU support."""
    
    def __init__(self, model_id: str, accelerator: Accelerator):
        """
        Initialize the embedding extractor.
        
        Args:
            model_id: HuggingFace model ID
            accelerator: Accelerator instance for distributed processing
        """
        self.model_id = model_id
        self.accelerator = accelerator
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_id}")
        # Use left padding as recommended for Qwen3-Embedding models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Load model with appropriate dtype
        dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
        
        # Try to use flash_attention_2 if available for better performance
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2"
            )
            logger.info("Using Flash Attention 2 for better performance")
        except:
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            logger.info("Flash Attention 2 not available, using standard attention")
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model is prepared by accelerator, no need for manual device placement
        self.model.eval()
        
        logger.info(f"Model loaded successfully on device: {accelerator.device}")
    
    @torch.no_grad()
    def extract_embeddings(self, texts: List[str], instruction: str = None) -> np.ndarray:
        """
        Extract embeddings for a list of texts using last token pooling.
        
        Args:
            texts: List of text strings
            instruction: Optional instruction for the embedding task
        
        Returns:
            numpy array of embeddings
        """
        # Use max_length of 8192 as recommended for Qwen3-Embedding
        max_length = 8192
        
        # Apply instruction formatting if provided
        if instruction:
            formatted_texts = [f'Instruct: {instruction}\nQuery: {text}' for text in texts]
        else:
            formatted_texts = texts
        
        # Tokenize with truncation and padding
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device (handled by accelerator)
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Use last token pooling as recommended for Qwen3-Embedding
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        
        # Convert to CPU and numpy
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        
        return embeddings


def process_dataset(
    dataset_path: str,
    model_id: str,
    embedding_dir: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    extract_key: str = 'transcription',
    overwrite_files: bool = False,
    mixed_precision: str = "fp16",
    instruction: str = None,
    seed: int = 42
):
    """
    Process the entire dataset to extract embeddings using Accelerate.
    Uses per-process temporary files to avoid distributed gathering timeouts.
    
    Args:
        dataset_path: Path to input JSONL file
        model_id: HuggingFace model ID
        embedding_dir: Directory to save embedding files
        output_path: Path to output JSONL file
        batch_size: Batch size per GPU for processing
        num_workers: Number of workers for data loading
        extract_key: The JSON key to extract text from for embedding
        overwrite_files: If True, ignore existing output and reprocess all samples
        mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
        instruction: Optional instruction for the embedding task
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
    
    # Check for existing processed samples (only on main process)
    processed_indices = set()
    if accelerator.is_main_process:
        if not overwrite_files and os.path.exists(output_path):
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
        elif overwrite_files and os.path.exists(output_path):
            logger.info(f"Force reprocess enabled, overwriting {output_path}")
    
    # Synchronize reading of existing processed indices
    try:
        accelerator.wait_for_everyone()
        
        # Broadcast processed indices to all processes
        # Convert set to list for gathering, then back to set
        if accelerator.is_main_process:
            processed_indices_list = list(processed_indices)
        else:
            processed_indices_list = []
        
        # Broadcast from main to all other processes
        objects = [processed_indices_list]
        broadcast_object_list(objects, from_process=0)
        processed_indices = set(objects[0])
    except Exception as e:
        logger.warning(f"Process {accelerator.process_index}: Error during initial sync: {e}")
        # Continue with empty processed_indices if sync fails
        processed_indices = set()
    
    # Initialize dataset
    dataset = TextDataset(dataset_path, extract_key, processed_indices)
    
    if len(dataset) == 0:
        if accelerator.is_main_process:
            logger.info("All samples have already been processed!")
        return
    
    if accelerator.is_main_process:
        logger.info(f"Processing {len(dataset)} remaining samples")
    
    # Create dataloader with distributed sampler to ensure even distribution
    from torch.utils.data.distributed import DistributedSampler
    
    # Use DistributedSampler for proper data distribution
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
        seed=seed
    )
    # Set epoch to ensure consistent data distribution
    sampler.set_epoch(0)
    
    # Create dataloader with careful settings to avoid hanging
    # Use 0 workers to avoid multiprocessing issues that cause hanging
    # This is slower but more reliable, especially near the end of datasets
    actual_num_workers = 0  # Force single-threaded to avoid hanging
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=actual_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # Disable since we're using 0 workers
        persistent_workers=False,
        timeout=0,  # Not needed with 0 workers
        prefetch_factor=None,  # Not applicable with 0 workers
        drop_last=False
    )
    
    logger.info(f"Process {accelerator.process_index}: Using {actual_num_workers} workers for DataLoader (single-threaded for reliability)")
    
    # Initialize model and prepare with accelerator
    extractor = EmbeddingExtractor(model_id, accelerator)
    
    # Prepare model only (dataloader already has distributed sampler)
    model = accelerator.prepare(extractor.model)
    extractor.model = model
    
    # Create temporary output file for this process
    temp_dir = Path(output_path).parent
    temp_prefix = f"{Path(output_path).stem}_temp_rank{accelerator.process_index}_"
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        prefix=temp_prefix,
        suffix='.jsonl',
        dir=temp_dir,
        delete=False
    )
    temp_filename = temp_file.name
    logger.info(f"Process {accelerator.process_index} writing to temporary file: {temp_filename}")
    
    # Log dataloader info  
    logger.info(f"Process {accelerator.process_index}: DataLoader has {len(dataloader)} batches")
    
    # Calculate expected samples for this process
    total_dataset_size = len(dataset)
    samples_per_process = total_dataset_size // accelerator.num_processes
    remainder = total_dataset_size % accelerator.num_processes
    
    if accelerator.process_index < remainder:
        expected_samples = samples_per_process + 1
    else:
        expected_samples = samples_per_process
        
    logger.info(f"Process {accelerator.process_index}: Expected to process ~{expected_samples} samples from total {total_dataset_size}")
    
    # Handle case where process has no data
    if len(dataloader) == 0:
        logger.warning(f"Process {accelerator.process_index} has no batches to process!")
        # Still need to create empty temp file for consistency
        temp_file.write("")
        temp_file.close()
        
        # Create completion marker
        completion_marker = temp_filename + ".done"
        with open(completion_marker, 'w') as f:
            f.write("0")
        
        # Just return without waiting
        logger.info(f"Process {accelerator.process_index} exiting (no data to process)")
        return
    
    # Process batches
    progress_bar = tqdm(
        total=len(dataloader),
        desc=f"Extracting embeddings (GPU {accelerator.process_index})",
        disable=not accelerator.is_local_main_process
    )
    
    samples_written_local = 0
    
    try:
        total_batches = len(dataloader)
        processed_batches = 0
        last_batch_time = time.time()
        batch_timeout = 120  # 2 minutes per batch max
        
        batch_idx = 0
        dataloader_iter = iter(dataloader)
        
        while batch_idx < total_batches:
            current_time = time.time()
            
            # Check if we're taking too long overall
            if current_time - last_batch_time > batch_timeout:
                logger.error(f"Process {accelerator.process_index}: Timeout after {batch_timeout}s without progress")
                break
            
            try:
                # Get next batch
                batch = next(dataloader_iter)
                
                # Skip empty batches
                if not batch or 'indices' not in batch or len(batch['indices']) == 0:
                    logger.warning(f"Process {accelerator.process_index}: Empty batch at index {batch_idx}")
                    batch_idx += 1
                    continue
                    
                indices = batch['indices']
                texts = batch['texts']
                original_data = batch['original_data']
                has_extract_key = batch['has_extract_key']
                
                # Update timing
                last_batch_time = current_time
                
            except StopIteration:
                logger.info(f"Process {accelerator.process_index}: Reached end of dataloader at batch {batch_idx}")
                break
            except Exception as e:
                logger.error(f"Process {accelerator.process_index}: Error getting batch {batch_idx}: {e}")
                batch_idx += 1
                if batch_idx > total_batches * 0.95:  # Near the end
                    logger.warning(f"Process {accelerator.process_index}: Near end, stopping to avoid further errors")
                    break
                continue
            
            # Log progress and write heartbeat periodically
            if batch_idx % 500 == 0:
                logger.info(f"Process {accelerator.process_index}: Processing batch {batch_idx}/{len(dataloader)} ({samples_written_local} samples written so far)")
                # Write heartbeat file
                heartbeat_file = temp_filename + ".heartbeat"
                with open(heartbeat_file, 'w') as f:
                    f.write(f"{batch_idx},{samples_written_local},{time.time()}")
        
            # Check which samples need embedding computation
            texts_to_embed = []
            indices_to_embed = []
            embedding_map = {}
            batch_results = []
            
            for idx, text, orig_data, has_key in zip(indices, texts, original_data, has_extract_key):
                # Remove the temporary flag from original data before processing
                clean_orig_data = orig_data.copy()
                clean_orig_data.pop('_has_extract_key', None)
                
                # If sample doesn't have extract_key, just add it to results without embedding
                if not has_key:
                    output_sample = clean_orig_data.copy()
                    batch_results.append(output_sample)
                    continue
                
                # For samples with extract_key, check if embedding already exists
                emb_filename = f"emb_{idx:09d}.npy"
                emb_filepath = os.path.join(embedding_dir, emb_filename)
                
                if os.path.exists(emb_filepath):
                    # Load existing embedding instead of recomputing
                    try:
                        existing_emb = np.load(emb_filepath)
                        embedding_map[idx] = existing_emb
                        
                        # Add to results with existing embedding path
                        output_sample = clean_orig_data.copy()
                        output_sample[attribute_name] = emb_filepath
                        batch_results.append(output_sample)
                    except Exception as e:
                        logger.warning(f"Failed to load existing embedding {emb_filepath}: {e}")
                        # Add to list for recomputation
                        texts_to_embed.append(text)
                        indices_to_embed.append(idx)
                else:
                    # Add to list for computation
                    texts_to_embed.append(text)
                    indices_to_embed.append(idx)
        
            # Extract embeddings only for samples that need it
            if texts_to_embed:
                if accelerator.is_local_main_process and len(texts_to_embed) < len(indices):
                    logger.debug(f"Batch {batch_idx}: Computing {len(texts_to_embed)} new embeddings, reusing {len(indices) - len(texts_to_embed)} existing")
                
                try:
                    # Log before extraction
                    logger.debug(f"Process {accelerator.process_index}: Starting embedding extraction for batch {batch_idx} with {len(texts_to_embed)} texts")
                    
                    embeddings = extractor.extract_embeddings(texts_to_embed, instruction=instruction)
                    
                    logger.debug(f"Process {accelerator.process_index}: Completed embedding extraction for batch {batch_idx}")
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"Process {accelerator.process_index}: CUDA OOM at batch {batch_idx}")
                    logger.error(f"Batch size: {len(texts_to_embed)}, Text lengths: {[len(t) for t in texts_to_embed[:5]]}")
                    # Try to clear cache and retry with smaller batch
                    torch.cuda.empty_cache()
                    
                    # Process in smaller sub-batches
                    embeddings = []
                    sub_batch_size = max(1, len(texts_to_embed) // 4)
                    for i in range(0, len(texts_to_embed), sub_batch_size):
                        sub_texts = texts_to_embed[i:i+sub_batch_size]
                        try:
                            sub_embeddings = extractor.extract_embeddings(sub_texts, instruction=instruction)
                            embeddings.extend(sub_embeddings)
                        except Exception as sub_e:
                            logger.error(f"Failed even with smaller batch: {sub_e}")
                            raise
                    embeddings = np.array(embeddings)
                except Exception as e:
                    logger.error(f"Process {accelerator.process_index}: Error extracting embeddings at batch {batch_idx}: {e}")
                    raise
                
                # Save new embeddings and prepare output data
                for idx, emb, text in zip(indices_to_embed, embeddings, texts_to_embed):
                    # Find original data for this index
                    orig_data = next(od for od, i in zip(original_data, indices) if i == idx)
                    
                    # Remove the temporary flag from original data before processing
                    clean_orig_data = orig_data.copy()
                    clean_orig_data.pop('_has_extract_key', None)
                    
                    # Create unique filename based on index
                    emb_filename = f"emb_{idx:09d}.npy"
                    emb_filepath = os.path.join(embedding_dir, emb_filename)
                    
                    # Save embedding
                    np.save(emb_filepath, emb)
                    
                    # Add embedding filepath to original data
                    output_sample = clean_orig_data.copy()
                    output_sample[attribute_name] = emb_filepath
                    batch_results.append(output_sample)
            elif batch_results:
                # All embeddings already exist for this batch
                if accelerator.is_local_main_process:
                    logger.debug(f"Batch {batch_idx}: All {len(indices)} embeddings already exist, reusing")
            
            # Write results directly to temporary file
            for sample in batch_results:
                temp_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                samples_written_local += 1
            
            # Flush periodically
            if (batch_idx + 1) % 500 == 0:
                temp_file.flush()
            
            progress_bar.update(1)
            
            # Clear cache periodically
            if accelerator.is_local_main_process and torch.cuda.is_available():
                if progress_bar.n % 500 == 0:
                    torch.cuda.empty_cache()
            
            # Increment processed batches counter
            processed_batches += 1
            
            # Safety check - if we've processed enough samples, stop
            # This prevents hanging when waiting for batches that may never come
            if samples_written_local >= expected_samples - batch_size:  # Within one batch of expected
                logger.info(f"Process {accelerator.process_index}: Processed {samples_written_local} samples (close to expected {expected_samples}), stopping")
                break
                
            # Additional safety check for near the end
            if batch_idx >= total_batches - 2 and samples_written_local > 0:
                logger.info(f"Process {accelerator.process_index}: At batch {batch_idx}/{total_batches}, processed {samples_written_local} samples, stopping to avoid hang")
                break
            
            # Increment batch index
            batch_idx += 1
    
        progress_bar.close()
        
    except Exception as e:
        logger.error(f"Process {accelerator.process_index} encountered error: {e}")
        progress_bar.close()
        raise
    finally:
        # Close temporary file
        temp_file.close()
        logger.info(f"Process {accelerator.process_index} wrote {samples_written_local} samples to temporary file")
    
    # Wait for all processes to complete writing with timeout
    logger.info(f"Process {accelerator.process_index} finished processing, waiting for others...")
    
    # Write a completion marker file to signal this process is done
    completion_marker = temp_filename + ".done"
    with open(completion_marker, 'w') as f:
        f.write(str(samples_written_local))
    
    # Instead of waiting for all processes, just check completion markers
    if accelerator.is_main_process:
        logger.info("Main process checking for completion markers...")
        
        # Wait for completion markers with timeout
        start_wait = time.time()
        max_wait_time = 300  # 5 minutes max wait
        check_interval = 5  # Check every 5 seconds
        
        while time.time() - start_wait < max_wait_time:
            completion_pattern = f"{Path(output_path).stem}_temp_rank*_*.jsonl.done"
            completion_files = glob.glob(os.path.join(temp_dir, completion_pattern))
            
            logger.info(f"Found {len(completion_files)}/{accelerator.num_processes} completion markers")
            
            if len(completion_files) == accelerator.num_processes:
                logger.info(f"All {accelerator.num_processes} processes have completed")
                break
            
            # Log which processes haven't completed
            completed_ranks = set()
            for cf in completion_files:
                # Extract rank from filename
                rank_match = re.search(r'_rank(\d+)_', cf)
                if rank_match:
                    completed_ranks.add(int(rank_match.group(1)))
            
            missing_ranks = set(range(accelerator.num_processes)) - completed_ranks
            if missing_ranks:
                logger.info(f"Still waiting for processes: {sorted(missing_ranks)}")
                
                # Check heartbeats of missing processes
                stale_processes = []
                for rank in missing_ranks:
                    heartbeat_pattern = f"{Path(output_path).stem}_temp_rank{rank}_*.jsonl.heartbeat"
                    heartbeat_files = glob.glob(os.path.join(temp_dir, heartbeat_pattern))
                    if heartbeat_files:
                        # Read the latest heartbeat
                        with open(heartbeat_files[0], 'r') as f:
                            content = f.read().strip()
                            if content:
                                batch_idx, samples, timestamp = content.split(',')
                                last_update = float(timestamp)
                                if time.time() - last_update > 120:  # 2 minutes without update
                                    stale_processes.append((rank, batch_idx, int(time.time() - last_update)))
                    else:
                        # No heartbeat file means process might have crashed early
                        stale_processes.append((rank, 'no heartbeat', 0))
                
                if stale_processes:
                    logger.warning(f"Potentially stale processes: {stale_processes}")
            
            time.sleep(check_interval)
        
        if time.time() - start_wait >= max_wait_time:
            logger.warning(f"Timeout after {max_wait_time}s. Proceeding with available files...")
            # Don't raise exception, just proceed with what we have
    else:
        # Non-main processes just exit after writing their files
        logger.info(f"Process {accelerator.process_index} completed, exiting...")
        return
    
    # Main process merges all temporary files
    if accelerator.is_main_process:
        logger.info("Merging temporary files into final output...")
        
        # Get all temporary files
        temp_pattern = f"{Path(output_path).stem}_temp_rank*_*.jsonl"
        temp_files = sorted(glob.glob(os.path.join(temp_dir, temp_pattern)))
        logger.info(f"Found {len(temp_files)} temporary files to merge")
        
        # Read existing indices if appending
        existing_indices = set()
        mode = 'w' if overwrite_files or not os.path.exists(output_path) else 'a'
        
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
        
        # Merge all temporary files
        total_written = 0
        all_samples = []
        seen_indices = set()
        duplicates_found = 0
        
        # Read all samples from temporary files
        for temp_file_path in temp_files:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        # Check both existing indices and indices we've seen in this merge
                        if sample['index'] not in existing_indices and sample['index'] not in seen_indices:
                            all_samples.append(sample)
                            seen_indices.add(sample['index'])
                        else:
                            duplicates_found += 1
                    except json.JSONDecodeError:
                        continue
        
        if duplicates_found > 0:
            logger.info(f"Found and skipped {duplicates_found} duplicate entries during merge")
        
        logger.info(f"Collected {len(all_samples)} unique samples to write")
        
        # Sort by index to maintain order
        all_samples.sort(key=lambda x: x['index'])
        
        # Write to final output file
        with open(output_path, mode, encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total_written += 1
        
        logger.info(f"Successfully merged and wrote {total_written} new samples to {output_path}")
        
        # Clean up temporary files and completion markers
        for temp_file_path in temp_files:
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
                # Also remove completion marker and heartbeat
                completion_marker = temp_file_path + ".done"
                if os.path.exists(completion_marker):
                    os.remove(completion_marker)
                heartbeat_file = temp_file_path + ".heartbeat"
                if os.path.exists(heartbeat_file):
                    os.remove(heartbeat_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
        
        logger.info(f"Embeddings saved to: {embedding_dir}")
        logger.info(f"Output JSONL saved to: {output_path}")
    
    # Final logging (no synchronization to avoid hanging)
    logger.info(f"Process {accelerator.process_index} completed all tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen embeddings from a dataset using multi-GPU acceleration"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model ID for the embedding model"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the input JSONL file"
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
        default=32,
        help="Batch size per GPU for processing (default: 32)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--extract_key",
        type=str,
        default="transcription",
        help="The JSON key to extract text from for embedding (default: 'transcription')"
    )
    
    parser.add_argument(
        "--instruction",
        type=str,
        default="Represent this speech transcript for semantic similarity search",
        help="Optional instruction for the embedding task (default: 'Represent this speech transcript for semantic similarity search')"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Log configuration (only from main process, handled by accelerate)
    logger.info("Configuration:")
    logger.info(f"  Model ID: {args.model_id}")
    logger.info(f"  Dataset path: {args.dataset_path}")
    logger.info(f"  Embedding directory: {args.embedding_dir}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Batch size per GPU: {args.batch_size}")
    logger.info(f"  Number of workers: {args.num_workers}")
    logger.info(f"  Force reprocess: {args.overwrite_files}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Extract key: {args.extract_key}")
    if args.instruction:
        logger.info(f"  Instruction: {args.instruction}")
    
    if torch.cuda.is_available():
        logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("  Running on CPU")
    
    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        embedding_dir=args.embedding_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        extract_key=args.extract_key,
        overwrite_files=args.overwrite_files,
        mixed_precision=args.mixed_precision,
        instruction=args.instruction,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
