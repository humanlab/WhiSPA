#!/usr/bin/env python3
"""
Embedding Extractor for WhiSPA and Qwen Models
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import List, Union, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from data.utils import last_token_pool, mean_token_pool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract embeddings from various models (WhiSPA, Qwen, etc.).
    
    Supports:
    - WhiSPA audio embeddings (local checkpoint or HuggingFace)
    - Qwen text embeddings  
    - Generic HuggingFace text models
    - Single or multi-GPU extraction via Accelerate
    - Automatic caching of embeddings
    
    Returns None if extraction fails (no embeddings produced).
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = 'cuda',
        batch_size: int = 32,
        num_workers: int = 32,
        cache_dir: str = '/tmp/WhiSPA/eval/encode',
        use_cache: bool = True,
        dtype: torch.dtype = torch.float32,
        accelerator: Optional[Accelerator] = None
    ):
        """
        Initialize embedding extractor.
        
        Args:
            model_id: Model identifier (HuggingFace ID or local path)
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for processing
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use cached embeddings
            dtype: Model dtype
            num_workers: Number of workers for parallel processing
            accelerator: Optional existing Accelerator instance to reuse
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.dtype = dtype
        self.num_workers = num_workers
        self.requested_device = device  # Store user's device preference
        
        # Ensure cache directory exists
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use provided accelerator or create new one if needed
        if accelerator is not None:
            # Use the provided accelerator
            self.accelerator = accelerator
            self.device = self.accelerator.device
            
            # Log GPU usage (only from main process)
            if self.accelerator.is_local_main_process:
                if self.accelerator.num_processes > 1:
                    logger.info(f"Using provided Accelerator with {self.accelerator.num_processes} GPUs")
                else:
                    logger.info(f"Using provided Accelerator on single GPU: {self.device}")
        elif device == 'cuda' and torch.cuda.is_available():
            # Create new accelerator for single GPU
            # Note: This path is only used when running without accelerate launch
            mixed_precision = 'no'
            if dtype == torch.bfloat16:
                mixed_precision = 'bf16'
            elif dtype == torch.float16:
                mixed_precision = 'fp16'
                
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            self.device = self.accelerator.device
            logger.info(f"Created new Accelerator on device: {self.device}")
        else:
            # Use CPU or single device without accelerator
            self.accelerator = None
            self.device = 'cpu' if device == 'cpu' else (device if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device} (no accelerator)")
        
        # Check if this is a WhiSPA model
        self.is_whispa = self._check_if_whispa()
        
        # Load model
        self._load_model()
    
    def _check_if_whispa(self):
        """Check if the model is a WhiSPA model."""
        # Check if it's a local WhiSPA checkpoint
        if os.path.exists(self.model_id) and os.path.isdir(self.model_id):
            config_path = os.path.join(self.model_id, 'config.json')
            if os.path.exists(config_path):
                return True
        
        # Check model ID for WhiSPA indicators
        model_lower = self.model_id.lower()
        return 'whispa' in model_lower or 'voxtral' in model_lower or 'jarhatz' in model_lower
        
    def _load_model(self):
        """Load the appropriate model based on model_id."""
        logger.info(f"Loading model: {self.model_id}")
        
        # Check if it's a local WhiSPA checkpoint
        if os.path.exists(self.model_id) and os.path.isdir(self.model_id):
            self._load_whispa_checkpoint()
        elif 'Jarhatz' in self.model_id:
            self._load_whispa_huggingface()
        elif 'qwen' in self.model_id.lower():
            self._load_qwen()
        else:
            # Try loading as a generic HuggingFace model
            self._load_generic_huggingface()
            
    def _load_whispa_checkpoint(self):
        """Load WhiSPA from local checkpoint using from_pretrained_local."""
        if self.accelerator and self.accelerator.is_local_main_process:
            logger.info(f"Loading WhiSPA from local checkpoint on process {self.accelerator.process_index}")
        elif not self.accelerator:
            logger.info("Loading WhiSPA from local checkpoint")
        
        # Load model and config using the built-in method
        # This will load config.json and model.safetensors from the directory
        logger.info(f"Loading WhiSPA model from: {self.model_id}")
        self.model = WhiSPAModel.from_pretrained_local(self.model_id)
        self.config = self.model.config
        
        # Update config for inference mode and device settings
        self.config.stage = 'inference'
        self.config.device = self.device
        self.config.dtype = self.dtype
        
        # Update model's config with our settings
        self.model.config = self.config
        
        # CRITICAL: Move model components to the correct device and dtype
        # The model is loaded on CPU by default, we need to move it
        self.model.spectral_encoder = self.model.spectral_encoder.to(dtype=self.dtype, device=self.device)
        self.model.multi_modal_projector = self.model.multi_modal_projector.to(dtype=self.dtype, device=self.device)
        self.model.language_model = self.model.language_model.to(dtype=self.dtype, device=self.device)
        if hasattr(self.model, 'voxtral'):
            self.model.voxtral = self.model.voxtral.to(dtype=self.dtype, device=self.device)
        if hasattr(self.model, 'activation'):
            self.model.activation = self.model.activation.to(self.device)
        
        self.model.eval()
        
        logger.info(f"WhiSPA model moved to device: {self.device}, dtype: {self.dtype}")
        
        if self.accelerator and self.accelerator.is_local_main_process:
            logger.info(f"WhiSPA model ready on process {self.accelerator.process_index}")
        elif not self.accelerator:
            logger.info(f"WhiSPA model ready")
        
        self.model_type = 'whispa'
        
    def _load_whispa_huggingface(self):
        """Load WhiSPA from HuggingFace."""
        logger.info("Loading WhiSPA from HuggingFace")
        self.model = WhiSPAModel.from_pretrained(self.model_id)
        
        # Update model config with device settings
        self.model.config.device = self.device
        self.model.config.dtype = self.dtype
        
        self.model.eval()
        
        # For WhiSPA models, do NOT use accelerator.prepare() as it breaks with sharding
        # The model handles device placement internally
        logger.info(f"WhiSPA model loaded on device: {self.device}")
        
        self.config = self.model.config
        self.model_type = 'whispa'
        
    def _load_qwen(self):
        """Load Qwen embedding model."""
        logger.info("Loading Qwen embedding model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side='left'
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # For non-WhiSPA models, use accelerator.prepare() if available
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            logger.info(f"Qwen model prepared with accelerator on device: {self.device}")
        else:
            self.model = self.model.to(self.device)
            logger.info(f"Qwen model loaded on device: {self.device}")
        
        self.model_type = 'qwen'
        
    def _load_generic_huggingface(self):
        """Load a generic HuggingFace model."""
        logger.info("Loading generic HuggingFace model")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )
            self.model.eval()
            
            # For non-WhiSPA models, use accelerator.prepare() if available
            if self.accelerator is not None:
                self.model = self.accelerator.prepare(self.model)
                logger.info(f"Model prepared with accelerator on device: {self.device}")
            else:
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded on device: {self.device}")
            
            self.model_type = 'generic'
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_id}: {e}")
    
    def _get_cache_path(self, data_hash: str) -> str:
        """Get cache file path for embeddings."""
        model_hash = hashlib.md5(self.model_id.encode()).hexdigest()[:8]
        cache_filename = f"embeddings_{model_hash}_{data_hash}.npz"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _compute_data_hash(self, data: List[Union[str, np.ndarray]]) -> str:
        """Compute hash of input data for caching."""
        # Include dataset length to avoid stale cache collisions
        prefix = f"len={len(data)}|"
        if isinstance(data[0], str):
            # Text data / file paths
            sample_slice = data[:min(200, len(data))]
            data_str = prefix + '\n'.join(sample_slice)
        else:
            # Numpy arrays or other types cast to str
            sample_slice = data[:min(200, len(data))]
            data_str = prefix + '\n'.join(str(d) for d in sample_slice)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    @torch.no_grad()
    def extract_audio_embeddings(self, audio_paths: List[str]) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Extract embeddings from audio files with multi-GPU support.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Tuple of (embeddings array, valid_indices) where valid_indices are indices
            in the input list that were successfully embedded. Returns None if extraction fails.
        """
        if self.model_type not in ['whispa']:
            raise ValueError(f"Model {self.model_id} does not support audio embeddings")
        
        # Check cache
        if self.use_cache:
            data_hash = self._compute_data_hash(audio_paths)
            cache_path = self._get_cache_path(data_hash)
            
            if os.path.exists(cache_path):
                logger.info(f"Loading cached embeddings from {cache_path}")
                cached = np.load(cache_path)
                embeddings = cached['embeddings']
                # Use indices from cache when available; otherwise, fall back safely
                if 'indices' in cached and 'n_inputs' in cached:
                    n_inputs = int(cached['n_inputs'])
                    if n_inputs == len(audio_paths):
                        indices = cached['indices'].astype(np.int64).tolist()
                        return embeddings, indices
                    else:
                        logger.warning(
                            f"Cached n_inputs ({n_inputs}) != current inputs ({len(audio_paths)}); ignoring cache"
                        )
                elif embeddings.shape[0] == len(audio_paths):
                    # Legacy cache without indices but full coverage
                    indices = list(range(len(audio_paths)))
                    return embeddings, indices
                else:
                    logger.warning(
                        "Cached embeddings count does not match current inputs and lacks indices; recomputing"
                    )
        else:
            cache_path = None
        
        # Multi-GPU extraction
        if self.accelerator and self.accelerator.num_processes > 1:
            logger.info(f"Using distributed extraction with {self.accelerator.num_processes} processes")
            return self._extract_audio_embeddings_distributed(audio_paths)
        
        # Single device extraction
        logger.info(f"Extracting audio embeddings for {len(audio_paths)} files")
        all_embeddings = []
        valid_indices: List[int] = []
        
        for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="Extracting audio embeddings"):
            batch_paths = audio_paths[i:i + self.batch_size]
            batch_indices = list(range(i, min(i + self.batch_size, len(audio_paths))))
            
            try:
                batch_embeddings = self.model.encode(batch_paths, language="en")
                if batch_embeddings is not None:
                    # Keep as tensor until final conversion
                    all_embeddings.append(batch_embeddings.cpu())
                    valid_indices.extend(batch_indices)
            except Exception as e:
                logger.error(f"Error in batch {i//self.batch_size}: {e}. Falling back to per-file processing.")
                # Fallback: try each file individually
                for offset, path in enumerate(batch_paths):
                    try:
                        single = self.model.encode([path], language="en")
                        if single is not None:
                            all_embeddings.append(single.cpu())
                            valid_indices.append(i + offset)
                    except Exception as e_single:
                        logger.warning(f"Failed to encode file at index {i + offset}: {e_single}")
        
        if not all_embeddings:
            logger.error("No embeddings extracted!")
            return None
        
        # Convert to numpy only at the end
        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}; valid samples: {len(valid_indices)} / {len(audio_paths)}")
        
        # Cache embeddings
        if self.use_cache and cache_path:
            logger.info(f"Caching embeddings to {cache_path}")
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                indices=np.array(valid_indices, dtype=np.int64),
                n_inputs=np.array(len(audio_paths), dtype=np.int64)
            )
        
        return embeddings, valid_indices
    
    def _extract_audio_embeddings_distributed(self, audio_paths: List[str]) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Extract audio embeddings using multiple GPUs via Accelerator.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Tuple of (embeddings array, valid_indices)
        """
        # Split data across processes
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        # Calculate indices for this process
        samples_per_process = len(audio_paths) // num_processes
        remainder = len(audio_paths) % num_processes
        
        # Distribute remainder samples to first processes
        if process_index < remainder:
            start_idx = process_index * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = process_index * samples_per_process + remainder
            end_idx = start_idx + samples_per_process
        
        # Get this process's subset of data
        process_audio_paths = audio_paths[start_idx:end_idx]
        
        logger.info(f"Process {process_index}: Extracting embeddings for {len(process_audio_paths)} files (indices {start_idx}-{end_idx})")
        
        # Extract embeddings for this process's data
        local_embeddings = []
        local_indices: List[int] = []
        
        # Use tqdm only on main process
        iterator = range(0, len(process_audio_paths), self.batch_size)
        if self.accelerator.is_local_main_process:
            iterator = tqdm(iterator, desc=f"Process {process_index} extracting")
        
        for i in iterator:
            batch_paths = process_audio_paths[i:i + self.batch_size]
            # Compute global indices for this batch
            batch_start_global = start_idx + i
            batch_indices_global = list(range(batch_start_global, batch_start_global + len(batch_paths)))
            
            try:
                batch_embeddings = self.model.encode(batch_paths, language="en")
                if batch_embeddings is not None:
                    # Keep as tensor
                    local_embeddings.append(batch_embeddings.cpu())
                    local_indices.extend(batch_indices_global)
                    
                    if i == 0:  # Log first batch shape
                        logger.info(f"Process {process_index}: First batch shape: {batch_embeddings.shape}")
            except Exception as e:
                logger.error(f"Process {process_index}: Error in batch {i//self.batch_size}: {e}. Falling back per-file.")
                # Fallback per-file
                for offset, path in enumerate(batch_paths):
                    try:
                        single = self.model.encode([path], language="en")
                        if single is not None:
                            local_embeddings.append(single.cpu())
                            local_indices.append(batch_start_global + offset)
                    except Exception as e_single:
                        logger.warning(f"Process {process_index}: Failed to encode file at global index {batch_start_global + offset}: {e_single}")
        
        if not local_embeddings:
            logger.warning(f"Process {process_index}: No embeddings extracted")
            local_tensor = None
        else:
            # Concatenate all tensors
            local_tensor = torch.cat(local_embeddings, dim=0)
            logger.info(f"Process {process_index}: Extracted {local_tensor.shape[0]} embeddings of dim {local_tensor.shape[1]}")
        
        # Gathering logic
        logger.info(f"Process {process_index}: Gathering results...")
        
        if local_tensor is not None:
            # Move to device for gathering
            local_tensor = local_tensor.to(self.accelerator.device)
            local_size = torch.tensor(local_tensor.shape[0], device=self.accelerator.device)
            indices_tensor = torch.tensor(local_indices, dtype=torch.long, device=self.accelerator.device)
        else:
            local_size = torch.tensor(0, device=self.accelerator.device)
            indices_tensor = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)
        
        # Gather sizes from all processes
        all_sizes = self.accelerator.gather(local_size)
        
        # Check if any process has data
        total_size = all_sizes.sum().item()
        if total_size == 0:
            if self.accelerator.is_main_process:
                logger.error("No embeddings extracted from any process!")
                return None
            return None
        
        # Get max size and embedding dimension
        max_size = all_sizes.max().item()
        
        # Determine embedding dimension from any process that has data
        if local_tensor is not None:
            embedding_dim = local_tensor.shape[1]
        else:
            # We need to get the dimension from another process
            # This is a bit tricky, but we can gather a dummy tensor
            embedding_dim = None
        
        # Gather embedding dimensions
        local_dim = torch.tensor(embedding_dim if embedding_dim else 0, device=self.accelerator.device)
        all_dims = self.accelerator.gather(local_dim)
        embedding_dim = all_dims[all_dims > 0][0].item() if (all_dims > 0).any() else None
        
        if embedding_dim is None:
            if self.accelerator.is_main_process:
                logger.error("Could not determine embedding dimension!")
                return None
            return None
        
        # Pad tensors for gathering
        if local_tensor is not None:
            padded = torch.zeros(max_size, embedding_dim, dtype=torch.float32, device=self.accelerator.device)
            padded[:local_tensor.shape[0]] = local_tensor
            padded_idx = torch.full((max_size,), fill_value=-1, dtype=torch.long, device=self.accelerator.device)
            padded_idx[:indices_tensor.shape[0]] = indices_tensor
        else:
            padded = torch.zeros(max_size, embedding_dim, dtype=torch.float32, device=self.accelerator.device)
            padded_idx = torch.full((max_size,), fill_value=-1, dtype=torch.long, device=self.accelerator.device)
        
        # Gather all padded embeddings
        all_padded = self.accelerator.gather(padded)
        all_padded_idx = self.accelerator.gather(padded_idx)
        
        # Only main process combines results
        if self.accelerator.is_main_process:
            embeddings_list = []
            indices_list = []
            
            for i in range(self.accelerator.num_processes):
                size = all_sizes[i].item()
                if size > 0:
                    # Extract this process's embeddings
                    proc_embeddings = all_padded[i * max_size:(i + 1) * max_size][:size]
                    embeddings_list.append(proc_embeddings)
                    proc_indices = all_padded_idx[i * max_size:(i + 1) * max_size][:size]
                    indices_list.append(proc_indices)
            
            # Concatenate and convert to numpy
            all_embeddings = torch.cat(embeddings_list, dim=0)
            embeddings = all_embeddings.cpu().numpy().astype(np.float32)
            valid_indices = torch.cat(indices_list, dim=0).cpu().numpy().astype(np.int64).tolist()
            
            logger.info(f"Main process: Combined embeddings shape: {embeddings.shape}; valid samples: {len(valid_indices)} / {len(audio_paths)}")
            
            # Cache embeddings
            if self.use_cache:
                data_hash = self._compute_data_hash(audio_paths)
                cache_path = self._get_cache_path(data_hash)
                logger.info(f"Caching embeddings to {cache_path}")
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    indices=np.array(valid_indices, dtype=np.int64),
                    n_inputs=np.array(len(audio_paths), dtype=np.int64)
                )
            
            return embeddings, valid_indices
        
        return None
    
    @torch.no_grad()
    def extract_text_embeddings(self, texts: List[str]) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Extract embeddings from text.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (embeddings array, valid_indices) where valid_indices are indices
            in the input list that were successfully embedded. Returns None if extraction fails.
        """
        # Check cache
        if self.use_cache:
            data_hash = self._compute_data_hash(texts)
            cache_path = self._get_cache_path(data_hash)
            
            if os.path.exists(cache_path):
                logger.info(f"Loading cached embeddings from {cache_path}")
                cached = np.load(cache_path)
                embeddings = cached['embeddings']
                if 'indices' in cached and 'n_inputs' in cached:
                    n_inputs = int(cached['n_inputs'])
                    if n_inputs == len(texts):
                        indices = cached['indices'].astype(np.int64).tolist()
                        return embeddings, indices
                    else:
                        logger.warning(
                            f"Cached n_inputs ({n_inputs}) != current inputs ({len(texts)}); ignoring cache"
                        )
                elif embeddings.shape[0] == len(texts):
                    indices = list(range(len(texts)))
                    return embeddings, indices
                else:
                    logger.warning("Cached embeddings count mismatch and no indices present; recomputing")
        else:
            cache_path = None
        
        # Multi-GPU extraction for non-WhiSPA models
        if self.accelerator and self.accelerator.num_processes > 1 and self.model_type != 'whispa':
            return self._extract_text_embeddings_distributed(texts)
        
        # Single device extraction
        logger.info(f"Extracting text embeddings for {len(texts)} samples")
        all_embeddings = []
        valid_indices: List[int] = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting text embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            if self.model_type == 'qwen':
                try:
                    batch_embeddings = self._extract_qwen_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"Text batch {i//self.batch_size} failed: {e}. Falling back per-sample.")
                    batch_embeddings = None
            else:
                try:
                    batch_embeddings = self._extract_generic_text_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"Text batch {i//self.batch_size} failed: {e}. Falling back per-sample.")
                    batch_embeddings = None
            
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)
                valid_indices.extend(list(range(i, min(i + self.batch_size, len(texts)))))
            else:
                # Per-sample fallback
                for offset, text in enumerate(batch_texts):
                    try:
                        if self.model_type == 'qwen':
                            single = self._extract_qwen_embeddings([text])
                        else:
                            single = self._extract_generic_text_embeddings([text])
                        if single is not None:
                            all_embeddings.append(single)
                            valid_indices.append(i + offset)
                    except Exception as e_single:
                        logger.warning(f"Failed to embed text at index {i + offset}: {e_single}")
        
        if not all_embeddings:
            logger.error("No text embeddings extracted!")
            return None
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Extracted text embeddings shape: {embeddings.shape}; valid samples: {len(valid_indices)} / {len(texts)}")
        
        # Cache embeddings
        if self.use_cache and cache_path:
            logger.info(f"Caching embeddings to {cache_path}")
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                indices=np.array(valid_indices, dtype=np.int64),
                n_inputs=np.array(len(texts), dtype=np.int64)
            )
        
        return embeddings, valid_indices
    
    def _extract_text_embeddings_distributed(self, texts: List[str]) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Extract text embeddings using multiple GPUs via Accelerator.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (embeddings array, valid_indices)
        """
        # Split data across processes
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        # Calculate indices for this process
        samples_per_process = len(texts) // num_processes
        remainder = len(texts) % num_processes
        
        # Distribute remainder samples to first processes
        if process_index < remainder:
            start_idx = process_index * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = process_index * samples_per_process + remainder
            end_idx = start_idx + samples_per_process
        
        # Get this process's subset of data
        process_texts = texts[start_idx:end_idx]
        
        if self.accelerator.is_local_main_process:
            logger.info(f"Process {process_index}: Extracting embeddings for {len(process_texts)} texts")
        
        # Extract embeddings for this process's data
        local_embeddings = []
        local_indices: List[int] = []
        
        for i in range(0, len(process_texts), self.batch_size):
            batch_texts = process_texts[i:i + self.batch_size]
            
            if self.model_type == 'qwen':
                try:
                    batch_embeddings = self._extract_qwen_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"Process {process_index}: Text batch {i//self.batch_size} failed: {e}. Falling back per-sample.")
                    batch_embeddings = None
            else:
                try:
                    batch_embeddings = self._extract_generic_text_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"Process {process_index}: Text batch {i//self.batch_size} failed: {e}. Falling back per-sample.")
                    batch_embeddings = None
            
            if batch_embeddings is not None:
                # Keep as numpy but don't repeatedly stack
                local_embeddings.append(batch_embeddings)
                # Compute global indices for this batch
                batch_start_global = start_idx + i
                local_indices.extend(list(range(batch_start_global, batch_start_global + len(batch_texts))))
            else:
                # Per-sample fallback
                for offset, text in enumerate(batch_texts):
                    try:
                        if self.model_type == 'qwen':
                            single = self._extract_qwen_embeddings([text])
                        else:
                            single = self._extract_generic_text_embeddings([text])
                        if single is not None:
                            local_embeddings.append(single)
                            local_indices.append(start_idx + i + offset)
                    except Exception as e_single:
                        logger.warning(f"Process {process_index}: Failed to embed text at global index {start_idx + i + offset}: {e_single}")
        
        if not local_embeddings:
            logger.warning(f"Process {process_index}: No text embeddings extracted")
            local_array = None
        else:
            # Stack all numpy arrays
            local_array = np.vstack(local_embeddings).astype(np.float32)
            logger.info(f"Process {process_index}: Extracted {local_array.shape}")
        
        # Convert to tensor for gathering
        if local_array is not None:
            local_tensor = torch.from_numpy(local_array).to(self.accelerator.device)
            local_size = torch.tensor(local_array.shape[0], device=self.accelerator.device)
            indices_tensor = torch.tensor(local_indices, dtype=torch.long, device=self.accelerator.device)
        else:
            local_tensor = None
            local_size = torch.tensor(0, device=self.accelerator.device)
            indices_tensor = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)
        
        # Gather sizes
        all_sizes = self.accelerator.gather(local_size)
        
        # Check if any process has data
        if all_sizes.sum().item() == 0:
            if self.accelerator.is_main_process:
                logger.error("No text embeddings from any process!")
                return None
            return None
        
        # Get dimensions
        max_size = all_sizes.max().item()
        
        if local_tensor is not None:
            embedding_dim = local_tensor.shape[1]
        else:
            embedding_dim = None
        
        # Gather dimensions
        local_dim = torch.tensor(embedding_dim if embedding_dim else 0, device=self.accelerator.device)
        all_dims = self.accelerator.gather(local_dim)
        embedding_dim = all_dims[all_dims > 0][0].item() if (all_dims > 0).any() else None
        
        if embedding_dim is None:
            if self.accelerator.is_main_process:
                logger.error("Could not determine embedding dimension!")
                return None
            return None
        
        # Pad for gathering
        if local_tensor is not None:
            padded = torch.zeros(max_size, embedding_dim, dtype=torch.float32, device=self.accelerator.device)
            padded[:local_tensor.shape[0]] = local_tensor
            padded_idx = torch.full((max_size,), fill_value=-1, dtype=torch.long, device=self.accelerator.device)
            padded_idx[:indices_tensor.shape[0]] = indices_tensor
        else:
            padded = torch.zeros(max_size, embedding_dim, dtype=torch.float32, device=self.accelerator.device)
            padded_idx = torch.full((max_size,), fill_value=-1, dtype=torch.long, device=self.accelerator.device)
        
        # Gather all padded embeddings
        all_padded = self.accelerator.gather(padded)
        all_padded_idx = self.accelerator.gather(padded_idx)
        
        # Only main process combines results
        if self.accelerator.is_main_process:
            embeddings_list = []
            indices_list = []
            
            for i in range(self.accelerator.num_processes):
                size = all_sizes[i].item()
                if size > 0:
                    proc_embeddings = all_padded[i * max_size:(i + 1) * max_size][:size]
                    embeddings_list.append(proc_embeddings)
                    proc_indices = all_padded_idx[i * max_size:(i + 1) * max_size][:size]
                    indices_list.append(proc_indices)
            
            # Concatenate and convert to numpy
            all_embeddings = torch.cat(embeddings_list, dim=0)
            embeddings = all_embeddings.cpu().numpy().astype(np.float32)
            valid_indices = torch.cat(indices_list, dim=0).cpu().numpy().astype(np.int64).tolist()
            
            logger.info(f"Main process: Combined text embeddings shape: {embeddings.shape}; valid samples: {len(valid_indices)} / {len(texts)}")
            
            # Cache embeddings
            if self.use_cache:
                data_hash = self._compute_data_hash(texts)
                cache_path = self._get_cache_path(data_hash)
                logger.info(f"Caching embeddings to {cache_path}")
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    indices=np.array(valid_indices, dtype=np.int64),
                    n_inputs=np.array(len(texts), dtype=np.int64)
                )
            
            return embeddings, valid_indices
        
        return None
    
    def _extract_qwen_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings using Qwen model."""
        # Use instruction for speech transcript embedding
        instruction = "Represent this speech transcript for semantic similarity search"
        formatted_texts = [f'Instruct: {instruction}\nQuery: {text}' for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Use last token pooling (recommended for Qwen3-Embedding)
        embeddings = last_token_pool(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        
        # Convert to float32 before numpy (handles bfloat16)
        return embeddings.cpu().float().numpy()
    
    def _extract_generic_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings using generic HuggingFace model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Use mean pooling by default
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = mean_token_pool(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
        
        return embeddings.cpu().float().numpy()
    
    def clear_cache(self):
        """Clear cached embeddings for this model."""
        model_hash = hashlib.md5(self.model_id.encode()).hexdigest()[:8]
        pattern = f"embeddings_{model_hash}_*.npz"
        
        for cache_file in Path(self.cache_dir).glob(pattern):
            logger.info(f"Removing cache file: {cache_file}")
            cache_file.unlink()
