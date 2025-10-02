#!/usr/bin/env python3
"""
Main Evaluation Runner for Embedding Models on IEMOCAP and MELD Datasets
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.iemocap.dataset import IEMOCAPDataset
from data.meld.dataset import MELDDataset
from eval.encode.embeddings import EmbeddingExtractor
from eval.encode.evaluator import RidgeEvaluator
from eval.encode.metrics import MetricsReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


"""
Multi-GPU WhiSPA:
  accelerate launch --num_processes 8 \
    eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type audio \
    --dataset_list iemocap meld \
    --batch_size 128 \
    --num_workers 64

Multi-GPU Qwen:
  accelerate launch --num_processes 8 \
    eval/encode/run.py \
    --model_id Qwen/Qwen3-Embedding-4B \
    --model_type text \
    --dataset_list iemocap meld \
    --batch_size 128 \
    --num_workers 64
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on IEMOCAP and MELD datasets"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model identifier (HuggingFace ID or local checkpoint path). Required unless --embedding_path is provided"
    )

    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Path to pre-computed embeddings (.npz file or directory containing dataset.npz files)"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "audio", "text"],
        default="auto",
        help="Type of embeddings to extract (auto-detect by default)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_list",
        type=str,
        nargs="*",
        choices=["iemocap", "meld"],
        default=None,
        help="List of datasets to evaluate on (default: all available datasets)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size for train/test split"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization parameter"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for data loading and parallel processing"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for computation"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype"
    )
    
    # Cache arguments
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/WhiSPA/eval/encode",
        help="Directory for caching embeddings"
    )
    
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable embedding caching"
    )
    
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear existing cache before running"
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def determine_model_type(model_id: str) -> str:
    """Determine if model is for audio or text embeddings."""
    audio_keywords = {'whispa', 'whisper', 'wav2vec', 'hubert', 'voxtral'}
    text_keywords = {'qwen', 'bert', 'roberta', 'sentence', 'jina', 'text'}
    
    model_id_lower = model_id.lower()
    
    # Check if it's a local WhiSPA checkpoint
    if os.path.exists(model_id) and os.path.isdir(model_id):
        # Check for config to determine type
        config_files = ['config.json', 'config.pth']
        for config_file in config_files:
            if os.path.exists(os.path.join(model_id, config_file)):
                # Assume it's WhiSPA (audio)
                return 'audio'
    
    # Check keywords
    if any(keyword in model_id_lower for keyword in audio_keywords):
        return 'audio'
    elif any(keyword in model_id_lower for keyword in text_keywords):
        return 'text'
    else:
        # Default to text for unknown models
        logger.warning(f"Could not determine model type for {model_id}, defaulting to text")
        return 'text'


def extract_embeddings(
    dataset: Union[IEMOCAPDataset, MELDDataset],
    extractor: EmbeddingExtractor,
    model_type: str
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    Extract embeddings for the dataset.
    
    Args:
        dataset: Dataset (IEMOCAP or MELD)
        extractor: Embedding extractor
        model_type: 'audio' or 'text'
        
    Returns:
        Tuple of (embeddings, valid_indices) or None if extraction fails
    """
    if model_type == 'audio':
        # Extract audio embeddings
        audio_paths = dataset.get_audio_paths()
        logger.info(f"Extracting audio embeddings for {len(audio_paths)} samples")
        embeddings = extractor.extract_audio_embeddings(audio_paths)
    else:
        # Extract text embeddings from transcriptions
        transcriptions = dataset.get_transcriptions()
        logger.info(f"Extracting text embeddings for {len(transcriptions)} samples")
        embeddings = extractor.extract_text_embeddings(transcriptions)
    
    return embeddings


def load_offline_embeddings(
    embedding_path: str,
    dataset_name: str,
    expected_n_samples: int
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    Load pre-computed embeddings from file.
    
    Args:
        embedding_path: Path to .npz file or directory containing dataset embeddings
        dataset_name: Name of the dataset ('iemocap' or 'meld')
        expected_n_samples: Expected number of samples in the dataset
        
    Returns:
        Tuple of (embeddings, valid_indices) or None if loading fails
    """
    # Determine the file path
    if os.path.isfile(embedding_path):
        # Single file provided
        file_path = embedding_path
    elif os.path.isdir(embedding_path):
        # Directory provided, look for dataset-specific file
        possible_names = [
            f"{dataset_name}.npz",
            f"{dataset_name}_embeddings.npz",
            f"{dataset_name.upper()}.npz",
            f"{dataset_name.upper()}_embeddings.npz"
        ]
        
        file_path = None
        for name in possible_names:
            candidate = os.path.join(embedding_path, name)
            if os.path.exists(candidate):
                file_path = candidate
                break
        
        if file_path is None:
            logger.error(f"No embedding file found for {dataset_name} in {embedding_path}")
            logger.error(f"Looked for: {', '.join(possible_names)}")
            return None
    else:
        logger.error(f"Embedding path does not exist: {embedding_path}")
        return None
    
    # Load the embeddings
    logger.info(f"Loading embeddings from: {file_path}")
    try:
        data = np.load(file_path)
        
        # Extract embeddings
        if 'embeddings' not in data:
            logger.error(f"No 'embeddings' array found in {file_path}")
            return None
        
        embeddings = data['embeddings']
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        
        # Extract or generate valid indices
        if 'indices' in data:
            valid_indices = data['indices'].astype(np.int64).tolist()
            logger.info(f"Loaded {len(valid_indices)} valid indices")
        else:
            # Assume all samples are valid
            valid_indices = list(range(embeddings.shape[0]))
            logger.info(f"No indices provided, assuming all {len(valid_indices)} samples are valid")
        
        # Validation
        if 'n_inputs' in data:
            n_inputs = int(data['n_inputs'])
            if n_inputs != expected_n_samples:
                logger.warning(
                    f"Embedding file has n_inputs={n_inputs} but dataset has {expected_n_samples} samples. "
                    "This may indicate a dataset version mismatch."
                )
        
        # Validate indices
        if max(valid_indices) >= expected_n_samples:
            logger.error(
                f"Invalid indices: max index {max(valid_indices)} >= dataset size {expected_n_samples}"
            )
            return None
        
        return embeddings, valid_indices
        
    except Exception as e:
        logger.error(f"Failed to load embeddings from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_outcome(
    outcome: str,
    embeddings: np.ndarray,
    dataset: Union[IEMOCAPDataset, MELDDataset],
    evaluator: RidgeEvaluator,
    valid_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluate embeddings on a specific outcome.
    
    Args:
        outcome: Outcome to evaluate
        embeddings: Embeddings array
        dataset: Dataset (IEMOCAP or MELD)
        evaluator: Ridge evaluator
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating outcome: {outcome}")
    
    # Get outcome info
    outcome_info = dataset.get_outcome_info(outcome)
    task_type = outcome_info['type']
    
    # Get targets
    y = dataset.get_targets(outcome, encoded=True)
    
    # Get stratification labels for CV
    stratify_labels = dataset.get_stratification_labels()
    
    # Subset labels to valid indices if provided
    if valid_indices is not None:
        y = y[valid_indices]
        if stratify_labels is not None:
            stratify_labels = stratify_labels[valid_indices]
    
    # Validate lengths
    if embeddings.shape[0] != len(y):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"X={embeddings.shape[0]}, y={len(y)}"
        )
    
    # Evaluate
    results = evaluator.evaluate(
        X=embeddings,
        y=y,
        task_type=task_type,
        stratify_labels=stratify_labels
    )
    
    # Add outcome info to results
    results['outcome_info'] = outcome_info
    
    return results


def evaluate_single_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    accelerator: Accelerator,
    extractor: Optional[EmbeddingExtractor],
    model_type: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate embeddings on a single dataset.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        args: Command line arguments
        accelerator: Accelerator instance
        extractor: Embedding extractor instance
        model_type: Type of model (audio/text)
        
    Returns:
        Tuple of (results, dataset_info)
    """
    # Load dataset
    if accelerator.is_main_process:
        logger.info(f"\nLoading {dataset_name.upper()} dataset...")
    
    if dataset_name == "iemocap":
        dataset = IEMOCAPDataset()
    elif dataset_name == "meld":
        dataset = MELDDataset()
    
    if accelerator.is_main_process:
        logger.info(f"Dataset loaded: {dataset}")
    
    # Extract or load embeddings
    if args.embedding_path:
        # Load offline embeddings
        if accelerator.is_main_process:
            logger.info(f"Loading offline embeddings for {dataset_name.upper()}...")
            result = load_offline_embeddings(
                args.embedding_path, 
                dataset_name, 
                len(dataset)
            )
            
            # Check if embeddings were successfully loaded
            if result is None:
                logger.error(f"Failed to load embeddings for {dataset_name}!")
                raise RuntimeError(f"Embedding loading failed for {dataset_name}")
            
            embeddings, valid_indices = result
            logger.info(
                f"Loaded embeddings shape for {dataset_name}: {embeddings.shape}; "
                f"valid samples used: {len(valid_indices)}"
            )
        else:
            # Other processes don't participate in offline loading
            return None, None
    else:
        # Extract embeddings using model
        if accelerator.is_main_process:
            logger.info(f"Extracting embeddings for {dataset_name.upper()}...")
        
        # All processes participate in embedding extraction
        result = extract_embeddings(dataset, extractor, model_type)
        
        # Synchronize all processes before continuing
        accelerator.wait_for_everyone()
        
        # Only main process continues with evaluation
        if not accelerator.is_main_process:
            return None, None  # Other processes return None
        
        # Check if embeddings were successfully extracted
        if result is None:
            logger.error(f"Failed to extract embeddings for {dataset_name}!")
            raise RuntimeError(f"Embedding extraction failed for {dataset_name}")
        
        embeddings, valid_indices = result
        logger.info(
            f"Embeddings shape for {dataset_name}: {embeddings.shape}; "
            f"valid samples used: {len(valid_indices)}"
        )
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = RidgeEvaluator(
        n_folds=args.n_folds,
        test_size=args.test_size,
        alpha=args.alpha,
        device=args.device,
        random_state=args.random_state
    )
    
    # Evaluate all outcomes for this dataset
    outcomes = list(dataset.OUTCOMES.keys())
    logger.info(f"Evaluating {len(outcomes)} outcomes for {dataset_name}: {outcomes}")
    
    # Evaluate each outcome
    results = {}
    for outcome in outcomes:
        try:
            outcome_results = evaluate_outcome(
                outcome=outcome,
                embeddings=embeddings,
                dataset=dataset,
                evaluator=evaluator,
                valid_indices=valid_indices
            )
            results[outcome] = outcome_results
            
            # Print brief summary
            if outcome_results['task_type'] == 'regression':
                test_r2 = outcome_results['test_results'].get('r2', None)
                logger.info(f"  {outcome}: R² = {test_r2:.3f}")
            else:
                test_acc = outcome_results['test_results'].get('accuracy', None)
                logger.info(f"  {outcome}: Accuracy = {test_acc:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to evaluate {outcome}: {e}")
            continue
    
    # Prepare dataset info for report
    dataset_info = {
        'dataset_name': dataset.get_name().upper(),
        'n_samples': int(embeddings.shape[0]),
        'n_train': int(embeddings.shape[0] * (1 - args.test_size)),
        'n_test': int(embeddings.shape[0] * args.test_size),
        'embedding_dim': embeddings.shape[1],
        'splits': dataset.df['split'].value_counts().to_dict() if 'split' in dataset.df.columns else {'all': len(dataset)}
    }
    
    # Add dataset-specific info
    if dataset_name == 'meld' and hasattr(dataset, 'get_split_statistics'):
        dataset_info['split_statistics'] = dataset.get_split_statistics()
    
    return results, dataset_info


def main():
    """Main evaluation pipeline with multi-GPU support."""
    args = parse_args()
    
    # Validate arguments
    if args.model_id is None and args.embedding_path is None:
        raise ValueError("Either --model_id or --embedding_path must be provided")
    
    # Determine which datasets to evaluate
    if args.dataset_list is not None and len(args.dataset_list) > 0:
        # Use the provided dataset list
        datasets_to_eval = args.dataset_list
    else:
        # Default: evaluate on all available datasets
        datasets_to_eval = ["iemocap", "meld"]
    
    # Initialize accelerator for potential multi-GPU execution
    # This will be a no-op if not launched with accelerate
    accelerator = Accelerator()
    
    # Set random seeds using accelerate's utility for consistency across processes
    set_seed(args.random_state)
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Only log on main process to avoid duplicate outputs
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("EMBEDDING EVALUATION PIPELINE")
        logger.info("=" * 60)
        if args.embedding_path:
            logger.info(f"Embeddings: {args.embedding_path} (offline)")
        else:
            logger.info(f"Model: {args.model_id}")
        logger.info(f"Datasets: {datasets_to_eval}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Dtype: {args.dtype}")
        logger.info(f"Cache: {not args.no_cache}")
        if accelerator.num_processes > 1:
            logger.info(f"Multi-GPU: {accelerator.num_processes} processes")
        logger.info("=" * 60)
    
    # Check if using offline embeddings
    if args.embedding_path:
        if accelerator.is_main_process:
            logger.info(f"Using offline embeddings from: {args.embedding_path}")
            logger.info("Skipping model loading...")
        extractor = None
        model_type = "offline"  # Placeholder type for offline embeddings
    else:
        # Determine model type
        if args.model_type == "auto":
            model_type = determine_model_type(args.model_id)
        else:
            model_type = args.model_type
        if accelerator.is_main_process:
            logger.info(f"Model type: {model_type}")
        
        # Initialize embedding extractor (pass the existing accelerator)
        if accelerator.is_main_process:
            logger.info("Initializing embedding extractor...")
        extractor = EmbeddingExtractor(
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            dtype=dtype,
            accelerator=accelerator  # Pass the existing accelerator
        )
        
        # Clear cache if requested (only on main process)
        if args.clear_cache and accelerator.is_main_process:
            logger.info("Clearing embedding cache...")
            extractor.clear_cache()
        
        # Wait for cache clearing to complete
        accelerator.wait_for_everyone()
    
    # Evaluate on each dataset
    all_results = {}
    all_dataset_info = {}
    
    for dataset_name in datasets_to_eval:
        try:
            results, dataset_info = evaluate_single_dataset(
                dataset_name, args, accelerator, extractor, model_type
            )
            
            # Only main process gets results
            if accelerator.is_main_process and results is not None:
                all_results[dataset_name] = results
                all_dataset_info[dataset_name] = dataset_info
                
        except Exception as e:
            logger.error(f"Failed to evaluate on {dataset_name}: {e}")
            if accelerator.is_main_process:
                import traceback
                traceback.print_exc()
            continue
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Only main process generates the final report
    if accelerator.is_main_process and all_results:
        # Generate and save report
        logger.info("\nGenerating evaluation report...")
        results_dir = "eval/encode/results"
        os.makedirs(results_dir, exist_ok=True)
        reporter = MetricsReporter(results_dir=results_dir)
        
        # If single dataset, use original format
        if len(all_results) == 1:
            dataset_name = list(all_results.keys())[0]
            reporter.print_summary(all_results[dataset_name])
            
            results_path = reporter.save_report(
                results=all_results[dataset_name],
                model_id=args.model_id if not args.embedding_path else f"Offline: {args.embedding_path}",
                dataset_info=all_dataset_info[dataset_name]
            )
        else:
            # For multiple datasets, create a combined report
            combined_dataset_info = {
                'dataset_name': f"Combined ({', '.join(datasets_to_eval)})",
                'datasets': all_dataset_info
            }
            
            # Print summaries for each dataset
            for dataset_name, results in all_results.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Results for {dataset_name.upper()}:")
                logger.info(f"{'='*60}")
                reporter.print_summary(results)
            
            # Save combined report with all results
            results_path = reporter.save_combined_report(
                all_results=all_results,
                model_id=args.model_id if not args.embedding_path else f"Offline: {args.embedding_path}",
                combined_dataset_info=combined_dataset_info
            )
        
        logger.info(f"\nEvaluation complete! Results saved to: {results_path}/")
        
        return all_results
    
    return None


if __name__ == "__main__":
    main()