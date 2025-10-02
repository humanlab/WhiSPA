# Embedding Evaluation Toolkit

This toolkit evaluates audio and text embedding models on the IEMOCAP and MELD datasets for emotion recognition and speech analysis tasks.

## Features

- **Multi-modal Support**: Evaluate both audio embeddings (WhiSPA, Whisper, etc.) and text embeddings (Qwen, BERT, etc.)
- **Offline Embeddings**: Evaluate pre-computed embeddings without loading models
- **Multi-GPU Support**: Distributed embedding extraction across multiple GPUs using Accelerate
- **Comprehensive Metrics**: Regression (MSE, MAE, RMSE, R², Pearson/Spearman correlation) and classification (accuracy, F1, ROC-AUC) metrics
- **GPU Acceleration**: Ridge regression with GPU support for faster evaluation
- **Caching**: Automatic caching of embeddings to `/tmp/WhiSPA/eval/encode` for efficient re-evaluation
- **K-Fold Cross-Validation**: Stratified k-fold CV with configurable folds
- **Detailed Reporting**: Markdown and JSON reports with tables and statistics

## Installation

Ensure you have the WhiSPA environment set up:

```bash
conda activate whispa
```

## Usage

### Basic Evaluation (Single GPU)

**Important:** When running with just `python`, only a single GPU will be used, even if multiple are available.

Evaluate WhiSPA audio embeddings on IEMOCAP:
```bash
python eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type audio \
    --dataset_list iemocap \
    --num_workers 32
```

Evaluate Qwen text embeddings on MELD:
```bash
python eval/encode/run.py \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --model_type text \
    --dataset_list meld \
    --num_workers 32
```

Evaluate on both datasets (default when --dataset_list is not specified):
```bash
python eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type audio \
    --num_workers 32
```

### Multi-GPU Evaluation

**To use multiple GPUs, you MUST launch with `accelerate launch`:**

```bash
# Use all available GPUs (auto-detect)
accelerate launch --multi_gpu eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/whispa-enc-3b/step-49410 \
    --model_type audio \
    --batch_size 16 \
    --num_workers 32

# Use specific number of GPUs (e.g., 4 GPUs)
accelerate launch --num_processes 4 eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type audio \
    --dataset_list iemocap meld \
    --batch_size 16 \
    --num_workers 32

# Use 2 GPUs for Qwen text embeddings  
accelerate launch --num_processes 2 eval/encode/run.py \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --model_type text \
    --dataset_list iemocap meld \
    --batch_size 64 \
    --num_workers 32
```

### Performance Tips

1. **Multi-GPU**: Use `accelerate launch --num_processes N` where N = number of GPUs
2. **Batch Size**: Increase batch size for better GPU utilization (e.g., 32-64)
3. **Workers**: Set `--num_workers` to 2-4x number of CPU cores
4. **Caching**: Embeddings are cached by default; use `--no_cache` to disable
5. **Mixed Precision**: Use `--dtype float16` for faster processing with minimal quality loss

### Offline Embeddings Evaluation

You can evaluate pre-computed embeddings without loading the model:

#### Single Dataset (Direct .npz file)
```bash
python eval/encode/run.py \
    --embedding_path /path/to/iemocap_embeddings.npz \
    --dataset_list iemocap
```

#### Multiple Datasets (Directory)
```bash
# Directory structure:
# embeddings_dir/
# ├── iemocap.npz  (or iemocap_embeddings.npz)
# └── meld.npz     (or meld_embeddings.npz)

python eval/encode/run.py \
    --embedding_path /path/to/embeddings_dir/ \
    --dataset_list iemocap meld
```

#### Embedding File Format
The `.npz` file should contain:
```python
{
    'embeddings': np.ndarray,  # shape: (n_samples, embedding_dim), required
    'indices': np.ndarray,     # valid sample indices, optional (defaults to all)
    'n_inputs': int,          # total number of inputs, optional
}
```

Example of creating compatible embeddings:
```python
import numpy as np

# Your embeddings: (n_samples, embedding_dim)
embeddings = model.encode(data)  

# Save in compatible format
np.savez_compressed(
    'iemocap_embeddings.npz',
    embeddings=embeddings,
    indices=np.arange(len(embeddings)),  # All samples valid
    n_inputs=len(embeddings)
)
```

### Advanced Options

```bash
python eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type auto \
    --dataset_list iemocap \
    --n_folds 10 \
    --test_size 0.2 \
    --alpha 1.0 \
    --batch_size 32 \
    --device cuda \
    --dtype float32 \
    --cache_dir /tmp/WhiSPA/eval/encode \
    --random_state 42

# For MELD dataset
python eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type auto \
    --dataset_list meld \
    --random_state 42

# Evaluate on both datasets
python eval/encode/run.py \
    --model_id /mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/Voxtral-Mini-3B \
    --model_type auto \
    --dataset_list iemocap meld \
    --random_state 42
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_id` | Model identifier (HuggingFace ID or local path) | Required* |
| `--model_type` | Type of embeddings (auto/audio/text) | auto |
| `--embedding_path` | Path to pre-computed embeddings (.npz file or directory) | None |
| `--dataset_list` | List of datasets to evaluate | All (iemocap meld) |
| `--n_folds` | Number of CV folds | 10 |
| `--test_size` | Test set proportion | 0.2 |
| `--alpha` | Ridge regularization | 1.0 |
| `--batch_size` | Batch size for extraction | 32 |
| `--num_workers` | Workers for parallel processing | 32 |
| `--device` | Computation device | cuda |
| `--dtype` | Model precision | float32 |
| `--cache_dir` | Embedding cache directory | /tmp/WhiSPA/eval/encode |
| `--no_cache` | Disable caching | False |
| `--clear_cache` | Clear cache before running | False |
| `--random_state` | Random seed | 42 |

*Note: Either `--model_id` or `--embedding_path` must be provided.

## Evaluated Outcomes

### IEMOCAP Dataset

#### Regression Tasks
- **Emotion Intensities**: frustrated, angry, sad, disgust, excited, fear, neutral, surprise, happy
- **Emotional Dimensions**: EmoAct (activation), EmoVal (valence), EmoDom (dominance)
- **Speech Features**: speaking_rate

#### Classification Tasks
- **gender**: Binary classification (Male/Female)
- **major_emotion**: Multi-class emotion classification

### MELD Dataset

#### Classification Tasks
- **speaker**: Multi-class speaker identification (with rare speakers grouped as "Other")
- **emotion**: Multi-class emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise)
- **sentiment**: Three-class sentiment classification (negative, neutral, positive)

## Output

Results are saved to `eval/encode/results/YY-MM-DD_HH-MM-SS-MS/` directory containing:
- `summary.md`: Formatted markdown report with all evaluation metrics
- `results.json`: Raw results in JSON format for further analysis

### Report Structure

#### Single Dataset Report
1. **Dataset Information**: Sample counts, splits, embedding dimensions
2. **Regression Results**: Tables with cross-validation and test metrics
3. **Classification Results**: Tables with accuracy, F1, ROC-AUC, confusion matrices
4. **Summary Statistics**: Averaged metrics across all tasks

#### Multiple Datasets Report
1. **Per-Dataset Sections**: Each dataset gets its own section with:
   - Dataset information and statistics
   - Regression task results
   - Classification task results
2. **Combined Summary Statistics**: Aggregated metrics across all datasets and tasks

## Testing

Run the test suite to verify installation:

```bash
python tests/test_eval_encode.py
```

This tests:
- Dataset loading
- Embedding extraction
- Ridge evaluation
- Metrics reporting

## Examples

### Example 1: Evaluate WhiSPA on IEMOCAP Emotion Recognition

```bash
python eval/encode/run.py \
    --model_id /mnt/vast/home/rajath-cmd/WhiSPA/checkpoints/whispa_best \
    --model_type audio \
    --dataset_list iemocap \
    --n_folds 5
```

### Example 2: Compare Text vs Audio Embeddings on MELD

```bash
# Evaluate text embeddings on MELD
python eval/encode/run.py \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --model_type text \
    --dataset_list meld

# Evaluate audio embeddings on MELD  
python eval/encode/run.py \
    --model_id mistralai/Voxtral-Mini-3B-2507 \
    --model_type audio \
    --dataset_list meld
```

### Example 3: Quick Test with Subset

```bash
# Quick test on IEMOCAP
python eval/encode/run.py \
    --model_id sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_list iemocap \
    --n_folds 3 \
    --batch_size 64

# Quick test on MELD
python eval/encode/run.py \
    --model_id sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_list meld \
    --n_folds 3 \
    --batch_size 64

# Evaluate on both datasets
python eval/encode/run.py \
    --model_id sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_list iemocap meld \
    --n_folds 3
```

## Architecture

The toolkit consists of modular components:

1. **dataset.py**: Dataset loaders for IEMOCAP and MELD with preprocessing
2. **embeddings.py**: Embedding extraction for various models
3. **evaluator.py**: GPU-accelerated Ridge regression with k-fold CV
4. **metrics.py**: Metrics computation and report generation
5. **run.py**: Main pipeline orchestrator

## Notes

- Embeddings are cached to `/tmp/WhiSPA/eval/encode` by default for efficiency
- Use `--clear_cache` to force re-extraction
- GPU is recommended for faster Ridge regression
- Stratified k-fold ensures balanced splits for classification
- Results include both cross-validation and held-out test metrics
