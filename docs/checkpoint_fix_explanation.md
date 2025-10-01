# Checkpoint Corruption Fix Explanation

## Problem

The model checkpointing in `train.py` was saving corrupted tensors with incorrect shapes:
- Some tensors were flattened (e.g., shape `[491520]` instead of `[1280, 128, 3]`)
- Some tensors had size 0 (especially language model parameters)
- This made the saved checkpoints unusable for resuming training

## Root Cause

The issue was in the `save_rotating_checkpoint` function which was:
1. Moving the model to CPU before saving (`unwrapped_model.cpu()`)
2. Manually unwrapping and saving the model instead of using Accelerate's built-in methods
3. This caused tensor corruption, especially with frozen parameters in distributed training

## Solution

The fix simplifies the checkpointing logic and uses Accelerate's proper APIs:

### 1. Use `accelerator.save_model()`
- This method properly handles distributed models (DDP/FSDP)
- Correctly saves all parameters including frozen ones
- Handles device placement automatically
- No need to manually move to CPU or unwrap

### 2. Use `accelerator.load_model()`
- Properly loads model weights in distributed context
- Handles frozen parameters correctly
- Must be called on the prepared model

### 3. Simplified checkpoint structure
- Model weights: saved via `accelerator.save_model()` 
- Config: saved separately via `config.save_pretrained()`
- Training state: saved as JSON and PT files
- Accelerator state: saved via `accelerator.save_state()`

## Key Changes

1. **Removed manual model manipulation**:
   - No more `model.cpu()` before saving
   - No more manual eval/train mode switching
   - Let Accelerate handle the complexity

2. **Fixed model loading on resume**:
   - Added `accelerator.load_model()` call when resuming
   - Loads weights after model is prepared by accelerator

3. **Removed redundant code**:
   - Deleted unused `save_checkpoint()` function
   - Simplified checkpoint rotation logic
   - Consolidated imports

## Testing

Created test scripts to verify the fix:
- `test_checkpoint_shapes.py`: Diagnoses tensor shape issues in saved checkpoints
- `test_frozen_params.py`: Verifies frozen parameters are included in state_dict
- `test_accelerate_checkpoint.py`: Tests save/load with Accelerate

## Important Notes

- Always use `accelerator.save_model()` for distributed training
- The model must be prepared by accelerator before loading weights
- Config must be saved separately as `save_model()` only saves weights
- Frozen parameters are included in state_dict even with `requires_grad=False`
