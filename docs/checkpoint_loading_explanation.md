# Checkpoint Loading Fix Explanation

## Previous Issue

The code was trying to use `accelerator.load_model()` which doesn't exist in the HuggingFace Accelerate library. This was causing the error:
```
'Accelerator' object has no attribute 'load_model'
```

## The Correct Approach

When resuming training with Accelerate:

1. **Load model weights BEFORE `accelerator.prepare()`**
   - Model weights must be loaded into the model before it's wrapped by Accelerate
   - This ensures the correct weights are distributed across devices

2. **Load accelerator state AFTER `accelerator.prepare()`**
   - This restores optimizer, scheduler, and RNG states
   - Must be done after the model is prepared

## Implementation

### In `main()` function:
```python
# Create model
model = WhiSPAModel(whispa_cfg)

# Load weights from checkpoint if resuming (BEFORE prepare)
if args.resume_from is not None:
    loaded_model = WhiSPAModel.from_pretrained_local(checkpoint_dir)
    model.load_state_dict(loaded_model.state_dict())
    
model.set_stage(whispa_cfg.stage)
model.train()
```

### In `train_loop()` function:
```python
# Prepare model with accelerator
model, optimizer, ... = accelerator.prepare(model, optimizer, ...)

# Load accelerator state if resuming
if resume_dir is not None:
    accelerator.load_state(checkpoint_dir)  # Restores optimizer, scheduler, RNG
```

## Key Points

1. **Model Loading**: Use `WhiSPAModel.from_pretrained_local()` which now supports:
   - Single `model.safetensors` files
   - Sharded safetensors with `model.safetensors.index.json`
   - Legacy `pytorch_model.bin` files
   - FSDP checkpoints (`pytorch_model_fsdp.bin`)

2. **Saving**: Continue using `accelerator.save_model()` which:
   - Properly handles distributed models
   - Creates sharded files for large models
   - Preserves all tensor shapes correctly

3. **State Management**:
   - Model weights: Loaded before `prepare()`
   - Training state: Loaded after `prepare()` via `accelerator.load_state()`

This approach ensures compatibility with all distributed training strategies (DDP, FSDP) and properly restores both model weights and training state.
