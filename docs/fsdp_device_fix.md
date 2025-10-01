# FSDP Device Placement Fix

## Problem

When using distributed training with FSDP (Fully Sharded Data Parallel), we get errors like:
```
ValueError: Inconsistent compute device and `device_id` on rank 5: cuda:0 vs cuda:5
```

This happens because:
1. The model is initialized on a specific CUDA device (cuda:0)
2. FSDP expects each rank to use its own GPU (rank 5 → cuda:5)
3. The model's forward() method moves tensors to specific devices

## Solution

Keep the model on CPU before `accelerator.prepare()`:

1. **In build_config()**: Set default device to CPU
   ```python
   model_cfg.setdefault("device", "cpu")
   ```

2. **Before model creation**: Ensure config uses CPU
   ```python
   whispa_cfg.device = torch.device("cpu")
   ```

3. **After loading from checkpoint**: Force model to CPU
   ```python
   model.config.device = torch.device("cpu")
   model = model.cpu()
   ```

## Why This Works

- FSDP needs full control over device placement
- When model is on CPU, FSDP can shard and place it correctly on each rank
- Accelerate handles moving model parts to appropriate GPUs

## Remaining Issue

The WhiSPAModel has device-specific calls in its forward() method:
```python
spectral_inputs.to(self.spectral_encoder.device)
text_input_ids.to(self.config.device)
```

These should work correctly after `accelerator.prepare()` moves the model to the right devices.

## Key Points

1. **Always** keep models on CPU before `accelerator.prepare()`
2. Let Accelerate handle all device placement for distributed training
3. Don't manually move models to specific CUDA devices in distributed setups
