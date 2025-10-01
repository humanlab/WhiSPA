#!/usr/bin/env python3
"""Test that model checkpointing works correctly with Accelerate"""

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
import tempfile
import shutil
from accelerate import Accelerator
from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()


def test_accelerate_save_load():
    """Test that accelerator.save_model preserves model weights correctly"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize accelerator
        accelerator = Accelerator()
        
        # Create model config
        cfg = WhiSPAConfig(
            backbone_model_id="mistralai/Voxtral-Mini-3B-2507",
            stage='train_enc',  # Test with frozen language model
            device=accelerator.device,
            dtype=torch.bfloat16,
        )
        
        # Create and prepare model
        model = WhiSPAModel(cfg)
        model = accelerator.prepare(model)
        
        # Get a sample of state dict before saving
        original_state = accelerator.unwrap_model(model).state_dict()
        sample_keys = list(original_state.keys())
        
        # Print some tensor shapes for debugging
        print("Original tensor shapes:")
        for k in sample_keys:
            print(f"  {k}: {original_state[k].shape}")
        
        # Save using accelerator
        save_dir = os.path.join(tmpdir, "test_checkpoint")
        accelerator.save_model(model, save_dir, safe_serialization=True)
        
        # Also save config
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.config.save_pretrained(save_dir)
        
        # Check saved files
        assert os.path.exists(os.path.join(save_dir, "model.safetensors")), "model.safetensors not found"
        assert os.path.exists(os.path.join(save_dir, "config.json")), "config.json not found"
        
        # Load the saved weights directly
        saved_state = load_file(os.path.join(save_dir, "model.safetensors"))
        
        # Check shapes match
        print("\nSaved tensor shapes:")
        mismatches = []
        for k in sample_keys:
            saved_shape = saved_state[k].shape
            orig_shape = original_state[k].shape
            print(f"  {k}: {saved_shape}")
            if saved_shape != orig_shape:
                mismatches.append(f"{k}: saved {saved_shape} vs original {orig_shape}")
        
        if mismatches:
            print("\nERROR: Shape mismatches found:")
            for m in mismatches:
                print(f"  {m}")
            raise AssertionError("Tensor shapes corrupted during save!")
        
        # Test loading with a fresh model
        new_model = WhiSPAModel.from_pretrained_local(save_dir)
        new_state = new_model.state_dict()
        
        # Verify all weights match
        for k in original_state:
            if k in new_state:
                orig_tensor = original_state[k].cpu()
                new_tensor = new_state[k].cpu()
                if orig_tensor.shape != new_tensor.shape:
                    raise AssertionError(f"Shape mismatch after load for {k}: {new_tensor.shape} vs {orig_tensor.shape}")
                if not torch.allclose(orig_tensor, new_tensor, rtol=1e-5, atol=1e-5):
                    raise AssertionError(f"Value mismatch for {k}")
        
        print("\n✓ Accelerate save/load test PASSED")
        print(f"✓ All tensor shapes preserved correctly")
        print(f"✓ All tensor values match")


if __name__ == "__main__":
    test_accelerate_save_load()
