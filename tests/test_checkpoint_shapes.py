#!/usr/bin/env python3
"""Test to diagnose tensor shape corruption in checkpoints"""

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from safetensors.torch import load_file
from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from dotenv import load_dotenv

load_dotenv()


def check_checkpoint_shapes(checkpoint_path: str):
    """Check if a saved checkpoint has correct tensor shapes"""
    
    print(f"Checking checkpoint: {checkpoint_path}")
    
    # Load the saved state dict
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"ERROR: {safetensors_path} not found")
        return
    
    saved_state = load_file(safetensors_path)
    
    # Load config and create a fresh model to get expected shapes
    config = WhiSPAConfig.from_pretrained(checkpoint_path)
    model = WhiSPAModel(config)
    expected_state = model.state_dict()
    
    # Compare shapes
    print("\nChecking tensor shapes...")
    shape_errors = []
    zero_size_tensors = []
    flattened_tensors = []
    
    for key in expected_state:
        if key not in saved_state:
            print(f"WARNING: {key} missing from checkpoint")
            continue
            
        expected_shape = expected_state[key].shape
        saved_shape = saved_state[key].shape
        
        if saved_shape != expected_shape:
            shape_errors.append((key, saved_shape, expected_shape))
            
            # Check if tensor is flattened
            if len(saved_shape) == 1 and saved_shape[0] == expected_state[key].numel():
                flattened_tensors.append(key)
            
            # Check if tensor has size 0
            if 0 in saved_shape:
                zero_size_tensors.append(key)
    
    # Report findings
    if shape_errors:
        print(f"\nERROR: Found {len(shape_errors)} shape mismatches:")
        for key, saved, expected in shape_errors[:10]:  # Show first 10
            print(f"  {key}: saved {saved} vs expected {expected}")
        
        if flattened_tensors:
            print(f"\n{len(flattened_tensors)} tensors appear to be flattened")
        
        if zero_size_tensors:
            print(f"\n{len(zero_size_tensors)} tensors have size 0")
            print("Zero-size tensors:")
            for key in zero_size_tensors[:10]:
                print(f"  {key}")
    else:
        print("✓ All tensor shapes are correct!")
    
    # Check parameter counts
    saved_params = sum(p.numel() for p in saved_state.values())
    expected_params = sum(p.numel() for p in expected_state.values())
    print(f"\nParameter count: saved={saved_params:,}, expected={expected_params:,}")
    
    return len(shape_errors) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    args = parser.parse_args()
    
    check_checkpoint_shapes(args.checkpoint_path)
