#!/usr/bin/env python3
"""Test loading sharded safetensors checkpoints"""

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from model.whispa import WhiSPAModel
from model.config import WhiSPAConfig
from dotenv import load_dotenv

load_dotenv()


def test_load_sharded_checkpoint():
    """Test loading a checkpoint with sharded safetensors files"""
    
    checkpoint_path = '/mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/whispa-enc-3b-fix/step-50'
    
    print(f"Testing checkpoint loading from: {checkpoint_path}")
    
    # Check what files exist
    files = os.listdir(checkpoint_path)
    print("\nFiles in checkpoint directory:")
    for f in sorted(files):
        if f.endswith(('.safetensors', '.json', '.bin')):
            size = os.path.getsize(os.path.join(checkpoint_path, f))
            print(f"  {f} ({size / 1024**3:.2f} GB)")
    
    # Try to load the model
    try:
        print("\nLoading model...")
        model = WhiSPAModel.from_pretrained_local(checkpoint_path)
        print("✓ Successfully loaded model from sharded checkpoint!")
        
        # Verify model structure
        state_dict = model.state_dict()
        print(f"\nLoaded {len(state_dict)} parameters")
        
        # Check a few key parameters
        sample_keys = [
            "spectral_encoder.conv1.weight",
            "multi_modal_projector.linear_1.weight", 
            "language_model.model.embed_tokens.weight"
        ]
        
        print("\nSample parameter shapes:")
        for key in sample_keys:
            if key in state_dict:
                print(f"  {key}: {state_dict[key].shape}")
            else:
                print(f"  {key}: NOT FOUND")
        
        # Test setting stage and device
        model.set_stage('inference')
        model.config.device = 'cuda'
        model.config.dtype = torch.bfloat16
        model.eval()
        
        print("\n✓ Model configuration updated successfully")
        print(f"  Stage: {model.config.stage}")
        print(f"  Device: {model.config.device}")
        print(f"  Dtype: {model.config.dtype}")
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        raise


if __name__ == "__main__":
    test_load_sharded_checkpoint()
