#!/usr/bin/env python3
"""Test that model loads correctly on CPU for distributed training"""

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from dotenv import load_dotenv

load_dotenv()


def test_cpu_model_loading():
    """Test that model can be loaded and kept on CPU"""
    
    # Test 1: Create new model on CPU
    print("Test 1: Creating new model on CPU...")
    config = WhiSPAConfig(
        backbone_model_id="mistralai/Voxtral-Mini-3B-2507",
        stage='train_enc',
        device='cpu',  # Explicitly CPU
        dtype=torch.float32,
    )
    model = WhiSPAModel(config)
    
    # Check all components are on CPU
    for name, param in model.named_parameters():
        if param.device.type != 'cpu':
            print(f"ERROR: Parameter {name} is on {param.device}, not CPU!")
            return False
    print("✓ All parameters are on CPU")
    
    # Test 2: Load from checkpoint and move to CPU
    checkpoint_path = '/mnt/vast/share/checkpoints/rajath-cmd/WhiSPA/whispa-enc-3b-fix/step-50'
    if os.path.exists(checkpoint_path):
        print(f"\nTest 2: Loading model from checkpoint and moving to CPU...")
        try:
            model = WhiSPAModel.from_pretrained_local(checkpoint_path)
            # Override device config
            model.config.device = torch.device('cpu')
            # Move model to CPU
            model = model.cpu()
            
            # Verify all on CPU
            cpu_count = 0
            for name, param in model.named_parameters():
                if param.device.type == 'cpu':
                    cpu_count += 1
                else:
                    print(f"ERROR: Parameter {name} is on {param.device}, not CPU!")
                    return False
            
            print(f"✓ All {cpu_count} parameters successfully moved to CPU")
            
        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
            return False
    else:
        print(f"\nSkipping checkpoint test - {checkpoint_path} not found")
    
    print("\n✓ CPU loading tests PASSED")
    return True


if __name__ == "__main__":
    test_cpu_model_loading()
