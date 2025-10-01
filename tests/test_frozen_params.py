#!/usr/bin/env python3
"""Test that frozen parameters are properly included in state_dict"""

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


def test_frozen_params_in_state_dict():
    """Verify that frozen parameters are included in state_dict"""
    
    # Create model in train_enc stage (language model frozen)
    cfg = WhiSPAConfig(
        backbone_model_id="mistralai/Voxtral-Mini-3B-2507",
        stage='train_enc',
        device='cpu',
        dtype=torch.float32,
    )
    model = WhiSPAModel(cfg)
    
    # Get state dict
    state = model.state_dict()
    
    # Count parameters by module
    spectral_params = sum(1 for k in state if k.startswith('spectral_encoder'))
    projector_params = sum(1 for k in state if k.startswith('multi_modal_projector'))
    language_params = sum(1 for k in state if k.startswith('language_model'))
    
    print(f"State dict contains:")
    print(f"  Spectral encoder params: {spectral_params}")
    print(f"  Projector params: {projector_params}")
    print(f"  Language model params: {language_params}")
    print(f"  Total params: {len(state)}")
    
    # Check requires_grad
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1
    
    print(f"\nParameter gradients:")
    print(f"  Trainable: {trainable}")
    print(f"  Frozen: {frozen}")
    
    # Verify language model params are in state dict even though frozen
    assert language_params > 0, "Language model parameters missing from state_dict!"
    
    # Check some specific language model tensors
    lm_keys = [k for k in state if k.startswith('language_model')][:5]
    print(f"\nSample language model tensors in state_dict:")
    for k in lm_keys:
        print(f"  {k}: shape={state[k].shape}")
    
    print("\n✓ Frozen parameters are correctly included in state_dict")


if __name__ == "__main__":
    test_frozen_params_in_state_dict()
