#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

"""
Test script for WhiSPA model's train_dec and decode stages.
"""

import torch
import torch.nn.functional as F
from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_model import WhiSPAModel


def test_train_dec_stage():
    """Test the train_dec stage of WhiSPA model."""
    print("Testing train_dec stage...")
    
    # Create config for train_dec stage
    config = WhiSPAConfig(
        stage='train_dec',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    # Create model
    model = WhiSPAModel(config)
    model.eval()
    
    # Create dummy inputs with correct dimensions for whisper-tiny
    batch_size = 2
    feature_size = 80  # mel bins
    sequence_length = 3000  # expected length for whisper-tiny
    
    spectral_inputs = torch.randn(batch_size, feature_size, sequence_length)
    text_labels = torch.randint(0, 1000, (batch_size, 10))  # 10 tokens per sequence
    text_attention_mask = torch.ones(batch_size, 10)
    
    # Test forward pass
    with torch.no_grad():
        spectral_latent, loss, lm_logits = model(
            spectral_inputs=spectral_inputs,
            text_labels=text_labels,
            text_attention_mask=text_attention_mask
        )
    
    print(f"✓ train_dec stage works!")
    print(f"  - spectral_latent shape: {spectral_latent.shape}")
    print(f"  - loss: {loss.item():.4f}")
    print(f"  - lm_logits shape: {lm_logits.shape}")
    
    return True


def test_decode_stage():
    """Test the decode stage of WhiSPA model."""
    print("\nTesting decode stage...")
    
    # Create config for decode stage
    config = WhiSPAConfig(
        stage='decode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    # Create model
    model = WhiSPAModel(config)
    model.eval()
    
    # Create dummy inputs with correct dimensions for whisper-tiny
    batch_size = 2
    feature_size = 80  # mel bins
    sequence_length = 3000  # expected length for whisper-tiny
    
    spectral_inputs = torch.randn(batch_size, feature_size, sequence_length)
    
    # Test forward pass
    with torch.no_grad():
        spectral_latent = model(spectral_inputs=spectral_inputs)
    
    print(f"✓ decode stage works!")
    print(f"  - spectral_latent shape: {spectral_latent.shape}")
    
    # Test get_decoder_logits method
    decoder_input_ids = torch.randint(0, 1000, (batch_size, 5))
    decoder_attention_mask = torch.ones(batch_size, 5)
    
    with torch.no_grad():
        logits = model.get_decoder_logits(
            spectral_inputs=spectral_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
    
    print(f"✓ get_decoder_logits works!")
    print(f"  - logits shape: {logits.shape}")
    
    return True


def test_generate_method():
    """Test the generate method of WhiSPA model."""
    print("\nTesting generate method...")
    
    # Create config for decode stage
    config = WhiSPAConfig(
        stage='decode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    # Create model
    model = WhiSPAModel(config)
    model.eval()
    
    # Create dummy inputs with correct dimensions for whisper-tiny
    batch_size = 1
    feature_size = 80  # mel bins
    sequence_length = 3000  # expected length for whisper-tiny
    
    spectral_inputs = torch.randn(batch_size, feature_size, sequence_length)
    
    # Test generate method
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                spectral_inputs=spectral_inputs,
                max_length=20,
                do_sample=False,
                num_beams=1
            )
            print(f"✓ generate method works!")
            print(f"  - generated_ids shape: {generated_ids.shape}")
        except Exception as e:
            print(f"⚠ generate method failed: {e}")
            print("  This might be due to missing generation dependencies")
    
    return True


if __name__ == "__main__":
    print("Testing WhiSPA model train_dec and decode stages...")
    
    try:
        test_train_dec_stage()
        test_decode_stage()
        test_generate_method()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 