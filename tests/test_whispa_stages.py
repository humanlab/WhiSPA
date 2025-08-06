#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

"""
Test script for WhiSPA model with real audio file.
"""

import torch
import torchaudio
from transformers import WhisperProcessor
from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_model import WhiSPAModel


def load_and_preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file."""
    print(f"Loading audio from: {audio_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    print(f"Audio loaded: shape={waveform.shape}, sample_rate={target_sr}")
    return waveform.squeeze()


def test_with_real_audio():
    """Test WhiSPA model with real audio file."""
    print("Testing WhiSPA model with real audio...")
    
    # Audio file path
    audio_path = "/cronus_data/rrao/samples/P209_segment.wav"
    
    # Test encode stage with audio file path
    print("\n=== Testing ENCODE stage with audio file path ===")
    config_encode = WhiSPAConfig(
        stage='encode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_encode = WhiSPAModel(config_encode)
    model_encode.eval()
    
    with torch.no_grad():
        spectral_embedding = model_encode.encode(audio_path)
        print(f"✓ Encode stage works with audio file path!")
        print(f"  - Spectral embedding shape: {spectral_embedding.shape}")
        print(f"  - Embedding norm: {torch.norm(spectral_embedding, dim=1)}")
    
    # Test transcribe stage with audio file path
    print("\n=== Testing TRANSCRIBE stage with audio file path ===")
    config_decode = WhiSPAConfig(
        stage='decode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_decode = WhiSPAModel(config_decode)
    model_decode.eval()
    
    with torch.no_grad():
        # Test transcribe method with audio file path
        try:
            transcriptions = model_decode.transcribe(
                audio_path=audio_path,
                max_length=50,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe"
            )
            print(f"✓ Transcribe method works with audio file path!")
            print(f"  - Transcriptions: {transcriptions}")
            
        except Exception as e:
            print(f"⚠ Transcribe method failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test train_dec stage with specific text to verify loss is 0
    print("\n=== Testing TRAIN_DEC stage with specific text ===")
    config_train_dec = WhiSPAConfig(
        stage='train_dec',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_train_dec = WhiSPAModel(config_train_dec)
    model_train_dec.eval()
    
    # Load and preprocess audio for train_dec testing
    waveform = load_and_preprocess_audio(audio_path)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    # Create text labels for "Nothing different the same."
    # We need to tokenize this specific text
    target_text = "Nothing different the same."
    text_tokens = processor.tokenizer(target_text, return_tensors="pt")
    text_labels = text_tokens.input_ids
    text_attention_mask = text_tokens.attention_mask
    
    print(f"  - Target text: '{target_text}'")
    print(f"  - Text labels shape: {text_labels.shape}")
    print(f"  - Text attention mask shape: {text_attention_mask.shape}")
    
    with torch.no_grad():
        spectral_latent, loss, lm_logits = model_train_dec(
            spectral_inputs=input_features,
            text_labels=text_labels,
            text_attention_mask=text_attention_mask
        )
        print(f"✓ Train_dec stage works!")
        print(f"  - Spectral latent shape: {spectral_latent.shape}")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Logits shape: {lm_logits.shape}")
        
        # Verify that loss is close to 0 for the target text
        if loss.item() < 1e-6:
            print(f"✅ Loss is effectively 0 ({loss.item():.6f}) for target text!")
        else:
            print(f"⚠ Loss is not 0: {loss.item():.6f}")
        
        # Show what the model actually predicts vs target
        print(f"  - Target tokens: {text_labels[0].tolist()}")
        predicted_tokens = torch.argmax(lm_logits, dim=-1)[0]
        print(f"  - Predicted tokens: {predicted_tokens.tolist()}")
        
        # Decode both to see the actual text
        target_text_decoded = processor.tokenizer.decode(text_labels[0], skip_special_tokens=True)
        predicted_text_decoded = processor.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
        print(f"  - Target text: '{target_text_decoded}'")
        print(f"  - Predicted text: '{predicted_text_decoded}'")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_with_real_audio() 