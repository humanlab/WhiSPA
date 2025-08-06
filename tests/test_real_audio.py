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
    
    # Load and preprocess audio
    waveform = load_and_preprocess_audio(audio_path)
    
    # Create processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # Process audio to get input features
    print("Processing audio to input features...")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    print(f"Input features shape: {input_features.shape}")
    
    # Test encode stage
    print("\n=== Testing ENCODE stage ===")
    config_encode = WhiSPAConfig(
        stage='encode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_encode = WhiSPAModel(config_encode)
    model_encode.eval()
    
    with torch.no_grad():
        spectral_embedding = model_encode(spectral_inputs=input_features)
        print(f"✓ Encode stage works!")
        print(f"  - Spectral embedding shape: {spectral_embedding.shape}")
        print(f"  - Embedding norm: {torch.norm(spectral_embedding, dim=1)}")
    
    # Test decode stage
    print("\n=== Testing DECODE stage ===")
    config_decode = WhiSPAConfig(
        stage='decode',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_decode = WhiSPAModel(config_decode)
    model_decode.eval()
    
    with torch.no_grad():
        spectral_latent = model_decode(spectral_inputs=input_features)
        print(f"✓ Decode stage works!")
        print(f"  - Spectral latent shape: {spectral_latent.shape}")
        
        # Test generation
        print("\n--- Testing Generation ---")
        try:
            generated_ids = model_decode.generate(
                spectral_inputs=input_features,
                max_length=50,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe"
            )
            print(f"✓ Generation works!")
            print(f"  - Generated IDs shape: {generated_ids.shape}")
            
            # Decode to text
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"  - Transcription: {transcription}")
            
        except Exception as e:
            print(f"⚠ Generation failed: {e}")
        
        # Test transcribe method
        print("\n--- Testing Transcribe Method ---")
        try:
            transcriptions = model_decode.transcribe(
                spectral_inputs=input_features,
                processor=processor,
                max_length=50,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe"
            )
            print(f"✓ Transcribe method works!")
            print(f"  - Transcriptions: {transcriptions}")
            
        except Exception as e:
            print(f"⚠ Transcribe method failed: {e}")
    
    # Test train_dec stage (simulation)
    print("\n=== Testing TRAIN_DEC stage (simulation) ===")
    config_train_dec = WhiSPAConfig(
        stage='train_dec',
        whisper_model_id='openai/whisper-tiny',
        dtype=torch.float32,
        device='cpu'
    )
    
    model_train_dec = WhiSPAModel(config_train_dec)
    model_train_dec.eval()
    
    # Create dummy labels for testing
    batch_size = input_features.shape[0]
    text_labels = torch.randint(0, 1000, (batch_size, 10))
    text_attention_mask = torch.ones(batch_size, 10)
    
    with torch.no_grad():
        spectral_latent, loss, lm_logits = model_train_dec(
            spectral_inputs=input_features,
            text_labels=text_labels,
            text_attention_mask=text_attention_mask
        )
        print(f"✓ Train_dec stage works!")
        print(f"  - Spectral latent shape: {spectral_latent.shape}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Logits shape: {lm_logits.shape}")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_with_real_audio() 