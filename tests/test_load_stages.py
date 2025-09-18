#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

"""
Test script for WhiSPA model loaded from local checkpoint.
Tests both single and batched calls across all stages.

This script loads a model from an existing checkpoint once and tests all stages
by dynamically changing model.config.stage:
- encode() functionality (stage='inference')
- transcribe() functionality (stage='inference')
- forward() in train_dec stage (stage='train_dec')
- forward() in train_enc stage (stage='train_enc')

The checkpoint path is determined by the CHECKPOINT_DIR environment variable 
(defaults to {CHECKPOINT_DIR}/Voxtral-Mini-3B).

Note: The model is loaded once and its stage is changed for each test to avoid
redundant loading operations.
"""

import torch
import random
import glob
import time
from dotenv import load_dotenv

load_dotenv()

from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel

# Constants
CHECKPOINT_PATH = os.path.join(os.getenv("CHECKPOINT_DIR"), "Voxtral-Mini-3B")
AUDIO_DIR = os.getenv("AUDIO_SAMPLES_DIR")
BATCH_SIZE = 8
DEVICE = "cpu"


def get_audio_files(audio_dir: str, num_files: int = None):
    """Get audio files from directory."""
    audio_patterns = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for pattern in audio_patterns:
        audio_files.extend(glob.glob(os.path.join(audio_dir, pattern)))
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    if num_files and num_files < len(audio_files):
        audio_files = random.sample(audio_files, num_files)
    
    return sorted(audio_files)


def test_with_real_audio_from_checkpoint():
    """Test WhiSPA model loaded from checkpoint on single and batch audio samples for all stages.
    
    Loads a single model instance and tests all stages by changing model.config.stage.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        raise ValueError(f"Checkpoint path does not exist: {CHECKPOINT_PATH}")
    
    print("=" * 80)
    print("Testing WhiSPA model loaded from local checkpoint")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print("=" * 80)
    
    # Get audio files
    try:
        all_audio_files = get_audio_files(AUDIO_DIR)
        print(f"Found {len(all_audio_files)} audio files in {AUDIO_DIR}")
        
        # Select files for testing
        single_audio = all_audio_files[0]
        batch_audio = random.sample(all_audio_files, min(BATCH_SIZE, len(all_audio_files)))
        
        print(f"\nSingle audio file: {os.path.basename(single_audio)}")
        print(f"Batch audio files ({len(batch_audio)}):")
        for f in batch_audio:
            print(f"  - {os.path.basename(f)}")
        
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return
    
    print("\nLoading model from checkpoint...")
    start_time = time.time()
    model = WhiSPAModel.from_pretrained_local(CHECKPOINT_PATH).eval()
    elapsed = time.time() - start_time
    print(f"✅ Model loaded from checkpoint in {elapsed:.3f} seconds")
    print(f"Initial stage: {model.config.stage}")
    
    # Test configurations: (audio_input, tag, description)
    test_cases = [
        (single_audio, "single", "Single audio inference"),
        (batch_audio, f"batch-{len(batch_audio)}", f"Batch inference with {len(batch_audio)} audio files")
    ]
    
    for audio_input, tag, description in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test Case: {description}")
        print(f"Tag: {tag}")
        print(f"{'=' * 80}")
        
        # ENCODE TEST
        print("\n1. Testing encode()")
        print("-" * 40)
        
        # Set stage to inference for encode
        model.config.stage = 'inference'
        print(f"  Set model stage to: {model.config.stage}")
        
        with torch.no_grad():
            try:
                spectral_embeddings = model.encode(audio_input, language="en")
                print(f"  ✅ encode() successful")
                print(f"     Output shape: {spectral_embeddings.shape}")
                print(f"     Dtype: {spectral_embeddings.dtype}")
                print(f"     Device: {spectral_embeddings.device}")
                
                # Verify expected shape
                if isinstance(audio_input, list):
                    expected_batch = len(audio_input)
                else:
                    expected_batch = 1
                
                assert spectral_embeddings.shape[0] == expected_batch, \
                    f"Batch size mismatch: expected {expected_batch}, got {spectral_embeddings.shape[0]}"
                
                # For Voxtral, embeddings should be (batch_size, 3072)
                assert spectral_embeddings.shape[1] == 3072, \
                    f"Embedding dimension mismatch: expected 3072, got {spectral_embeddings.shape[1]}"
                    
            except Exception as e:
                print(f"  ❌ encode() failed: {e}")
                import traceback
                traceback.print_exc()
        
        # TRANSCRIBE TEST
        print("\n2. Testing transcribe()")  
        print("-" * 40)
        
        # Ensure stage is inference for transcribe
        model.config.stage = 'inference'
        print(f"  Set model stage to: {model.config.stage}")
        
        with torch.no_grad():
            try:
                transcriptions = model.transcribe(
                    audio=audio_input, 
                    language="en", 
                    max_new_tokens=100,  # Reduced for faster testing
                    do_sample=False, 
                    num_beams=1
                )
                print(f"  ✅ transcribe() successful")
                print(f"     Generated {len(transcriptions)} transcription(s)")
                
                for i, transcript in enumerate(transcriptions):
                    preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
                    print(f"     [{i}]: '{preview}'")
                    
                # Verify we got the right number of transcriptions
                expected_transcriptions = len(audio_input) if isinstance(audio_input, list) else 1
                assert len(transcriptions) == expected_transcriptions, \
                    f"Transcription count mismatch: expected {expected_transcriptions}, got {len(transcriptions)}"
                    
            except Exception as e:
                print(f"  ❌ transcribe() failed: {e}")
                import traceback
                traceback.print_exc()
        
        # TRAIN_DEC TEST
        print("\n3. Testing forward(train_dec)")
        print("-" * 40)
        
        # Set stage to train_dec
        model.config.stage = 'train_dec'
        print(f"  Set model stage to: {model.config.stage}")
        
        try:
            # Use model's processor to prepare inputs with spans
            proc_inputs = model.processor(audio_input, language="en")
            spectral_inputs = proc_inputs["spectral_inputs"]
            sample_spans = proc_inputs["sample_spans"]
            input_ids = proc_inputs["text_input_ids"]
            attention_mask = proc_inputs["text_attention_mask"]
            labels = input_ids.clone()
            
            with torch.no_grad():
                outputs = model(
                    spectral_inputs=spectral_inputs,
                    sample_spans=sample_spans,
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                    text_labels=labels,
                )
                print(f"  ✅ train_dec forward() successful")
                print(f"     Loss: {outputs.loss.item():.4f}")
                print(f"     Logits shape: {outputs.logits.shape}")

                # Verify batch size in output
                expected_batch = len(audio_input) if isinstance(audio_input, list) else 1
                assert sample_spans.shape[0] == expected_batch, \
                    f"Sample spans batch mismatch: expected {expected_batch}, got {sample_spans.shape[0]}"
                    
        except Exception as e:
            print(f"  ❌ train_dec forward() failed: {e}")
            import traceback
            traceback.print_exc()
        
        # TRAIN_ENC TEST
        print("\n4. Testing forward(train_enc)")
        print("-" * 40)
        
        # Set stage to train_enc
        model.config.stage = 'train_enc'
        print(f"  Set model stage to: {model.config.stage}")
        
        try:
            # Prepare inputs (reuse from train_dec if available, or create new)
            if 'spectral_inputs' not in locals():
                proc_inputs = model.processor(audio_input, language="en")
                spectral_inputs = proc_inputs["spectral_inputs"]
                sample_spans = proc_inputs["sample_spans"]
            
            # Create random target embeddings per sample
            B = sample_spans.shape[0]
            hidden_size = model.voxtral.config.text_config.hidden_size
            target_audio = torch.randn(B, hidden_size).to(model.config.device)
            target_text = torch.randn(B, hidden_size).to(model.config.device)
            target_psych = torch.randn(B, hidden_size).to(model.config.device)
            
            with torch.no_grad():
                out = model(
                    spectral_inputs=spectral_inputs,
                    sample_spans=sample_spans,
                    target_audio_embs=target_audio,
                    target_text_embs=target_text,
                    target_psych_embs=target_psych,
                )
                print(f"  ✅ train_enc forward() successful")
                print(f"     Total loss: {out['total_loss'].item():.4f}")
                print(f"     Acoustic loss: {out['acoustic_loss'].item():.4f}")
                print(f"     Semantic loss: {out['semantic_loss'].item():.4f}")
                print(f"     Affective loss: {out.get('affective_loss', torch.tensor(0.0)).item():.4f}")
                
        except Exception as e:
            print(f"  ❌ train_enc forward() failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("All tests completed!")
    print(f"{'=' * 80}")
    
    # Clean up model
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Run tests
    test_with_real_audio_from_checkpoint()
