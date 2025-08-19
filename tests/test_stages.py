#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

"""
Test script for WhiSPA model with real audio file.
"""

import torch
from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel


def test_with_real_audio():
    """Test WhiSPA model end-to-end on 1 and 2 audio samples for all stages."""
    print("Testing WhiSPA model with real audio...")

    model_id = "mistralai/Voxtral-Mini-3B-2507"
    audio1 = "/mnt/vast/data/speech/gigaspeech/data/data/audio/dev_files/dev_chunks_0000/dev_chunks_0000/POD1000000027_S0000288.wav"
    audio2 = "/mnt/vast/data/speech/gigaspeech/data/data/audio/dev_files/dev_chunks_0000/dev_chunks_0000/POD1000000004_S0000014.wav"

    for audio_input, tag in [(audio1, "1-audio"), ([audio1, audio2], "2-audio")]:
        print(f"\n=== Case: {tag} ===")

        # ENCODE
        print("- encode()")
        config_encode = WhiSPAConfig(stage='encode', audio_model_id=model_id, dtype=torch.bfloat16, device='cpu')
        model_encode = WhiSPAModel(config_encode).eval()
        with torch.no_grad():
            try:
                spectral_embeddings = model_encode.encode(audio_input, language="en")
                print("  ✓ encode OK", spectral_embeddings.shape)
            except Exception as e:
                print("  ⚠ encode failed:", e)

        # TRANSCRIBE
        print("- transcribe()")
        config_decode = WhiSPAConfig(stage='decode', audio_model_id=model_id, dtype=torch.bfloat16, device='cpu')
        model_decode = WhiSPAModel(config_decode).eval()
        with torch.no_grad():
            try:
                transcriptions = model_decode.transcribe(audio=audio_input, language="en", max_new_tokens=500, do_sample=False, num_beams=1)
                for transcript in transcriptions:
                    print(f"  ✓ transcribe OK: `{transcript}`")
            except Exception as e:
                print("  ⚠ transcribe failed:", e)

        # TRAIN_DEC
        print("- forward(train_dec)")
        config_train_dec = WhiSPAConfig(stage='train_dec', audio_model_id=model_id, dtype=torch.bfloat16, device='cpu')
        model_train_dec = WhiSPAModel(config_train_dec).eval()
        # Use model's processor to prepare inputs with spans
        proc_inputs = model_train_dec.processor(audio_input, language="en")
        spectral_inputs = proc_inputs["spectral_inputs"]
        sample_spans = proc_inputs["sample_spans"]
        input_ids = proc_inputs["text_input_ids"]
        attention_mask = proc_inputs["text_attention_mask"]
        labels = input_ids.clone()
        with torch.no_grad():
            try:
                outputs = model_train_dec(
                    spectral_inputs=spectral_inputs,
                    sample_spans=sample_spans,
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                    text_labels=labels,
                )
                print("  ✓ train_dec OK", outputs.loss.item(), outputs.logits.shape)
            except Exception as e:
                print("  ⚠ train_dec failed:", e)

        # TRAIN_ENC
        print("- forward(train_enc)")
        config_train_enc = WhiSPAConfig(stage='train_enc', audio_model_id=model_id, dtype=torch.bfloat16, device='cpu')
        model_train_enc = WhiSPAModel(config_train_enc).eval()
        # Reuse spectral_inputs and spans; make random target embeddings per sample
        B = sample_spans.shape[0]
        hidden_size = model_train_enc.voxtral_config.text_config.hidden_size
        target_audio = torch.randn(B, hidden_size)
        target_text = torch.randn(B, hidden_size)
        with torch.no_grad():
            try:
                out = model_train_enc(
                    spectral_inputs=spectral_inputs,
                    sample_spans=sample_spans,
                    target_audio_embs=target_audio,
                    target_text_embs=target_text,
                )
                print("  ✓ train_enc OK", out["total_loss"].item())
            except Exception as e:
                print("  ⚠ train_enc failed:", e)


if __name__ == "__main__":
    test_with_real_audio() 