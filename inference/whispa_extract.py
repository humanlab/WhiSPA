import sys, os
# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torchaudio
from transformers import WhisperProcessor
from pretrain.whispa_model import WhiSPAModel


DEBUG = True
TEMP_DIR = "/cronus_data/rrao/temp/"

"""
EXTRACTIONS:

CUDA_VISIBLE_DEVICES=1 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot 1" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot 1" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot 1"

CUDA_VISIBLE_DEVICES=1 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot 2 - batch 1" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot 2 - batch 1" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot 2 - batch 1"

CUDA_VISIBLE_DEVICES=1 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot 2 - batch 2" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot 2 - batch 2" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot 2 - batch 2"

CUDA_VISIBLE_DEVICES=1 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot 2 - batch 2 extended" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot 2 - batch 2 extended" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot 2 - batch 2 extended"

CUDA_VISIBLE_DEVICES=3 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot 2 - batch 3" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot 2 - batch 3" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot 2 - batch 3"

CUDA_VISIBLE_DEVICES=3 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot Study 2.0 - Batch 4" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot Study 2.0 - Batch 4" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot Study 2.0 - Batch 4"

CUDA_VISIBLE_DEVICES=3 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot Study 2.0 - Batch 5" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot Study 2.0 - Batch 5" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot Study 2.0 - Batch 5"

CUDA_VISIBLE_DEVICES=3 python whispa_extract.py \
--load_path "/cronus_data/rrao/WhiSPA/checkpoints_v1/whispa_1034_5_32_1e-5_1e-2" \
--audio_dir "/cronus_data/PTSD_STOP/data/Daily Study Audio Files/Pilot Study 2.0 - Batch 6" \
--transcript_dir "/cronus_data/PTSD_STOP/data/Daily Study Whisper Transcripts New/Pilot Study 2.0 - Batch 6" \
--output_dir "/cronus_data/PTSD_STOP/data/Daily Study WhiSPA_Medium Embeddings B1-6/Pilot Study 2.0 - Batch 6"
"""

def main():
    parser = argparse.ArgumentParser(description='Whisper Embedding Extraction Script')

    parser.add_argument("--load_path", type=str, required=True, help="Specify the model ID for Whisper")
    parser.add_argument("--audio_dir", type=str, required=True, help="Specify the path to the audio files")
    parser.add_argument("--transcript_dir", type=str, required=True, help="Specify the path to the transcript files")
    parser.add_argument("--output_dir", type=str, required=True, help="Specify the path to save the embeddings (.csv)")
    parser.add_argument("--device", type=str, default="cuda", help="Specify the device to use (default: cuda)")
    args = parser.parse_args()

    if os.path.exists(args.audio_dir) and os.path.exists(args.transcript_dir):
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        driver(args.load_path, args.audio_dir, args.transcript_dir, args.output_dir, args.device)
    else:
        print('Could not find `audio_dir` and/or `transcript_dir`.\nPlease double-check your input directories.')


def driver(load_path, audio_dir, transcript_dir, output_dir, device):
    audio_filenames = os.listdir(audio_dir)

    # Load the Whisper model and processor
    config = torch.load(os.path.join(load_path, 'config.pth'), weights_only=False)
    config.device = device
    print(config)
    whispa = WhiSPAModel(config).to(device)
    
    state_dict = torch.load(os.path.join(load_path, 'best.pth'))
    try:
        whispa.load_state_dict(state_dict)
    except:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        whispa.load_state_dict(state_dict)
    
    processor = WhisperProcessor.from_pretrained(whispa.config.whisper_model_id, device_map=device)

    cols = ['segment_id', 'filename'] + [f'f{i:04d}_{stat}' for stat in ['mea', 'med', 'var', 'min', 'max'] for i in range(1034)]

    for f_idx, audio_filename in enumerate(audio_filenames):
        log(f'[{f_idx + 1}/{len(audio_filenames)}] {audio_filename}')

        segments = get_segments(audio_filename, transcript_dir)

        if segments:
            # Create a new .csv file for each audio file
            output_df = pd.DataFrame(columns=cols)
            output_df.to_csv(f'{output_dir}/{audio_filename[:-4]}.csv', index=False)

            split_audio(audio_filename, audio_dir, segments)

            # Extract Whisper Embeddings per Segment
            for s_idx in range(len(segments)):
                audio_clip_name = f'{audio_filename[:-4]} - {s_idx:04d}.wav'
                audio_clip = os.path.join(TEMP_DIR, audio_clip_name)
                log(f'\t[{s_idx + 1}/{len(segments)}] {audio_clip_name}')

                try:
                    # Audio processing
                    waveform = preprocess_audio(audio_clip)
                    input_features = processor(
                        waveform.squeeze(),
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.to(device)

                    # Text preprocessing
                    text_tokens = torch.tensor(segments[s_idx][2], device=device).unsqueeze(0)
                    text_attention_mask = torch.ones_like(text_tokens, device=device)

                    # WhiSPA embedding
                    with torch.no_grad():
                        embeddings = whispa(
                            audio_inputs=input_features,
                            text_input_ids=text_tokens,
                            text_attention_mask=text_attention_mask,
                        )

                    emb_mea = torch.mean(embeddings, dim=0)
                    emb_med, _ = torch.median(embeddings, dim=0)
                    emb_var = torch.var(embeddings, dim=0)
                    emb_min, _ = torch.min(embeddings, dim=0)
                    emb_max, _ = torch.max(embeddings, dim=0)
                    emb_cat = torch.cat([emb_mea, emb_med, emb_var, emb_min, emb_max]).cpu().numpy().tolist()

                    emb_df = pd.DataFrame([[audio_clip_name[:-4], audio_filename] + emb_cat], columns=cols)
                    emb_df.to_csv(f'{output_dir}/{audio_filename[:-4]}.csv', mode='a', header=False, index=False)
                except Exception as e:
                    log(f'\tFailed. Error: [{e}]')
                    failure_path = f'{os.path.dirname(output_dir)}/{os.path.basename(output_dir)} Failures.txt'
                    with open(failure_path, 'a+') as f:
                        f.write(f'[{f_idx} - {s_idx}] {audio_clip_name}\n')

                # Delete the segment's temporary audio file
                if os.path.exists(audio_clip):
                    os.remove(audio_clip)


def get_segments(audio_filename, transcript_dir):
    transcript_path = os.path.join(transcript_dir, f'{audio_filename[:-4]}.json')
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r') as f:
            data = json.load(f)
            return [(segment['start'], segment['end'], segment['tokens']) for segment in data['segments']]
    else:
        segments = []
        transcript_path = os.path.join(transcript_dir, f'{audio_filename[:-4]}-left.json')
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                data = json.load(f)
                segments.extend([(segment['start'], segment['end'], segment['tokens']) for segment in data['segments']])
        transcript_path = os.path.join(transcript_dir, f'{audio_filename[:-4]}-right.json')
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                data = json.load(f)
                segments.extend([(segment['start'], segment['end'], segment['tokens']) for segment in data['segments']])
        return segments


def split_audio(audio_filename, audio_dir, segments):
    audio_data, sample_rate = librosa.load(os.path.join(audio_dir, audio_filename))

    for s_idx, segment in enumerate(segments):
        start_index = int(segment[0] * sample_rate)
        end_index = int(segment[1] * sample_rate)
        segment_audio = audio_data[start_index : end_index]

        # Temporarily save the audio clip as a .wav file 
        # because OW only reads .wav format for vocal acoustics
        output_path = os.path.join(TEMP_DIR, f'{audio_filename[:-4]} - {s_idx:04d}.wav')
        sf.write(output_path, segment_audio, sample_rate)


def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:  # More than one channel (e.g., stereo)
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    return waveform


def log(msg):
    if DEBUG:
        print(msg)


if __name__ == '__main__':
    main()