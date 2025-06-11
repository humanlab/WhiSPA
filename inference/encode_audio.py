import sys, os
# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torchaudio
import huggingface_hub
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from pretrain.whispa_model import UniSpeechModel

from dotenv import load_dotenv

load_dotenv()


def load_args():
    parser = argparse.ArgumentParser(description='Script to inference WhiSPA model (Generates Embeddings)')
    parser.add_argument(
        '--model_id',
        default='Jarhatz/whispa_394_v1',
        choices=[
            'Jarhatz/whispa_394_v1'
        ],
        type=str,
        help='Specify the filename to the model directory. It will use `config.pth` and `best.pth` saved in: <CHECKPOINT_DIR>/<MODEL_NAME>/`\nOr specify the HuggingFace model id for a SBERT autoencoder from the sentence-transformers/ library. `Ex. sentence-transformers/all-MiniLM-L12-v2`'
    )
    parser.add_argument(
        '--hf_token',
        required=True,
        type=str,
        help='Specify your HuggingFace access token for loading and using the pretrained model from transformers.'
    )
    parser.add_argument(
        '--audio_path',
        required=True,
        type=str,
        help='Path to specific audio file or directory containing audio files'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        type=str,
        help='Path to save the embeddings'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=[
            'cpu',
            'cuda'
        ],
        type=str,
        help='Specify whether to use CPU or GPU'
    )
    return parser.parse_args()


def load_model(model_id, device):
    processor = WhisperProcessor.from_pretrained(
        'openai/whisper-tiny',
        device_map=device
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(
        'openai/whisper-tiny'
    ).to(device)
    whispa = UniSpeechModel.from_pretrained(
        model_id
    ).to(device)
    return processor, whisper, whispa


def encode_audios(
    audio_path,
    model_id,
    device='cpu'
):
    processor, whisper, whispa = load_model(model_id, device)
    
    embs = []
    filenames = []

    if os.path.isdir(audio_path):
        for filename in os.listdir(audio_path):
            try:
                embs.append(get_embedding(
                    os.path.join(audio_path, filename),
                    processor,
                    whisper,
                    whispa,
                    device
                ))
                filenames.append(filename)
            except Exception as e:
                print(f'\"{filename}\" failed with the following error:')
                print(Warning(e))
    else:
        embs.append(get_embedding(
            audio_path,
            processor,
            whisper,
            whispa,
            device
        ))
        filenames.append(os.path.basename(audio_path))
    
    return torch.cat(embs), filenames


def get_embedding(audio_path, processor, whisper, whispa, device):
    # Audio processing
    waveform = preprocess_audio(audio_path)
    input_features = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    # Whisper-based tokenization
    tokens = whisper.generate(input_features)

    # WhiSPA embedding
    emb = whispa(
        audio_inputs=input_features,
        text_input_ids=tokens,
        text_attention_mask=torch.ones(tokens.size(), device=device),
    )
    return emb


def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert stereo (or multi-channel) to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform


if __name__ == '__main__':
    args = load_args()
    huggingface_hub.login(args.hf_token)

    # Get embeddings
    embeddings, filenames = encode_audios(args.audio_path, args.model_id, args.device)

    # Save embeddings
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, 'embeddings.npz')
    np.savez(output_path, embeddings=embeddings.detach().cpu().numpy(), filenames=filenames)

    print(f'Embeddings saved to {output_path}')