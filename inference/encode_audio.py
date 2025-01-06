import os
import argparse
import torch
import torchaudio
import huggingface_hub
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from dotenv import load_dotenv

load_dotenv()

from pretrain.whispa_model import WhiSPAModel


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
        '--audio_filepaths',
        required=True,
        type=str,
        help='Path to specific audio file or directory containing audio files'
    )
    parser.add_argument(
        '--cache_dir',
        required=False,
        type=str,
        help='Path to specific cache directory for storing model weights'
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


def load_model(model_id, cache_dir, device):
    processor = WhisperProcessor.from_pretrained(
        'openai/whisper-tiny',
        cache_dir=cache_dir,
        device_map=device
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(
        'openai/whisper-tiny',
        cache_dir=cache_dir,
    ).to(device)
    whispa = WhiSPAModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    ).to(device)
    return processor, whisper, whispa


def encode_audios(
    audio_filepaths,
    model_id,
    cache_dir=None,
    device='cpu'
):
    processor, whisper, whispa = load_model(model_id, cache_dir, device)
    
    embs = []

    if os.path.isdir(audio_filepaths):
        for filename in os.listdir(audio_filepaths):
            try:
                embs.append(get_embedding(
                    os.path.join(audio_filepaths, filename),
                    processor,
                    whisper,
                    whispa,
                    device
                ))
            except Exception as e:
                print(f'\"{filename}\" failed with the following error:')
                print(Warning(e))
    else:
        embs.append(get_embedding(
            audio_filepaths,
            processor,
            whisper,
            whispa,
            device
        ))
    
    return torch.cat(embs)


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

    embs = encode_audios(args.audio_filepaths, args.model_id, args.cache_dir, args.device)

    print(embs.shape)