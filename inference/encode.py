import sys, os
# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Dict, List, Union
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from pretrain.whispa_model import WhiSPAModel

from dotenv import load_dotenv

load_dotenv()


# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a'}


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """
    Preprocess audio file for WhiSPA encoding.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Preprocessed audio waveform tensor
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert stereo (or multi-channel) to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=16000
        )(waveform)
    
    return waveform


def encode(
    whispa: WhiSPAModel,
    whisper: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_paths: Union[List[str], str]
) -> Dict[str, torch.Tensor]:
    """
    Encode audio files into embeddings using WhiSPA.
    
    Args:
        whispa: The WhiSPA model
        whisper: The Whisper model for tokenization
        processor: The Whisper processor
        audio_paths: Single audio path or list of audio paths
        
    Returns:
        Dictionary mapping audio paths to their embedding tensors
    """
    # Ensure audio_paths is a list
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    
    embeddings = {}
    device = next(whispa.parameters()).device
    
    for audio_path in audio_paths:
        try:
            # Preprocess audio
            waveform = preprocess_audio(audio_path)
            
            # Extract features
            input_features = processor(
                waveform.squeeze(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate tokens using Whisper
            with torch.no_grad():
                tokens = whisper.generate(input_features)
            
            # Get WhiSPA embedding
            with torch.no_grad():
                embedding = whispa(
                    audio_inputs=input_features,
                    text_input_ids=tokens,
                    text_attention_mask=torch.ones(tokens.size(), device=device),
                )
            
            embeddings[audio_path] = embedding.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            print(f'Error processing "{audio_path}": {e}')
            continue
    
    return embeddings


def load_models(model_id: str, device: str = 'cpu') -> tuple:
    """
    Load WhiSPA and Whisper models from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID for WhiSPA
        device: Device to load models on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (processor, whisper, whispa)
    """
    processor = WhisperProcessor.from_pretrained(
        'openai/whisper-tiny',
        device_map=device
    )
    
    whisper = WhisperForConditionalGeneration.from_pretrained(
        'openai/whisper-tiny'
    ).to(device)
    
    whispa = WhiSPAModel.from_pretrained(
        model_id
    ).to(device)
    
    return processor, whisper, whispa


def get_audio_files(path: str) -> List[str]:
    """
    Get all audio files from a path (file or directory).
    
    Args:
        path: File or directory path
        
    Returns:
        List of audio file paths
    """
    path = Path(path)
    
    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return [str(path)]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    elif path.is_dir():
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(path.glob(f'*{ext}'))
            audio_files.extend(path.glob(f'*{ext.upper()}'))
        return [str(f) for f in sorted(audio_files)]
    
    else:
        raise ValueError(f"Path does not exist: {path}")


def save_embeddings(embeddings: Dict[str, torch.Tensor], output_dir: str) -> None:
    """
    Save embeddings as individual .npy files.
    
    Args:
        embeddings: Dictionary mapping audio paths to embeddings
        output_dir: Directory to save embeddings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for audio_path, embedding in embeddings.items():
        # Get base filename without extension
        base_name = Path(audio_path).stem
        
        # Save as .npy file
        npy_path = output_path / f"{base_name}.npy"
        np.save(npy_path, embedding.detach().cpu().numpy())
        
        print(f"Saved: {npy_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate WhiSPA embeddings for audio files'
    )
    parser.add_argument(
        '--model_id',
        default='Jarhatz/WhiSPA-V1-Small',
        choices=[
            'Jarhatz/WhiSPA-V1-Small',
            'Jarhatz/WhiSPA-V1-Tiny',
        ],
        type=str,
        help='HuggingFace model ID for WhiSPA'
    )
    parser.add_argument(
        '--audio_path',
        required=True,
        type=str,
        help='Path to audio file or directory containing audio files'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        type=str,
        help='Directory to save embedding .npy files'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        type=str,
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading models...")
    processor, whisper, whispa = load_models(args.model_id, args.device)
    
    # Get audio files
    print(f"Finding audio files in: {args.audio_path}")
    audio_files = get_audio_files(args.audio_path)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Encode audio files
    print(f"Encoding audio files...")
    embeddings = encode(whispa, whisper, processor, audio_files)
    
    # Save embeddings
    print(f"Saving embeddings to: {args.output_path}")
    save_embeddings(embeddings, args.output_path)
    
    print(f"Successfully processed {len(embeddings)} files")


if __name__ == '__main__':
    main()