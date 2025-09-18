from typing import Optional, Dict, Any
import re
import json
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv

load_dotenv()


def fenced_json_str_to_json(text: str) -> Optional[str]:
    # Prefer ```json ...``` then any ``` ... ```
    pattern = re.compile(r"```(\w+)?\s*[\r\n]+(.*?)[\r\n]+```", re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    # Prefer label json
    for m in matches:
        lang = (m.group(1) or "").strip().lower()
        if lang == "json":
            return m.group(2).strip()
    return matches[0].group(2).strip()


def json_to_fenced_json_str(obj: Dict[str, Any]) -> str:
    json_str = json.dumps(obj, indent=2, ensure_ascii=False)
    return f"```json\n{json_str}\n```"


def preprocess_audio(audio_path, target_sampling_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo (or multi-channel) to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != target_sampling_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sampling_rate)(waveform)
    return waveform


def collate_train(batch):
    audio_inputs = None
    try:
        # Batch padding
        audio_inputs = torch.cat([w for w, _, _, _, _, _ in batch], dim=0)
    except Exception:
        audio_inputs = pad_sequence(
            [w.squeeze(0) for w, _, _, _, _, _ in batch],
            batch_first=True,
            padding_value=0.0
        )
    
    acoustic_inputs = None
    # Check if acoustic inputs are preprocessed
    if isinstance(batch[0][2], torch.Tensor):
        # Truncate preprocessed audio to max length
        MAX_LENGTH = 400000
        acoustic_inputs = [a[:, :MAX_LENGTH] for _, _, a, _, _, _ in batch]
        # Batch padding
        try:
            acoustic_inputs = torch.cat(acoustic_inputs, dim=0)
        except Exception:
            acoustic_inputs = pad_sequence(
                [a.squeeze(0) for a in acoustic_inputs],
                batch_first=True,
                padding_value=0.0
            )

    return {
        'audio_inputs': audio_inputs,
        'message': [m for _, m, _, _, _, _  in batch],
        'acoustic_inputs': acoustic_inputs,
        'linguistic_embs': torch.cat([l for _, _, _, l, _, _ in batch], dim=0) if isinstance(batch[0][-3], torch.Tensor) else None,
        'acoustic_embs': torch.cat([a for _, _, _, _, a, _ in batch], dim=0) if isinstance(batch[0][-2], torch.Tensor) else None,
        'psych_embs': torch.cat([o for _, _, _, _, _, o in batch], dim=0) if isinstance(batch[0][-1], torch.Tensor) else None
    }


def collate_inference(batch):
    try:
        audio_inputs = torch.cat([a for _, _, a, _ in batch], dim=0) if isinstance(batch[0][2], torch.Tensor) else None
    except Exception:
        audio_inputs = pad_sequence(
            [a.squeeze(0) for _, _, a, _ in batch],
            batch_first=True,
            padding_value=0.0
        )
    return {
        'dataset_name': [d for d, _, _, _ in batch],
        'message_id': [i for _, i, _, _ in batch],
        'audio_inputs': audio_inputs,
        'message': [m for _, _, _, m  in batch]
    }