from typing import Optional, Dict, Any
import re
import json
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv

load_dotenv()


def mean_token_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool the embeddings over valid (non-masked) tokens.

    Args:
        embeddings: Token embeddings from the model (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask indicating valid tokens (batch_size, seq_len)

    Returns:
        Mean pooled embeddings (batch_size, hidden_dim)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last token from sequences, handling both left and right padding.
    
    Args:
        last_hidden_states: Hidden states from the model
        attention_mask: Attention mask indicating valid tokens
    
    Returns:
        Pooled embeddings using the last valid token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


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