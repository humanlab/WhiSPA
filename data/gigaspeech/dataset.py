#!/usr/bin/env python3

import os
import json
from typing import List, Dict, Any, Optional

from torch.utils.data import Dataset
from data.utils import iter_jsonl
from dotenv import load_dotenv

load_dotenv()


class GigaSpeechDataset(Dataset):
    """
    JSONL-backed dataset for GigaSpeech suitable for both train_enc and train_dec.

    Expected schema per line (minimum):
      { "id": int, "audio": "/abs/path.wav", "transcription": "..." }

    Additional keys (optional) can hold teacher embedding paths (e.g., npy files)
    that can be consumed by the training script when running in train_enc stage.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            dataset_path: Path to dataset_clean.jsonl. If None, defaults to
                os.path.join(os.getenv('GIGASPEECH_DIR'), 'dataset_clean.jsonl') per instructions.
            limit: Optional maximum number of samples to load for quick tests.
        """
        # Per user instruction, use GIGASPEECH_DIR root + dataset_clean.jsonl for GigaSpeech
        if dataset_path is None:
            base = os.getenv('GIGASPEECH_DIR')
            if base is None:
                raise RuntimeError('GIGASPEECH_DIR environment variable is not set')
            dataset_path = os.path.join(base, 'dataset_clean.jsonl')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"JSONL not found: {dataset_path}")

        self.samples: List[Dict[str, Any]] = []
        for idx, obj in enumerate(iter_jsonl(dataset_path)):
            if 'audio' not in obj:
                continue
            # Ensure absolute path and that file exists
            audio_path = obj['audio']
            if isinstance(audio_path, str):
                obj['audio'] = os.path.abspath(audio_path)
            if not os.path.exists(obj['audio']):
                continue
            # Normalize keys
            if 'id' not in obj:
                obj['id'] = idx
            if 'transcription' not in obj:
                obj['transcription'] = obj.get('text', '')

            self.samples.append(obj)
            if limit is not None and len(self.samples) >= limit:
                break

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {dataset_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


