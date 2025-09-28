#!/usr/bin/env python3

import os
import json
from typing import List, Dict, Any, Optional

from torch.utils.data import Dataset
from data.utils import iter_jsonl
from dotenv import load_dotenv

load_dotenv()


class PeopleSpeechDataset(Dataset):
    """
    JSONL-backed dataset for PeopleSpeech, concatenating train/validation/test JSONLs
    into a single list, suitable for both train_enc and train_dec.

    Expected per-line schema (minimum):
      { "id": int, "audio": "/abs/path.wav", "transcription": "..." }
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        splits: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            base_dir: Base directory containing split subdirs with dataset_clean.jsonl.
                If None, uses os.getenv('PEOPLESPEECH_DIR').
            splits: List of splits to concatenate. Defaults to ["train", "validation", "test"].
            limit: Optional cap on total loaded samples for quick tests.
        """
        if base_dir is None:
            base_dir = os.getenv('PEOPLESPEECH_DIR')
        if base_dir is None:
            raise RuntimeError('PEOPLESPEECH_DIR environment variable is not set')

        if splits is None:
            splits = ["train", "validation", "test"]

        files: List[str] = []
        for split in splits:
            path = os.path.join(base_dir, split, 'dataset_clean.jsonl')
            if os.path.exists(path):
                files.append(path)

        if not files:
            raise FileNotFoundError(
                f"No dataset_clean.jsonl files found under {base_dir} for splits {splits}"
            )

        self.samples: List[Dict[str, Any]] = []
        next_id = 0
        for fp in files:
            for obj in iter_jsonl(fp):
                if 'audio' not in obj:
                    continue
                audio_path = obj['audio']
                if isinstance(audio_path, str):
                    obj['audio'] = os.path.abspath(audio_path)
                if not os.path.exists(obj['audio']):
                    continue
                if 'id' not in obj:
                    obj['id'] = next_id
                if 'transcription' not in obj:
                    obj['transcription'] = obj.get('text', '')

                self.samples.append(obj)
                next_id = max(next_id + 1, obj['id'] + 1)
                if limit is not None and len(self.samples) >= limit:
                    break
            if limit is not None and len(self.samples) >= limit:
                break

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under {base_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
