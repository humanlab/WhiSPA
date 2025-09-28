import os
import json
import shutil
import subprocess
from typing import Dict

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import torchaudio
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_ffprobe(file_path: str) -> Dict:
    """Run ffprobe to get metadata as JSON. Returns empty dict on failure."""
    if not shutil.which("ffprobe"):
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=sample_rate,channels,codec_name",
        "-of",
        "json",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)
    except Exception:
        return {}


def worker_probe_only(wav_path: str, transcription: str, record_id: int) -> Dict:
    """Probe metadata with ffprobe and return final record dict for JSONL."""
    probed = run_ffprobe(wav_path)
    fmt = probed.get("format", {}) if isinstance(probed, dict) else {}
    streams = probed.get("streams", []) if isinstance(probed, dict) else []
    s0 = streams[0] if streams else {}

    # Parse fields
    duration_seconds = None
    if fmt.get("duration") is not None:
        try:
            duration_seconds = float(fmt["duration"])  # type: ignore[arg-type]
        except Exception:
            duration_seconds = None

    bit_rate = None
    if fmt.get("bit_rate") is not None:
        try:
            bit_rate = int(fmt["bit_rate"])  # type: ignore[arg-type]
        except Exception:
            bit_rate = None

    sr_val = s0.get("sample_rate")
    if isinstance(sr_val, str) and sr_val.isdigit():
        sr = int(sr_val)
    elif isinstance(sr_val, int):
        sr = int(sr_val)
    else:
        sr = None

    channels = int(s0.get("channels")) if isinstance(s0.get("channels"), int) else None
    codec_name = s0.get("codec_name") if isinstance(s0.get("codec_name"), str) else "pcm_s16le"

    file_size_bytes = int(os.path.getsize(wav_path))

    record = {
        "id": record_id,
        "audio": os.path.abspath(wav_path),
        "transcription": transcription,
        "audio_metadata": {
            "duration_seconds": duration_seconds,
            "sample_rate": sr,
            "num_channels": channels,
            "bytes": file_size_bytes,
            "codec_name": codec_name,
            "bit_rate": bit_rate,
        },
    }

    return record


def main() -> None:
    load_dotenv()

    ds = load_dataset('MLCommons/peoples_speech', 'clean')

    output_dir = os.getenv("PEOPLESPEECH_DIR")
    ensure_dir(output_dir)

    # Maintain a single globally incrementing ID across all splits
    global_id = 0

    # Deterministic split order
    splits_in_order = [s for s in ["train", "validation", "test"] if s in ds]

    # Global progress bar across all splits
    total_count = sum(len(ds[s]) for s in splits_in_order)
    pbar = tqdm(total=total_count, desc="Processing PeopleSpeech", unit="sample")

    # Ensure ffprobe is available before starting
    if not shutil.which("ffprobe"):
        raise EnvironmentError("ffprobe not found in PATH. Please install ffmpeg/ffprobe.")

    for split in splits_in_order:
        split_dir = os.path.join(output_dir, split)
        audio_dir = os.path.join(split_dir, "audio")
        ensure_dir(audio_dir)

        jsonl_path = os.path.join(split_dir, "dataset.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for sample in ds[split]:
                audio_field = sample["audio"]
                wav_filename = f"{global_id}.wav"
                wav_path = os.path.abspath(os.path.join(audio_dir, wav_filename))
                transcription = sample.get("text", "")

                # Decode to waveform and save via torchaudio
                ensure_dir(os.path.dirname(wav_path))
                waveform = None
                sr = None
                try:
                    audio_samples = audio_field.get_all_samples()
                    waveform = audio_samples.data
                    sr = int(audio_samples.sample_rate)
                except Exception:
                    if isinstance(audio_field, dict):
                        data = audio_field.get("array")
                        sr = int(audio_field.get("sampling_rate", 16000))
                        if data is not None:
                            wf = torch.as_tensor(data)
                            if wf.ndim == 1:
                                wf = wf.unsqueeze(0)
                            elif wf.ndim == 2 and wf.shape[0] > wf.shape[1]:
                                wf = wf.transpose(0, 1)
                            waveform = wf.float()

                if waveform is None or sr is None:
                    pbar.update(1)
                    global_id += 1
                    continue

                torchaudio.save_with_torchcodec(wav_path, waveform, sr)

                # Probe metadata synchronously
                rec = worker_probe_only(wav_path, transcription, global_id)
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                pbar.update(1)
                global_id += 1

    pbar.close()


if __name__ == "__main__":
    main()
