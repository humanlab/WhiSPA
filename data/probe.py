import argparse
import gzip
import io
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def open_maybe_gzip(path: str) -> io.TextIOBase:
    if path == "-":
        return sys.stdin
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="r", encoding="utf-8", newline="")


def iter_jsonl(source: io.TextIOBase, errors: str = "skip") -> Iterator[Dict[str, Any]]:
    line_num = 0
    for raw_line in source:
        line_num += 1
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"JSON decode error on line {line_num}: {exc}"
            if errors == "raise":
                raise ValueError(msg) from exc
            logging.warning(msg)
            continue


def iter_json(path: str) -> Iterator[Dict[str, Any]]:
    with open_maybe_gzip(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON file: {path}: {exc}") from exc
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def iter_any(path: str, errors: str = "skip") -> Iterator[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        with open_maybe_gzip(path) as f:
            yield from iter_jsonl(f, errors=errors)
    elif lower.endswith(".json") or lower.endswith(".json.gz"):
        yield from iter_json(path)
    else:
        with open(path, mode="r", encoding="utf-8") as f:
            yield from iter_jsonl(f, errors=errors)


def iter_dataset(dataset_path: str, errors: str = "skip") -> Iterator[Dict[str, Any]]:
    if os.path.isdir(dataset_path):
        for root, _, files in os.walk(dataset_path):
            for name in files:
                if not (name.endswith(".json") or name.endswith(".jsonl") or name.endswith(".json.gz") or name.endswith(".jsonl.gz")):
                    continue
                fp = os.path.join(root, name)
                yield from iter_any(fp, errors=errors)
    else:
        yield from iter_any(dataset_path, errors=errors)


WORD_RE = re.compile(r"[A-Za-z0-9']+")


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        x = float(value)
        self.count += 1
        if self.min_value is None or x < self.min_value:
            self.min_value = x
        if self.max_value is None or x > self.max_value:
            self.max_value = x
        delta = x - self.mean
        self.mean += delta * (1.0 / self.count)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def to_summary(self) -> Dict[str, Any]:
        variance = (self.M2 / (self.count - 1)) if self.count > 1 else 0.0
        std = math.sqrt(variance) if variance > 0 else 0.0
        return {
            "count": self.count,
            "mean": self.mean if self.count else 0.0,
            "std": std,
            "min": self.min_value if self.min_value is not None else 0.0,
            "max": self.max_value if self.max_value is not None else 0.0,
        }


def tokenize_transcription(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in WORD_RE.findall(text)]


def extract_audio_metadata(sample: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Prefer top-level, then metadata.audio, then audio_metadata
    top = sample
    meta_audio = None
    if isinstance(sample.get("metadata"), dict):
        meta_audio = sample["metadata"].get("audio")
    audio_meta = sample.get("audio_metadata") if isinstance(sample.get("audio_metadata"), dict) else None

    def pick(keys: List[str]) -> Optional[Any]:
        for key in keys:
            if key in top and top.get(key) is not None:
                return top.get(key)
            if isinstance(meta_audio, dict) and key in meta_audio and meta_audio.get(key) is not None:
                return meta_audio.get(key)
            if isinstance(audio_meta, dict) and key in audio_meta and audio_meta.get(key) is not None:
                return audio_meta.get(key)
        return None

    duration_seconds = pick(["duration_seconds"])  # float
    num_channels = pick(["num_channels"])  # int
    sample_rate = pick(["sample_rate"])  # int
    bit_rate = pick(["bit_rate"])  # int
    bytes_size = pick(["bytes", "file_size_bytes"])  # int
    codec_name = pick(["codec_name"])  # str

    return {
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "num_channels": int(num_channels) if num_channels is not None else None,
        "sample_rate": int(sample_rate) if sample_rate is not None else None,
        "bit_rate": int(bit_rate) if bit_rate is not None else None,
        "bytes": int(bytes_size) if bytes_size is not None else None,
        "codec_name": str(codec_name) if codec_name is not None else None,
    }


class DatasetStatsAggregator:
    def __init__(self, check_audio_exists: bool = False) -> None:
        self.num_samples = 0
        self.total_duration_seconds = 0.0
        self.duration_stats = RunningStats()
        self.bit_rate_stats = RunningStats()
        self.bytes_stats = RunningStats()
        self.words_seen: set[str] = set()
        self.word_counts: Counter[str] = Counter()
        self.total_word_tokens = 0
        self.words_per_sample_stats = RunningStats()
        self.sample_rate_counts: Counter[int] = Counter()
        self.num_channels_counts: Counter[int] = Counter()
        self.codec_counts: Counter[str] = Counter()
        self.missing_transcription = 0
        self.missing_duration = 0
        self.check_audio_exists = check_audio_exists
        self.audio_paths_missing = 0
        self.audio_paths_present = 0

    def update(self, sample: Dict[str, Any]) -> None:
        self.num_samples += 1

        # Transcription and vocab
        transcription = sample.get("transcription")
        tokens = tokenize_transcription(transcription)
        if transcription is None:
            self.missing_transcription += 1
        self.total_word_tokens += len(tokens)
        self.words_per_sample_stats.add(float(len(tokens)))
        self.words_seen.update(tokens)
        self.word_counts.update(tokens)

        # Audio metadata
        audio_meta = extract_audio_metadata(sample)
        duration = audio_meta.get("duration_seconds")
        if duration is None:
            self.missing_duration += 1
        else:
            self.total_duration_seconds += float(duration)
        self.duration_stats.add(audio_meta.get("duration_seconds"))
        self.bit_rate_stats.add(audio_meta.get("bit_rate"))
        self.bytes_stats.add(audio_meta.get("bytes"))

        sr = audio_meta.get("sample_rate")
        if isinstance(sr, int):
            self.sample_rate_counts[sr] += 1
        nc = audio_meta.get("num_channels")
        if isinstance(nc, int):
            self.num_channels_counts[nc] += 1
        codec = audio_meta.get("codec_name")
        if isinstance(codec, str) and codec:
            self.codec_counts[codec] += 1

        # Audio path existence (optional)
        if self.check_audio_exists:
            audio_path = sample.get("audio") or sample.get("path")
            try:
                if audio_path and os.path.exists(audio_path):
                    self.audio_paths_present += 1
                else:
                    self.audio_paths_missing += 1
            except OSError:
                self.audio_paths_missing += 1

    def to_summary(self) -> Dict[str, Any]:
        hours = self.total_duration_seconds / 3600.0 if self.total_duration_seconds else 0.0
        duration_summary = self.duration_stats.to_summary()
        bit_rate_summary = self.bit_rate_stats.to_summary()
        bytes_summary = self.bytes_stats.to_summary()
        mean_words_per_sample = self.words_per_sample_stats.mean if self.words_per_sample_stats.count else 0.0
        top_words = self.word_counts.most_common(50)
        return {
            "samples": self.num_samples,
            "audio": {
                "sample_rate_counts": dict(self.sample_rate_counts),
                "num_channels_counts": dict(self.num_channels_counts),
                "codec_counts": dict(self.codec_counts),
                "bit_rate_summary": bit_rate_summary,
                "bytes_summary": bytes_summary,
            },
            "audio_path_check": (
                {
                    "present": self.audio_paths_present,
                    "missing": self.audio_paths_missing,
                }
                if self.check_audio_exists
                else None
            ),
            "duration": {
                "total_seconds": self.total_duration_seconds,
                "total_hours": hours,
                "mean_seconds": duration_summary.get("mean", 0.0),
                "std_seconds": duration_summary.get("std", 0.0),
                "min_seconds": duration_summary.get("min", 0.0),
                "max_seconds": duration_summary.get("max", 0.0),
                "missing_count": self.missing_duration,
            },
            "transcription": {
                "unique_words": len(self.words_seen),
                "total_word_tokens": self.total_word_tokens,
                "mean_words_per_sample": mean_words_per_sample,
                "top_words": [{"word": w, "count": c} for w, c in top_words],
                "missing_count": self.missing_transcription,
            }
        }


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process(
    dataset_path: str,
    errors: str = "skip",
    limit: Optional[int] = None,
    check_audio_exists: bool = False,
) -> Dict[str, Any]:
    agg = DatasetStatsAggregator(check_audio_exists=check_audio_exists)
    processed = 0
    for sample in tqdm(
        iter_dataset(dataset_path, errors=errors),
        desc="Processing samples",
        unit="sample",
    ):
        agg.update(sample)
        processed += 1
        if limit is not None and processed >= limit:
            break
    summary = agg.to_summary()
    summary["processed_samples"] = processed
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Summarize a dataset of JSON/JSONL samples (like people.json schema): "
            "computes total hours, mean duration, unique words, and audio metadata stats."
        )
    )
    p.add_argument("--dataset_path", required=True, help="Path to dataset file or directory (.json/.jsonl[.gz])")
    p.add_argument("--out_file", default=None, help="Optional path to write summary JSON")
    p.add_argument(
        "--errors",
        choices=["skip", "raise"],
        default="skip",
        help="On JSON parse error: skip the line (default) or raise",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional max samples to process")
    p.add_argument("--check-audio-exists", action="store_true", help="Check if audio files exist on disk")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging()
    try:
        summary = process(
            dataset_path=str(args.dataset_path),
            errors=str(args.errors),
            limit=args.limit if args.limit is not None else None,
            check_audio_exists=bool(args.check_audio_exists),
        )
        # Print to console
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        # Optional write to file
        if args.out_file:
            write_json(str(args.out_file), summary)
            logging.info("summary written to %s", args.out_file)
        return 0
    except KeyboardInterrupt:
        logging.warning("interrupted by user; partial output (printed above if any)")
        return 130


if __name__ == "__main__":
    sys.exit(main())


