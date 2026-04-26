#!/usr/bin/env python3
"""
Prepare Google MusicCaps for the EchoVision music->label model.

Pipeline:
1) Load MusicCaps metadata from Hugging Face.
2) Download and trim each YouTube clip to the labeled window.
3) Convert audio to log-Mel spectrogram features.
4) Build a multi-label target space from the aspect list field.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from yt_dlp import YoutubeDL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MusicCaps data for CNN training.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/musiccaps"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/musiccaps"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--min-label-frequency", type=int, default=2)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Optional yt-dlp browser cookie source (e.g. chrome, safari, firefox).",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Optional Netscape cookies.txt path (preferred over browser extraction).",
    )
    return parser.parse_args()


def safe_key(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def parse_aspects(aspect_value: Any) -> list[str]:
    if aspect_value is None:
        return []
    if isinstance(aspect_value, list):
        raw_items = aspect_value
    else:
        s = str(aspect_value).strip()
        raw_items: list[Any]
        if s.startswith("[") and "]" in s:
            parsed: Any = None
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError, TypeError):
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = re.split(r",|;|\|", s)
        else:
            raw_items = re.split(r",|;|\|", s)

    cleaned: list[str] = []
    for item in raw_items:
        text = re.sub(r"\s+", " ", str(item).strip().lower())
        while len(text) >= 2 and text[0] == text[-1] == "'":
            text = text[1:-1].strip().lower()
        while len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1].strip().lower()
        text = re.sub(r"^[\[\('\"]+", "", text)
        text = re.sub(r"[\]\)'\"]+$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def clean_error_text(message: str) -> str:
    # Strip ANSI escape sequences from third-party CLI errors.
    return re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", message)


def youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def download_audio(
    youtube_id: str,
    cache_dir: Path,
    retries: int = 3,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    overwrite: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{youtube_id}.wav"
    if output_path.exists() and not overwrite:
        return output_path

    # Some YouTube videos do not expose the same audio format set. We try a few
    # selectors from strict -> permissive to reduce "Requested format is not available".
    format_candidates = [
        "bestaudio*",
        "bestaudio/best",
        "best",
    ]
    cookie_modes: list[dict[str, Any]] = []
    if cookies_file is not None:
        cookie_modes.append({"cookiefile": str(cookies_file)})
    elif cookies_from_browser:
        # Keep this shape simple and robust across yt-dlp versions.
        cookie_modes.append({"cookiesfrombrowser": (cookies_from_browser,)})
    # Always include a no-cookie fallback so browser cookie failures do not abort long jobs.
    cookie_modes.append({})

    last_exc: Exception | None = None
    for cookie_mode in cookie_modes:
        for fmt in format_candidates:
            ydl_opts = {
                "format": fmt,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "retries": retries,
                "fragment_retries": retries,
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"
                    )
                },
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "web"],
                    }
                },
                "outtmpl": str(cache_dir / f"{youtube_id}.%(ext)s"),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }
                ],
            }
            ydl_opts.update(cookie_mode)

            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url(youtube_id)])
                last_exc = None
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if last_exc is None:
            break

    if last_exc is not None:
        raise last_exc

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create audio file for {youtube_id}")
    return output_path


def trim_and_resample(
    wav_path: Path,
    start_s: float,
    duration_s: float,
    sample_rate: int,
) -> np.ndarray:
    audio, _ = librosa.load(
        str(wav_path),
        sr=sample_rate,
        mono=True,
        offset=max(0.0, start_s),
        duration=max(0.1, duration_s),
    )
    if audio.size == 0:
        raise ValueError("Loaded empty audio after trim.")
    return audio.astype(np.float32)


def audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def build_vocab(items: list[list[str]], min_freq: int) -> list[str]:
    counts: dict[str, int] = {}
    for labels in items:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    vocab = [label for label, count in counts.items() if count >= min_freq]
    vocab.sort()
    return vocab


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mel").mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("google/MusicCaps", split=args.split)
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.download_all:
        args.max_items = None

    start = min(args.start_index, len(dataset))
    if args.max_items is not None:
        stop = min(start + args.max_items, len(dataset))
    else:
        stop = len(dataset)
    dataset = dataset.select(range(start, stop))

    rows: list[dict[str, Any]] = []
    all_label_lists: list[list[str]] = []
    failed: list[dict[str, str]] = []

    for idx, row in enumerate(tqdm(dataset, desc="Preparing MusicCaps")):
        youtube_id = safe_key(row, ["ytid", "yt_id", "youtube_id"])
        if not youtube_id:
            failed.append({"row_index": str(idx), "reason": "missing youtube id"})
            continue

        start_s = float(safe_key(row, ["start_s", "start", "start_sec"], 0.0))
        end_s = safe_key(row, ["end_s", "end", "end_sec"], None)
        duration_s = float(end_s - start_s) if end_s is not None else 10.0
        duration_s = 10.0 if duration_s <= 0 else duration_s

        caption = str(safe_key(row, ["caption"], "")).strip()
        label_list = parse_aspects(safe_key(row, ["aspect_list", "aspects", "tags"], []))
        all_label_lists.append(label_list)

        example_id = f"{idx:05d}_{youtube_id}"
        audio_file = args.output_dir / "audio" / f"{example_id}.wav"
        mel_file = args.output_dir / "mel" / f"{example_id}.npy"

        try:
            if not args.metadata_only:
                if (audio_file.exists() and mel_file.exists() and not args.overwrite):
                    pass
                else:
                    full_audio = download_audio(
                        youtube_id,
                        args.cache_dir,
                        retries=args.download_retries,
                        cookies_from_browser=args.cookies_from_browser,
                        cookies_file=args.cookies_file,
                        overwrite=args.overwrite,
                    )
                    audio = trim_and_resample(full_audio, start_s, duration_s, args.sample_rate)
                    sf.write(str(audio_file), audio, args.sample_rate)
                    mel = audio_to_logmel(
                        audio,
                        sample_rate=args.sample_rate,
                        n_mels=args.n_mels,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                    )
                    np.save(mel_file, mel)
        except Exception as exc:  # noqa: BLE001
            reason = clean_error_text(f"{type(exc).__name__}: {exc}")
            failed.append({"row_index": str(idx), "reason": reason})
            continue

        rows.append(
            {
                "example_id": example_id,
                "youtube_id": youtube_id,
                "start_s": start_s,
                "duration_s": duration_s,
                "caption": caption,
                "labels": json.dumps(label_list, ensure_ascii=True),
                "audio_path": str(audio_file),
                "mel_path": str(mel_file),
            }
        )

    vocab = build_vocab(all_label_lists, min_freq=args.min_label_frequency)
    vocab_index = {label: i for i, label in enumerate(vocab)}

    labels_matrix = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, row in enumerate(rows):
        labels = json.loads(row["labels"])
        for label in labels:
            j = vocab_index.get(label)
            if j is not None:
                labels_matrix[i, j] = 1.0

    metadata_path = args.output_dir / "metadata.csv"
    vocab_path = args.output_dir / "label_vocab.json"
    labels_path = args.output_dir / "labels_multihot.npy"
    failed_path = args.output_dir / "failed_downloads.json"
    config_path = args.output_dir / "preprocess_config.json"

    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    np.save(labels_path, labels_matrix)
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "dataset": "google/MusicCaps",
                "split": args.split,
                "sample_rate": args.sample_rate,
                "n_mels": args.n_mels,
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
                "min_label_frequency": args.min_label_frequency,
                "metadata_only": args.metadata_only,
                "processed_items": len(rows),
                "failed_items": len(failed),
                "vocab_size": len(vocab),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved labels matrix: {labels_path} | shape={labels_matrix.shape}")
    print(f"Saved label vocab: {vocab_path} | size={len(vocab)}")
    print(f"Failed items: {len(failed)} -> {failed_path}")
    if len(rows) == 0 and not args.metadata_only:
        print(
            "No examples were processed. Check failed_downloads.json for details. "
            "If YouTube blocks downloads, try --metadata-only to inspect labels first.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
