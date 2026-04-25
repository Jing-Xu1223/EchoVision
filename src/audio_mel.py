"""Mel-spectrogram helpers aligned with ``prepare_musiccaps`` and training."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


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


def fit_mel_to_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
    cur_frames = mel.shape[1]
    if cur_frames == target_frames:
        return mel
    if cur_frames > target_frames:
        return mel[:, :target_frames]
    pad_width = target_frames - cur_frames
    return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel.min())


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    return ((mel - mel.mean()) / (mel.std() + 1e-6)).astype(np.float32)


def load_audio_segment(
    path: Path,
    *,
    sample_rate: int,
    start_s: float = 0.0,
    duration_s: float | None = 10.0,
) -> np.ndarray:
    """Load mono audio resampled to ``sample_rate``; optionally cap length."""
    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=True,
        offset=max(0.0, start_s),
        duration=duration_s,
    )
    if audio.size == 0:
        raise ValueError("Loaded empty audio.")
    return audio.astype(np.float32)
