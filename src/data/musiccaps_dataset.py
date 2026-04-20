from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    example_id: str
    mel_path: str
    index: int


class MusicCapsMelDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        labels_npy: Path,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        target_frames: int = 320,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.metadata = pd.read_csv(metadata_csv)
        self.labels = np.load(labels_npy).astype(np.float32)
        self.target_frames = target_frames
        if len(self.metadata) != len(self.labels):
            raise ValueError("metadata and labels length mismatch")

        indices = np.arange(len(self.metadata))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        val_size = max(1, int(len(indices) * val_ratio))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        chosen = train_idx if split == "train" else val_idx

        self.records = [
            SampleRecord(
                example_id=str(self.metadata.iloc[i]["example_id"]),
                mel_path=str(self.metadata.iloc[i]["mel_path"]),
                index=int(i),
            )
            for i in chosen
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[item]
        mel = np.load(record.mel_path).astype(np.float32)
        mel = self._fit_to_frames(mel, self.target_frames)

        # Normalize each example to stabilize training.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, n_mels, time]
        label_tensor = torch.from_numpy(self.labels[record.index])
        return mel_tensor, label_tensor, record.example_id

    @staticmethod
    def _fit_to_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
        cur_frames = mel.shape[1]
        if cur_frames == target_frames:
            return mel
        if cur_frames > target_frames:
            return mel[:, :target_frames]

        pad_width = target_frames - cur_frames
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel.min())
