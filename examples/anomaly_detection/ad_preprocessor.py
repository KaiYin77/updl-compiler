#!/usr/bin/env python3

import glob
import os
from typing import Any, Iterable, List, Sequence

import librosa
import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from updl_compiler.core.preprocessors.base import DataPreprocessor
from updl_compiler.core.logger import log_error, log_info


class AnomalyDetectionPreprocessor(DataPreprocessor):
    """Convert ToyCar audio clips into log-mel frame stacks used by the autoencoder."""

    def __init__(
        self,
        n_mels: int = 128,
        frames: int = 5,
        n_fft: int = 1024,
        hop_length: int = 512,
        power: float = 2.0,
        sample_rate: int | None = None,
        center_start: int = 50,
        center_end: int = 250,
    ) -> None:
        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.sample_rate = sample_rate
        self.center_start = center_start
        self.center_end = center_end
        self.feature_dim = self.n_mels * self.frames

    def _load_log_mel(self, file_path: str) -> np.ndarray:
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )
        log_mel = 20.0 / self.power * np.log10(mel + np.finfo(np.float32).eps)
        return log_mel[:, self.center_start:self.center_end]

    def extract_feature_windows(self, file_path: str) -> np.ndarray:
        log_mel = self._load_log_mel(file_path)
        total_frames = log_mel.shape[1] - self.frames + 1
        if total_frames <= 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        windows = np.zeros((total_frames, self.feature_dim), dtype=np.float32)
        for t in range(self.frames):
            start = self.n_mels * t
            stop = start + self.n_mels
            windows[:, start:stop] = log_mel[:, t : t + total_frames].T
        return windows

    def count_windows_in_file(self, file_path: str) -> int:
        log_mel = self._load_log_mel(file_path)
        return max(0, log_mel.shape[1] - self.frames + 1)

    def list_audio_files(self, dataset_dir: str) -> List[str]:
        pattern = os.path.join(dataset_dir, "**", "*.wav")
        files = sorted(glob.glob(pattern, recursive=True))
        return files

    def iter_windows(self, dataset_dir: str) -> Iterable[np.ndarray]:
        for file_path in self.list_audio_files(dataset_dir):
            windows = self.extract_feature_windows(file_path)
            for window in windows:
                yield window

    def preprocess_sample(self, sample: Any) -> tf.Tensor:
        if isinstance(sample, np.ndarray):
            vector = sample.astype(np.float32, copy=False)
        elif isinstance(sample, str):
            windows = self.extract_feature_windows(sample)
            if windows.size == 0:
                raise ValueError(f"No frames extracted from {sample}")
            vector = windows[0]
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")

        tensor = tf.convert_to_tensor(vector, dtype=tf.float32)
        if tensor.shape.rank == 1:
            tensor = tf.expand_dims(tensor, axis=0)
        return tensor

    def load_calibration_data(self, data_source: Any, count: int | None = None) -> List[Any]:
        if isinstance(data_source, list):
            return data_source if count is None else data_source[:count]

        if isinstance(data_source, str):
            dataset_dir = data_source
            indices_file = None
        elif isinstance(data_source, dict):
            dataset_dir = data_source.get("dataset_dir")
            indices_file = data_source.get("indices_file")
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        if not dataset_dir or not os.path.exists(dataset_dir):
            log_error(f"Calibration dataset directory not found: {dataset_dir}")
            return []

        target_indices: Sequence[int] | None = None
        if indices_file and os.path.exists(indices_file):
            with open(indices_file, "r", encoding="utf-8") as f:
                target_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
            target_indices = sorted(set(target_indices))
            log_info(f"Loaded {len(target_indices)} calibration indices")

        samples: List[np.ndarray] = []
        default_target = count if count is not None else 120
        limit = default_target if not target_indices else len(target_indices)
        if limit == 0:
            log_error("Calibration request asked for zero samples.")
            return []

        if target_indices:
            needed = len(target_indices)
            next_targets = iter(target_indices)
            current_target = next(next_targets, None)
            for global_idx, window in enumerate(self.iter_windows(dataset_dir)):
                if current_target is None:
                    break
                if global_idx == current_target:
                    samples.append(window)
                    current_target = next(next_targets, None)
                    if len(samples) >= needed:
                        break
        else:
            for window in self.iter_windows(dataset_dir):
                samples.append(window)
                if len(samples) >= limit:
                    break

        log_info(f"Prepared {len(samples)} calibration samples")
        return samples
