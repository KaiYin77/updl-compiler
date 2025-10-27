#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
SWW (Streaming Wake Word) preprocessor for audio data
"""

import os
import sys
from typing import Any, List, Union

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sww_dataset import preprocess_audio as dataset_preprocess_audio
from sww_dataset import extract_log_mel_features as dataset_extract_log_mel_features
from sww_dataset import get_file_lists
from updl_compiler.core.logger import log_error, log_info
from updl_compiler.core.preprocessors.base import DataPreprocessor


# Labels used in SWW models
SWW_LABELS = ["marvin", "silence", "unknown"]


class SWWPreprocessor(DataPreprocessor):
    """Preprocessor for SWW models using speech commands dataset"""

    def __init__(
        self,
        sample_rate: int = 16000,
        clip_duration_ms: int = 1000,
        window_size_ms: float = 64.0,  # benchmark default
        window_stride_ms: float = 32.0,  # benchmark default
        dct_coefficient_count: int = 40,  # benchmark default
    ):
        """
        Initialize SWW preprocessor with audio processing parameters

        Args:
            sample_rate: Audio sample rate in Hz
            clip_duration_ms: Length of audio clips in milliseconds
            window_size_ms: STFT window size in milliseconds
            window_stride_ms: STFT window stride in milliseconds
            dct_coefficient_count: Number of Mel-frequency bins
        """
        self.sample_rate = sample_rate
        self.clip_duration_ms = clip_duration_ms
        self.window_size_ms = window_size_ms
        self.window_stride_ms = window_stride_ms
        self.dct_coefficient_count = dct_coefficient_count

        # Derived parameters
        self.desired_samples = int(sample_rate * clip_duration_ms / 1000)

    def decode_audio(self, audio_binary: tf.Tensor) -> tf.Tensor:
        """Decode WAV-encoded audio to float32 tensor"""
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        return tf.squeeze(audio, axis=-1)

    def _build_audio_dict(self, audio_waveform: tf.Tensor) -> dict:
        """Wrap waveform in dataset-style dict for shared preprocessing"""
        return {
            "audio": tf.cast(audio_waveform, tf.float32),
            # Placeholder label; retained to satisfy dataset helper expectations
            "label": tf.constant("unknown"),
        }

    def preprocess_sample(self, sample: Union[dict, str, tf.Tensor]) -> tf.Tensor:
        """
        Convert audio sample to preprocessed tensor for SWW model

        Args:
            sample: Can be:
                - tf.Tensor representing audio waveform
                - Dict containing 'audio' key
                - String path to audio file

        Returns:
            tf.Tensor: Preprocessed audio features ready for model inference
        """
        if isinstance(sample, dict):
            if "audio" not in sample:
                raise ValueError("Dict sample must contain 'audio' key")
            audio = sample["audio"]
        elif isinstance(sample, str):
            audio_binary = tf.io.read_file(sample)
            audio = self.decode_audio(audio_binary)
        else:
            audio = sample

        audio_dict = self._build_audio_dict(audio)
        audio_dict = dataset_preprocess_audio(audio_dict, target_length=self.desired_samples)
        audio_dict = dataset_extract_log_mel_features(
            audio_dict,
            sample_rate=self.sample_rate,
            window_size_ms=self.window_size_ms,
            window_stride_ms=self.window_stride_ms,
            dct_coefficient_count=self.dct_coefficient_count,
        )

        features = audio_dict["audio"]
        if len(features.shape) == 3:
            features = tf.expand_dims(features, axis=0)

        return features

    def load_calibration_data(self, data_source, count: int = None) -> List[Any]:
        """
        Load calibration samples from SWW dataset, preserving dataset ordering.
        """
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
            log_error(f"SWW Preprocessor: Dataset directory not found at {dataset_dir}")
            return []

        train_files, test_files, val_files = get_file_lists(dataset_dir)
        all_audio_files = []
        for file_list in (train_files, val_files, test_files):
            all_audio_files.extend(sorted(file_list))

        if not all_audio_files:
            log_error(f"SWW Preprocessor: No audio files found in {dataset_dir}")
            return []

        calibration_indices = None
        if indices_file and os.path.exists(indices_file):
            try:
                with open(indices_file, "r") as f:
                    calibration_indices = [
                        int(line.strip()) for line in f if line.strip().isdigit()
                    ]
                log_info(
                    f"Loaded {len(calibration_indices)} calibration indices from {indices_file}"
                )
            except Exception as exc:
                log_error(
                    f"SWW Preprocessor: Cannot read calibration indices from {indices_file} - {exc}"
                )
                calibration_indices = None

        if calibration_indices:
            calibration_samples = []
            limit = calibration_indices if count is None else calibration_indices[:count]
            for idx in limit:
                if idx < len(all_audio_files):
                    calibration_samples.append(all_audio_files[idx])
                else:
                    log_error(
                        f"SWW Preprocessor: Calibration index {idx} exceeds dataset size "
                        f"({len(all_audio_files)} samples)"
                    )
        else:
            samples_count = count if count is not None else 120
            calibration_samples = all_audio_files[:samples_count]

        log_info(f"Selected {len(calibration_samples)} calibration samples")
        return calibration_samples
