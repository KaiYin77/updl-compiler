#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
KWS (Keyword Spotting) preprocessor for speech commands dataset
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import List, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler.core.preprocessors.base import DataPreprocessor
from updl_compiler.core.logger import log_info, log_error


# Labels used in KWS models
WORD_LABELS = [
    "Down", "Go", "Left", "No", "Off", "On",
    "Right", "Stop", "Up", "Yes", "Silence", "Unknown"
]


class KWSPreprocessor(DataPreprocessor):
    """Preprocessor for KWS models using speech commands dataset"""

    def __init__(self, sample_rate=16000, clip_duration_ms=1000, window_size_ms=30,
                 window_stride_ms=20, dct_coefficient_count=10):
        """
        Initialize KWS preprocessor with audio processing parameters

        Args:
            sample_rate: Audio sample rate in Hz
            clip_duration_ms: Duration of audio clips in milliseconds
            window_size_ms: STFT window size in milliseconds
            window_stride_ms: STFT window stride in milliseconds
            dct_coefficient_count: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.clip_duration_ms = clip_duration_ms
        self.window_size_ms = window_size_ms
        self.window_stride_ms = window_stride_ms
        self.dct_coefficient_count = dct_coefficient_count

        # Calculate model settings
        self.desired_samples = int(sample_rate * clip_duration_ms / 1000)
        self.window_size_samples = int(sample_rate * window_size_ms / 1000)
        self.window_stride_samples = int(sample_rate * window_stride_ms / 1000)

        length_minus_window = self.desired_samples - self.window_size_samples
        if length_minus_window < 0:
            self.spectrogram_length = 0
        else:
            self.spectrogram_length = 1 + int(length_minus_window / self.window_stride_samples)

    def preprocess_sample(self, sample) -> tf.Tensor:
        """
        Convert speech commands dataset sample to MFCC features

        Args:
            sample: Dict containing 'audio' and 'label' from speech_commands dataset

        Returns:
            tf.Tensor: MFCC features tensor ready for model inference
        """
        # Extract audio from dataset sample
        audio = sample['audio']

        # Convert to proper format
        audio = tf.cast(audio, tf.float32)

        # Ensure correct length
        if tf.shape(audio)[0] < self.desired_samples:
            # Pad if too short
            padding = self.desired_samples - tf.shape(audio)[0]
            audio = tf.pad(audio, [[0, padding]], mode="CONSTANT")
        elif tf.shape(audio)[0] > self.desired_samples:
            # Trim if too long
            audio = audio[:self.desired_samples]

        # Extract MFCC features
        features = self._convert_audio_to_mfcc_features(audio)

        return features

    def _convert_audio_to_mfcc_features(self, wav):
        """Extract MFCC features using TensorFlow to match training pipeline"""

        # Normalize audio
        wav = tf.cast(wav, tf.float32)
        max_val = tf.reduce_max(tf.abs(wav))
        wav = wav / (max_val + 1e-6)

        # Apply time offset (matching the training pipeline)
        padded_wav = tf.pad(wav, [[2, 2]], mode="CONSTANT")
        shifted_wav = tf.slice(padded_wav, [2], [self.desired_samples])

        # Compute STFT with Hann window
        stfts = tf.signal.stft(
            shifted_wav,
            frame_length=self.window_size_samples,
            frame_step=self.window_stride_samples,
            window_fn=tf.signal.hann_window,
        )

        spectrogram = tf.abs(stfts)

        # Compute Mel spectrogram
        num_spectrogram_bins = tf.shape(stfts)[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=40,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=20.0,
            upper_edge_hertz=4000.0,
        )

        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([40]))

        # Compute log-mel spectrogram and extract MFCCs
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.dct_coefficient_count]

        # Reshape to [spectrogram_length, dct_coefficient_count, 1]
        processed_features = tf.reshape(
            mfccs,
            [self.spectrogram_length, self.dct_coefficient_count, 1]
        )

        # Add batch dimension
        features = tf.expand_dims(processed_features, axis=0)

        return features

    def load_calibration_data(self, data_source, count: int = None) -> List[Any]:
        """
        Load calibration samples from speech commands dataset

        Args:
            data_source: Can be:
                - Path to TensorFlow dataset directory
                - Dict with 'dataset_dir' and optional 'indices_file'
                - List of pre-loaded samples
            count: Number of calibration samples to load (if None, uses all available from indices file or default 120)

        Returns:
            List: List of calibration samples
        """
        if isinstance(data_source, list):
            # Already preprocessed samples
            return data_source if count is None else data_source[:count]

        if isinstance(data_source, str):
            # String path to dataset directory
            dataset_dir = data_source
            indices_file = None
        elif isinstance(data_source, dict):
            # Dict with dataset_dir and optional indices_file
            dataset_dir = data_source.get('dataset_dir')
            indices_file = data_source.get('indices_file')
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        # Load speech commands dataset
        try:
            log_info(f"Loading speech_commands dataset from: {dataset_dir}")

            ds_train, ds_info = tfds.load(
                'speech_commands',
                split='train',
                data_dir=dataset_dir,
                with_info=True
            )

            log_info(f"Dataset loaded: {ds_info.splits['train'].num_examples} training samples")

        except Exception as e:
            log_error(f"Error loading dataset: {e}")
            return []

        # Load calibration indices if provided
        calibration_indices = None
        if indices_file and os.path.exists(indices_file):
            try:
                with open(indices_file, 'r') as f:
                    calibration_indices = []
                    for line in f:
                        line = line.strip()
                        if line and line.isdigit():
                            calibration_indices.append(int(line))
                log_info(f"Loaded {len(calibration_indices)} calibration indices from {indices_file}")
            except Exception as e:
                log_error(f"Error loading calibration indices: {e}")

        # Convert dataset to list and select samples
        if calibration_indices:
            # Use specific indices
            ds_list = list(ds_train.take(max(calibration_indices) + 100))  # Take enough samples
            calibration_samples = []
            # If count is None, use all indices; otherwise use the specified count
            indices_to_use = calibration_indices if count is None else calibration_indices[:count]
            for idx in indices_to_use:
                if idx < len(ds_list):
                    calibration_samples.append(ds_list[idx])
                else:
                    log_error(f"Index {idx} is out of range, skipping")
        else:
            # Use first N samples (default 120 if count is None)
            samples_count = count if count is not None else 120
            calibration_samples = list(ds_train.take(samples_count))

        log_info(f"Selected {len(calibration_samples)} calibration samples")
        return calibration_samples