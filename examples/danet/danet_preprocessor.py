#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
DANet (Dynamic Adaptive Network) preprocessor for IMU sensor fusion
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Any
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler.core.preprocessors.base import DataPreprocessor
from updl_compiler.core.logger import log_info, log_error


class DANetPreprocessor(DataPreprocessor):
    """Preprocessor for DANet models using IMU sensor data"""

    def __init__(self, acc_factor=9.81, gyro_factor=4.0, diff_factor=5.0,
                 lpf_alpha_rise=0.5, lpf_alpha_fall=0.01):
        """
        Initialize DANet preprocessor with sensor processing parameters

        Args:
            acc_factor: Scaling factor for accelerometer normalization
            gyro_factor: Scaling factor for gyroscope normalization
            diff_factor: Scaling factor for acceleration difference normalization
            lpf_alpha_rise: Low-pass filter alpha for rising acceleration differences
            lpf_alpha_fall: Low-pass filter alpha for falling acceleration differences
        """
        self.acc_factor = acc_factor
        self.gyro_factor = gyro_factor
        self.diff_factor = diff_factor
        self.lpf_alpha_rise = lpf_alpha_rise
        self.lpf_alpha_fall = lpf_alpha_fall

    def preprocess_sample(self, sample) -> tf.Tensor:
        """
        Convert IMU sensor sample to DANet input features

        Args:
            sample: Dict containing accelerometer and gyroscope data

        Returns:
            tf.Tensor: Feature tensor ready for DANet inference
        """
        # Extract sensor data
        acc_data = np.array([sample['ax'], sample['ay'], sample['az']], dtype=np.float32)
        gyro_data = np.array([sample['gx'], sample['gy'], sample['gz']], dtype=np.float32)
        dt = sample.get('dt', 0.005)  # Default 200Hz sampling rate

        # Apply softsign normalization
        nn_acc = self._softsign_normalize(acc_data, self.acc_factor)
        nn_gyro = self._softsign_normalize(gyro_data, self.gyro_factor)

        # Calculate acceleration difference with asymmetric filtering
        acc_norm = np.linalg.norm(acc_data)
        acc_diff = abs(acc_norm - 9.81)

        # Apply asymmetric low-pass filtering (simplified for single sample)
        nn_diff = self._softsign_normalize(np.array([acc_diff]), self.diff_factor)

        # Default beta value for initial inference
        beta_prev = np.array([0.1], dtype=np.float32)
        dt_array = np.array([dt], dtype=np.float32)

        # Concatenate all features: [acc(3), gyro(3), diff(1), beta(1), dt(1)]
        features = np.concatenate([nn_acc, nn_gyro, nn_diff, beta_prev, dt_array])

        # Convert to tensor and add batch dimension
        return tf.expand_dims(tf.convert_to_tensor(features, dtype=tf.float32), 0)

    def postprocess_output(self, raw_beta):
        """
        Post-process model output to get final beta value
        Model outputs raw sigmoid [0,1], scale to [0.001, 0.301]

        Args:
            raw_beta: Raw model output (sigmoid values)

        Returns:
            Scaled beta value
        """
        return raw_beta * 0.3 + 0.001

    def _softsign_normalize(self, x, factor):
        """Apply softsign normalization: x / (factor * (1 + |x/factor|))"""
        scaled = x / factor
        return scaled / (1 + np.abs(scaled))

    def load_calibration_data(self, data_source, count: int = None) -> List[Any]:
        """
        Load calibration samples from IMU dataset

        Args:
            data_source: Can be:
                - Path to CSV file with IMU data
                - Dict with 'dataset_path' and optional 'indices_file'
                - List of pre-loaded samples
            count: Number of calibration samples to load

        Returns:
            List: List of calibration samples
        """
        if isinstance(data_source, list):
            # Already preprocessed samples
            return data_source if count is None else data_source[:count]

        if isinstance(data_source, str):
            # String path to CSV file
            csv_path = data_source
            indices_file = None
        elif isinstance(data_source, dict):
            # Dict with csv_path and optional indices_file
            csv_path = data_source.get('dataset_path')
            indices_file = data_source.get('indices_file')
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        # Load IMU dataset
        try:
            log_info(f"Loading IMU dataset from: {csv_path}")
            df = pd.read_csv(csv_path)

            # Verify required columns exist
            required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")

            log_info(f"Dataset loaded: {len(df)} IMU samples")

        except Exception as e:
            log_error(f"DANet Preprocessor: Cannot load IMU dataset from {csv_path} - {e}")
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
                log_error(f"DANet Preprocessor: Cannot read calibration indices from {indices_file} - {e}")

        # Convert dataframe to list of samples and select calibration samples
        if calibration_indices:
            # Use specific indices
            calibration_samples = []
            indices_to_use = calibration_indices if count is None else calibration_indices[:count]
            for idx in indices_to_use:
                if idx < len(df):
                    sample = df.iloc[idx].to_dict()
                    calibration_samples.append(sample)
                else:
                    log_error(f"DANet Preprocessor: Calibration index {idx} exceeds dataset size ({len(df)} samples)")
        else:
            # Use first N samples (default 120 if count is None)
            samples_count = count if count is not None else 120
            samples_count = min(samples_count, len(df))
            calibration_samples = []
            for i in range(samples_count):
                sample = df.iloc[i].to_dict()
                calibration_samples.append(sample)

        log_info(f"Selected {len(calibration_samples)} calibration samples")
        return calibration_samples