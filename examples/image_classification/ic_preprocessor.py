#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
IC (Image Classification) preprocessor for CIFAR-10 dataset
"""

import os
import tensorflow as tf
import numpy as np
from typing import List, Any
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler.core.preprocessors.base import DataPreprocessor
from updl_compiler.core.logger import log_info, log_error


# Labels used in CIFAR-10
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class ICPreprocessor(DataPreprocessor):
    """Preprocessor for IC models using CIFAR-10 dataset"""

    def __init__(self, image_size=32):
        """
        Initialize IC preprocessor with image processing parameters

        Args:
            image_size: Size of CIFAR-10 images (32x32)
        """
        self.image_size = image_size

    def preprocess_sample(self, sample) -> tf.Tensor:
        """
        Convert CIFAR-10 sample to preprocessed tensor for IC model

        Args:
            sample: Can be:
                - tf.Tensor representing an image
                - Dict containing 'image' key
                - Numpy array from CIFAR-10 dataset

        Returns:
            tf.Tensor: Preprocessed image tensor ready for model inference
        """
        # Extract image from different input formats
        if isinstance(sample, dict):
            if 'image' in sample:
                image = sample['image']
            else:
                raise ValueError("Dict sample must contain 'image' key")
        else:
            # Assume it's already a tensor or numpy array
            image = sample

        # Convert to float32
        image = tf.cast(image, tf.float32)

        # Ensure correct shape [32, 32, 3]
        if len(tf.shape(image)) == 3:
            if tf.shape(image)[0] != self.image_size or tf.shape(image)[1] != self.image_size:
                image = tf.image.resize(image, [self.image_size, self.image_size])
        else:
            raise ValueError(f"Expected 3D image tensor, got shape {tf.shape(image)}")

        # Normalize to [0, 1] range (CIFAR-10 is typically in [0, 255])
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0

        # Add batch dimension if not present
        if len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis=0)

        return image

    def unpickle(self, file):
        """Load the CIFAR-10 data from pickle files"""
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def load_cifar_10_data(self, data_dir):
        """
        Load CIFAR-10 training data from pickle files

        Returns:
            cifar_train_data: Training images as numpy array
        """
        # Load training data
        cifar_train_data = None

        for i in range(1, 6):
            cifar_train_data_dict = self.unpickle(os.path.join(data_dir, f"data_batch_{i}"))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))

        # Reshape and transpose to get [N, 32, 32, 3] format
        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)

        return cifar_train_data

    def load_calibration_data(self, data_source, count: int = None) -> List[Any]:
        """
        Load calibration samples from CIFAR-10 dataset

        Args:
            data_source: Can be:
                - Path to CIFAR-10 dataset directory (containing data_batch_* files)
                - Dict with 'dataset_dir' and optional 'indices_file'
                - List of pre-loaded samples

            count: Number of calibration samples to load (if None, uses all available from indices file or default 120)

        Returns:
            List: List of calibration samples (image arrays)
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

        # Check if CIFAR-10 dataset exists
        if not os.path.exists(dataset_dir):
            log_error(f"IC Preprocessor: CIFAR-10 dataset directory not found at {dataset_dir}")
            return []

        # Check for required files
        required_files = [f"data_batch_{i}" for i in range(1, 6)]
        for file in required_files:
            if not os.path.exists(os.path.join(dataset_dir, file)):
                log_error(f"IC Preprocessor: Required CIFAR-10 file {file} not found in {dataset_dir}")
                return []

        try:
            # Load CIFAR-10 data
            log_info(f"Loading CIFAR-10 dataset from: {dataset_dir}")
            cifar_train_data = self.load_cifar_10_data(dataset_dir)
            log_info(f"Dataset loaded: {len(cifar_train_data)} training samples")

        except Exception as e:
            log_error(f"IC Preprocessor: Cannot load CIFAR-10 dataset from {dataset_dir} - {e}")
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
                log_error(f"IC Preprocessor: Cannot read calibration indices from {indices_file} - {e}")

        # Select calibration samples
        if calibration_indices:
            # Use specific indices
            calibration_samples = []
            # If count is None, use all indices; otherwise use the specified count
            indices_to_use = calibration_indices if count is None else calibration_indices[:count]
            for idx in indices_to_use:
                if idx < len(cifar_train_data):
                    calibration_samples.append(cifar_train_data[idx])
                else:
                    log_error(f"IC Preprocessor: Calibration index {idx} exceeds dataset size ({len(cifar_train_data)} samples). This index will be skipped.")
        else:
            # Use first N samples (default 120 if count is None)
            samples_count = count if count is not None else 120
            calibration_samples = cifar_train_data[:samples_count]

        log_info(f"Selected {len(calibration_samples)} calibration samples")
        return calibration_samples