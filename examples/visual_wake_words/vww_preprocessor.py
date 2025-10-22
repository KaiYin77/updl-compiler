#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
VWW (Visual Wake Words) preprocessor for COCO dataset
"""

import os
import tensorflow as tf
import numpy as np
from typing import List, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler.core.preprocessors.base import DataPreprocessor
from updl_compiler.core.logger import log_info, log_error


# Labels used in VWW models
VWW_LABELS = ["no_person", "person"]


class VWWPreprocessor(DataPreprocessor):
    """Preprocessor for VWW models using COCO dataset"""

    def __init__(self, image_size=96):
        """
        Initialize VWW preprocessor with image processing parameters

        Args:
            image_size: Size to resize images to (square, image_size x image_size)
        """
        self.image_size = image_size

    def preprocess_sample(self, sample) -> tf.Tensor:
        """
        Convert image sample to preprocessed tensor for VWW model

        Args:
            sample: Can be:
                - tf.Tensor representing an image
                - Dict containing 'image' key
                - String path to image file

        Returns:
            tf.Tensor: Preprocessed image tensor ready for model inference
        """
        # Extract image from different input formats
        if isinstance(sample, dict):
            if 'image' in sample:
                image = sample['image']
            else:
                raise ValueError("Dict sample must contain 'image' key")
        elif isinstance(sample, str):
            # Load image from file path
            image = tf.io.read_file(sample)
            image = tf.image.decode_image(image, channels=3)
        else:
            # Assume it's already a tensor
            image = sample

        # Convert to float32
        image = tf.cast(image, tf.float32)

        # Resize image to target size
        image = tf.image.resize(image, [self.image_size, self.image_size])

        # Normalize to [0, 1] range
        image = image / 255.0

        # Add batch dimension if not present
        if len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis=0)

        return image

    def load_calibration_data(self, data_source, count: int = None) -> List[Any]:
        """
        Load calibration samples from VWW dataset

        Args:
            data_source: Can be:
                - Path to VWW dataset directory (containing person/ and no_person/ subdirs)
                - Dict with 'dataset_dir' and optional 'indices_file'
                - List of pre-loaded samples

            count: Number of calibration samples to load (if None, uses all available from indices file or default 120)

        Returns:
            List: List of calibration samples (image paths or tensors)
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

        # Look for VWW dataset structure
        vww_dataset_path = os.path.join(dataset_dir, 'vw_coco2014_96')

        if not os.path.exists(vww_dataset_path):
            log_error(f"VWW Preprocessor: VWW dataset not found at {vww_dataset_path}. Please run download.sh first.")
            return []

        # Collect all image files from person and no_person directories
        all_image_files = []

        for class_dir in ['person', 'no_person']:
            class_path = os.path.join(vww_dataset_path, class_dir)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_image_files.append(os.path.join(class_path, img_file))

        log_info(f"Found {len(all_image_files)} images in VWW dataset")

        if not all_image_files:
            log_error(f"VWW Preprocessor: No image files found in {vww_dataset_path}")
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
                log_error(f"VWW Preprocessor: Cannot read calibration indices from {indices_file} - {e}. Check file format and permissions.")

        # Select calibration samples
        if calibration_indices:
            # Use specific indices
            calibration_samples = []
            # If count is None, use all indices; otherwise use the specified count
            indices_to_use = calibration_indices if count is None else calibration_indices[:count]
            for idx in indices_to_use:
                if idx < len(all_image_files):
                    calibration_samples.append(all_image_files[idx])
                else:
                    log_error(f"VWW Preprocessor: Calibration index {idx} exceeds dataset size ({len(all_image_files)} samples). This index will be skipped.")
        else:
            # Use first N samples (default 120 if count is None)
            samples_count = count if count is not None else 120
            calibration_samples = all_image_files[:samples_count]

        log_info(f"Selected {len(calibration_samples)} calibration samples")
        return calibration_samples