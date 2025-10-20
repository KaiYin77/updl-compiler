#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for data preprocessors
"""

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Any


class DataPreprocessor(ABC):
    """Base class for model-specific data preprocessors"""

    @abstractmethod
    def preprocess_sample(self, sample) -> tf.Tensor:
        """
        Convert raw data sample to model input format

        Args:
            sample: Raw data sample (format depends on model type)

        Returns:
            tf.Tensor: Preprocessed tensor ready for model inference
        """
        pass

    @abstractmethod
    def load_calibration_data(self, data_source, count: int = 120) -> List[Any]:
        """
        Load calibration dataset for quantization analysis

        Args:
            data_source: Path or source of calibration data
            count: Number of calibration samples to load

        Returns:
            List: List of calibration samples
        """
        pass