#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from .logger import log_info, log_debug


def load_model(input_file):
    """Load Keras model from HDF5 file

    Args:
        input_file: Path to HDF5 model file

    Returns:
        Loaded Keras model
    """
    log_info(f"Loading model from: {input_file}")

    try:
        model = tf.keras.models.load_model(input_file)
        log_info(f"Model loaded successfully: {model.name}")
        log_debug(f"Model summary:")
        if log_debug:  # Only print if debug logging is enabled
            model.summary()
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {input_file}: {e}")
