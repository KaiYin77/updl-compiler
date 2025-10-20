#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from .logger import log_info, log_debug


def extract_activation_function(layer):
    """Extract activation function from layer config"""
    try:
        if hasattr(layer, "get_config"):
            config = layer.get_config()
            return config.get("activation", "linear")
    except:
        pass
    return "linear"


def split_dense_softmax_layers(model):
    """
    Split final Dense layer with softmax activation into separate Dense (linear) + Softmax layers

    Args:
        model: Original Keras model

    Returns:
        Modified model with separated Dense and Softmax layers
    """
    log_info("Checking for Dense+Softmax layers to split...")

    # Find the last Dense layer with softmax activation
    last_dense_idx = None
    last_dense_layer = None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            activation = extract_activation_function(layer)
            if activation == "softmax":
                last_dense_idx = i
                last_dense_layer = layer
                log_info(f"Found Dense+Softmax layer at index {i}: {layer.name}")
                break

    if last_dense_idx is None:
        log_info("No Dense layer with softmax activation found - returning original model")
        return model

    # Count existing Dense and Softmax layers to determine proper naming
    dense_count = 0
    softmax_count = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_count += 1
        elif isinstance(layer, tf.keras.layers.Softmax) or (
            hasattr(layer, "name") and "softmax" in layer.name.lower()
        ):
            softmax_count += 1

    # Get the input to the Dense layer (output from previous layer)
    if last_dense_idx > 0:
        dense_input = model.layers[last_dense_idx - 1].output
    else:
        dense_input = model.input

    # Create new Dense layer with linear activation (same weights and bias)
    dense_config = last_dense_layer.get_config()
    dense_config["activation"] = "linear"  # Change activation to linear

    # Generate proper Dense layer name following Keras convention
    if dense_count == 1:
        new_dense_name = "dense"
    else:
        new_dense_name = f"dense_{dense_count - 1}"

    dense_config["name"] = new_dense_name

    # Create new Dense layer
    new_dense = tf.keras.layers.Dense.from_config(dense_config)
    dense_output = new_dense(dense_input)

    # Copy weights from original layer
    new_dense.set_weights(last_dense_layer.get_weights())

    # Generate proper Softmax layer name following Keras convention
    if softmax_count == 0:
        softmax_name = "softmax"
    else:
        softmax_name = f"softmax_{softmax_count}"

    # Add Softmax activation as separate layer
    softmax_layer = tf.keras.layers.Softmax(name=softmax_name)
    final_output = softmax_layer(dense_output)

    # Create new model
    new_model = tf.keras.Model(inputs=model.input, outputs=final_output)

    log_info(f"Created new Dense layer: {new_dense.name} (activation=linear)")
    log_info(f"Created new Softmax layer: {softmax_layer.name}")
    log_info(f"Original model layers: {len(model.layers)} → New model layers: {len(new_model.layers)}")

    return new_model


def load_model(input_file, auto_split=True):
    """Load Keras model from HDF5 file or directory with optional auto-splitting

    Args:
        input_file: Path to HDF5 model file or SavedModel directory
        auto_split: Whether to automatically split Dense+Softmax layers

    Returns:
        Loaded Keras model (optionally with split layers)
    """
    log_info(f"Loading model from: {input_file}")

    try:
        model = tf.keras.models.load_model(input_file)
        log_info(f"Model loaded successfully: {model.name}")
        log_debug(f"Model summary:")
        model.summary()

        # Auto-split Dense+Softmax layers if requested
        if auto_split:
            model = split_dense_softmax_layers(model)

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {input_file}: {e}")
