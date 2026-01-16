#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Layer capture and layout utilities for test generation."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

from ..core.schema.uph5 import LTYPE_LIST


def list_capture_layers(
    model: tf.keras.Model,
    *,
    skip_input_layers: bool = True,
) -> List[tf.keras.layers.Layer]:
    """Return an ordered list of layers suitable for activation capture."""

    layers: List[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if skip_input_layers and isinstance(layer, tf.keras.layers.InputLayer):
            continue
        layers.append(layer)
    return layers


def capture_layer_outputs(
    model: tf.keras.Model,
    features: np.ndarray,
    target_layers: Sequence[str | tf.keras.layers.Layer],
    *,
    batch_size: int | None = None,
) -> List[np.ndarray]:
    """Run inference and return activations for the specified layers."""

    resolved_layers: List[tf.keras.layers.Layer] = []
    for layer_ref in target_layers:
        if isinstance(layer_ref, tf.keras.layers.Layer):
            resolved_layers.append(layer_ref)
        else:
            resolved_layers.append(model.get_layer(layer_ref))

    capture_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in resolved_layers],
        name="updl_activation_capture",
    )
    activations = capture_model.predict(
        features,
        batch_size=batch_size or features.shape[0],
        verbose=0,
    )
    if not isinstance(activations, list):
        activations = [activations]
    return activations


def transform_activation_layout(
    activation: np.ndarray, layer_type: str, layout: str
) -> np.ndarray:
    """Convert activations between TensorFlow (NHWC) and UPDL (NCHW) layouts."""
    if layout == "tf":
        return activation

    if len(activation.shape) == 4:
        return np.transpose(activation, (0, 3, 1, 2))
    if len(activation.shape) == 3:
        return np.transpose(activation, (0, 2, 1))
    return activation


def requires_activation_layout_transform(layer_type: str) -> bool:
    """Return True if the layer produces spatial activations needing NHWC->NCHW."""
    spatial_layers = [
        "Conv1D",
        "Conv2D",
        "DepthwiseConv2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "BatchNormalization",
        "Activation",
        "Dropout",
        "Add",
    ]
    return layer_type in spatial_layers


def find_major_computational_layer(
    model: tf.keras.Model, target_layer_index: int
) -> str:
    """Return the layer type that should drive layout decisions."""
    layers = list_capture_layers(model)
    target_layer = layers[target_layer_index]
    target_layer_type = target_layer.__class__.__name__

    if target_layer_type in LTYPE_LIST:
        return target_layer_type

    dependent_layer_types = ["BatchNormalization", "Activation", "Dropout"]
    if target_layer_type in dependent_layer_types:
        for i in range(target_layer_index - 1, -1, -1):
            prev_layer = layers[i]
            prev_layer_type = prev_layer.__class__.__name__
            if prev_layer_type in LTYPE_LIST:
                return prev_layer_type

    return target_layer_type


def get_layer_layout_info(
    model: tf.keras.Model, layer_indices: List[int], layout: str
) -> Dict[int, Dict[str, str]]:
    """Describe layout expectations for each selected layer."""
    layers = list_capture_layers(model)
    layout_info = {}

    for idx in layer_indices:
        layer = layers[idx]
        layer_type = layer.__class__.__name__
        major_layer_type = find_major_computational_layer(model, idx)

        layout_info[idx] = {
            "layer_name": layer.name,
            "layer_type": layer_type,
            "major_layer_type": major_layer_type,
            "requires_transformation": requires_activation_layout_transform(
                major_layer_type
            ),
            "layout_format": layout,
        }

    return layout_info
