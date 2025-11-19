#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Keyword spotting model builders.

The DS-CNN topology is adapted from the MLCommons Tiny training reference
(`up-tiny/benchmark/training/keyword_spotting/keras_model.py`) while the module
layout mirrors the image classification example in this repository
(`examples/image_classification/ic_model.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

from kws_preprocessor import WORD_LABELS


@dataclass(frozen=True)
class KWSModelConfig:
    """Configuration for building a keyword spotting model."""

    sample_rate: int = 16_000
    clip_duration_ms: int = 1_000
    window_size_ms: int = 30
    window_stride_ms: int = 20
    dct_coefficient_count: int = 10
    label_count: int = len(WORD_LABELS)
    model_architecture: str = "ds_cnn"
    learning_rate: float = 1e-3
    filters: int = 64
    weight_decay: float = 1e-4
    dropout: float = 0.2
    final_dropout: float = 0.4


def prepare_model_settings(config: KWSModelConfig) -> dict[str, int]:
    """Derive feature dimensions that the training and inference pipelines share."""
    desired_samples = int(config.sample_rate * config.clip_duration_ms / 1000)
    window_size_samples = int(config.sample_rate * config.window_size_ms / 1000)
    window_stride_samples = int(config.sample_rate * config.window_stride_ms / 1000)

    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    fingerprint_size = config.dct_coefficient_count * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": config.dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": config.label_count,
    }


def build_kws_model(config: KWSModelConfig | None = None) -> tf.keras.Model:
    """Return a compiled keyword spotting model."""
    config = config or KWSModelConfig()
    settings = prepare_model_settings(config)

    if settings["spectrogram_length"] <= 0:
        raise ValueError("Computed spectrogram length is non-positive; check window settings.")

    input_shape = (
        settings["spectrogram_length"],
        settings["dct_coefficient_count"],
        1,
    )

    architecture = config.model_architecture.lower()
    if architecture == "ds_cnn":
        model = _build_ds_cnn(input_shape, config, settings)
    elif architecture == "fc4":
        model = _build_fc4(input_shape, config, settings)
    else:
        raise ValueError(f"Unsupported architecture '{config.model_architecture}'.")

    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def _build_ds_cnn(
    input_shape: tuple[int, int, int],
    config: KWSModelConfig,
    settings: dict[str, int],
) -> tf.keras.Model:
    """Construct the DS-CNN topology used in MLCommons Tiny keyword spotting."""
    regularizer = regularizers.l2(config.weight_decay)
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        config.filters,
        kernel_size=(10, 4),
        strides=(2, 2),
        padding="same",
        kernel_regularizer=regularizer,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(rate=config.dropout)(x)

    for _ in range(4):
        x = layers.DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            config.filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.Dropout(rate=config.final_dropout)(x)
    x = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(settings["label_count"])(x)
    outputs = layers.Softmax()(x)

    return models.Model(inputs=inputs, outputs=outputs, name="kws_ds_cnn")


def _build_fc4(
    input_shape: tuple[int, int, int],
    config: KWSModelConfig,
    settings: dict[str, int],
) -> tf.keras.Model:
    """Construct the fully-connected baseline from the TinyML reference."""
    flat_dim = settings["spectrogram_length"] * settings["dct_coefficient_count"]
    model = models.Sequential(name="kws_fc4")
    model.add(layers.Reshape((flat_dim,), input_shape=input_shape))

    for _ in range(3):
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(rate=config.dropout))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(settings["label_count"], activation="softmax"))
    return model
