#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Dataset sampling and preprocessing helpers for test generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .configs import GenerationConfig
from ..core.quantization import QuantizationConfig
from ..core.quantizer import quantize_input_data_fp32_to_int16

LabelExtractor = Callable[[dict], str]


def load_layer_quant_params(
    path: Path,
) -> List[Tuple[str, List[str], dict[str, float]]]:
    """Load per-layer quantization parameters from a compiler cache JSON file."""
    data = json.loads(path.read_text())
    layers: List[Tuple[str, List[str], dict[str, float]]] = []

    for fusable_layer, ops in data["layers"].items():
        op_order = list(ops.keys())
        terminal_op = op_order[-1]
        cfg = ops[terminal_op]
        layers.append(
            (
                fusable_layer,
                op_order,
                {
                    "scale": float(cfg["scale"]),
                    "zero_point": int(cfg.get("zero_point", 0)),
                    "min_val": float(cfg.get("min_val", 0.0)),
                    "max_val": float(cfg.get("max_val", 0.0)),
                },
            )
        )

    return layers


def load_input_scale_zero_point(config: GenerationConfig) -> Tuple[float, int]:
    """Return (scale, zero_point) from the compiler-generated JSON cache."""
    quant_config = QuantizationConfig()
    if not quant_config.load_params_from_json(str(config.quant_params_path)):
        raise RuntimeError(
            f"Cannot load quantization params from {config.quant_params_path}"
        )

    params = quant_config.get_input_params()
    if not params:
        raise RuntimeError(
            f"No input quantization parameters found in {config.quant_params_path}"
        )

    try:
        scale = float(params["scale"])
        zero_point = int(params.get("zero_point", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed input quantization parameters in {config.quant_params_path}: {exc}"
        ) from exc

    return scale, zero_point


def sample_tfds_dataset(config: GenerationConfig) -> List[dict]:
    """Sample `sample_count` elements from the configured TFDS dataset."""
    if not config.dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {config.dataset_dir}")

    dataset = tfds.load(
        config.dataset_name,
        split="train",
        data_dir=str(config.dataset_dir),
    )
    shuffled = dataset.shuffle(
        buffer_size=10_000,
        seed=config.random_seed,
        reshuffle_each_iteration=False,
    )
    samples = list(shuffled.take(config.sample_count))
    if len(samples) < config.sample_count:
        print(
            f"[WARN] Requested {config.sample_count} samples but dataset only yielded {len(samples)}; "
            "continuing with available samples."
        )
    return samples


def extract_labeled_features(
    preprocessor: Callable[[dict], tf.Tensor],
    samples: Iterable[dict],
    label_extractor: LabelExtractor,
) -> List[Tuple[str, np.ndarray]]:
    """Run the model preprocessor over dataset samples and attach labels."""
    labeled_features: List[Tuple[str, np.ndarray]] = []
    for sample in samples:
        label = label_extractor(sample)
        features = preprocessor(sample).numpy().astype(np.float32)
        labeled_features.append((label, features))
    return labeled_features


def quantize_features(
    labeled_features: Sequence[Tuple[str, np.ndarray]],
    scale: float,
    zero_point: int,
) -> Tuple[List[np.ndarray], List[str]]:
    """Quantize float features to int16 and return flattened vectors plus labels."""
    quantized_vectors: List[np.ndarray] = []
    labels: List[str] = []

    for idx, (label, feature_np) in enumerate(labeled_features):
        quantized, overflow_count = quantize_input_data_fp32_to_int16(
            feature_np,
            scale,
            zero_point,
            return_overflow_count=True,
        )
        if overflow_count:
            print(
                f"[WARN] Clipped {overflow_count} values outside int16 range for sample {idx} ({label})."
            )
        quantized_vectors.append(quantized.flatten())
        labels.append(label)

    return quantized_vectors, labels


def collect_features_and_labels(
    layer_config: GenerationConfig,
    preprocessor: Any,
    label_extractor: LabelExtractor,
    custom_data_loader: Callable[[Path, int, int], List[dict]] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Sample data using the provided config and return float features + labels."""
    if custom_data_loader is not None:
        raw_samples = custom_data_loader(
            layer_config.dataset_dir,
            layer_config.sample_count,
            layer_config.random_seed,
        )
    else:
        raw_samples = sample_tfds_dataset(layer_config)

    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        label_extractor,
    )

    features = np.concatenate([feat for _, feat in labeled_features], axis=0)
    labels = [label for label, _ in labeled_features]
    return features, labels
