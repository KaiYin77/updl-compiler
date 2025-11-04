#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""
Reusable helpers for generating quantized int16 input fixtures.

The functions in this module are intentionally lightweight so that example
scripts (e.g. keyword spotting, sound detection) can share the same workflow:

1. Load input quantization parameters from the compiler cache.
2. Sample a dataset via TensorFlow Datasets.
3. Run the project-specific preprocessor to obtain float32 feature tensors.
4. Quantize features to int16 using the shared quantizer.
5. Emit a C array via the compiler serializer helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ..core.quantization import QuantizationConfig
from ..core.quantizer import quantize_input_data_fp32_to_int16
from ..core.serializer import serialize_input_feature_to_c_array
from ..core.license import MLPERF_APACHE_LICENSE_HEADER


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for dataset-driven feature generation."""

    dataset_dir: Path
    quant_params_path: Path
    output_c_path: Path
    dataset_name: str = "speech_commands"
    sample_count: int = 20
    random_seed: int = 1234
    array_name: str = "g_model_inputs_int16"
    outer_dim_token: str | None = None
    inner_dim_token: str | None = None
    include_directive: str | None = None
    output_header_path: Path | None = None
    header_guard: str | None = None


def _camelize(identifier: str) -> str:
    """Convert snake_case (or dash-separated) tokens into CamelCase."""
    parts = identifier.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts if part)


def build_config_with_tokens(
    *,
    base_dir: Path,
    dataset_dir: Path,
    model_token: str,
    backend_token: str,
    dtype_token: str,
    dataset_name: str = "speech_commands",
    sample_count: int = 20,
    random_seed: int = 1234,
    **overrides,
) -> GenerationConfig:
    """
    Construct a GenerationConfig using conventional naming patterns.

    This helper centralizes how example scripts derive file names, C identifiers,
    and include directives from a handful of tokens. Callers can optionally
    override any GenerationConfig field by passing keyword arguments.
    """

    base_dir = base_dir.resolve()
    model_token = model_token.strip()
    backend_token = backend_token.strip()
    dtype_token = dtype_token.strip()

    model_camel = _camelize(model_token)
    output_dir = base_dir / backend_token
    header_basename = f"{model_token}_mock_input_data_{dtype_token}.h"

    config_kwargs = {
        "dataset_dir": dataset_dir,
        "quant_params_path": base_dir
        / ".updlc_cache"
        / f"{model_token}_{backend_token}_model_quantize_params.json",
        "output_c_path": output_dir / f"{model_token}_test_input_data_{dtype_token}.c",
        "dataset_name": dataset_name,
        "sample_count": sample_count,
        "random_seed": random_seed,
        "array_name": f"g_{model_token}_inputs_{dtype_token}",
        "outer_dim_token": f"kNum{model_camel}TestInputs",
        "inner_dim_token": f"k{model_camel}InputSize",
        "include_directive": f'#include "{header_basename}"',
        "output_header_path": output_dir / header_basename,
    }

    if overrides:
        config_kwargs.update(overrides)

    return GenerationConfig(**config_kwargs)


def load_input_scale_zero_point(config: GenerationConfig) -> Tuple[float, int]:
    """Return (scale, zero_point) from the compiler-generated JSON cache."""
    quant_config = QuantizationConfig()
    if not quant_config.load_params_from_json(str(config.quant_params_path)):
        raise RuntimeError(f"Cannot load quantization params from {config.quant_params_path}")

    params = quant_config.get_input_params()
    if not params:
        raise RuntimeError(f"No input quantization parameters found in {config.quant_params_path}")

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


LabelExtractor = Callable[[dict], str]


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


def write_c_array(
    config: GenerationConfig,
    quantized_samples: Sequence[np.ndarray],
    labels: Sequence[str],
    input_size: int,
    *,
    license_header: str | None = None,
) -> Path:
    """Serialize quantized samples into a C source file."""
    body = serialize_input_feature_to_c_array(
        quantized_samples,
        labels,
        input_size,
        array_name=config.array_name,
        outer_dim_token=config.outer_dim_token,
        inner_dim_token=config.inner_dim_token,
        include_directive=config.include_directive,
        license_header=license_header,
    )

    output_path = config.output_c_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def write_header_file(
    config: GenerationConfig,
    sample_count: int,
    input_size: int,
    *,
    license_header: str | None = MLPERF_APACHE_LICENSE_HEADER,
) -> Path:
    """Create a header declaring the generated int16 input array."""
    if config.output_header_path is None:
        raise ValueError("output_header_path is not configured.")

    header_path = config.output_header_path
    if not header_path.is_absolute():
        header_path = Path.cwd() / header_path
    header_path.parent.mkdir(parents=True, exist_ok=True)

    guard = config.header_guard
    if not guard:
        guard = header_path.name.replace(".", "_").upper()

    lines: list[str] = []
    if license_header:
        lines.append(license_header.strip("\n"))
        lines.append("")

    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    if config.outer_dim_token:
        lines.append(f"#define {config.outer_dim_token} {sample_count}")
    if config.inner_dim_token:
        lines.append(f"#define {config.inner_dim_token} {input_size}")

    outer = config.outer_dim_token or str(sample_count)
    inner = config.inner_dim_token or str(input_size)

    lines.append("")
    lines.append(
        f"extern const int16_t {config.array_name}[{outer}][{inner}];"
    )
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")

    header_path.write_text("\n".join(lines), encoding="utf-8")
    return header_path
