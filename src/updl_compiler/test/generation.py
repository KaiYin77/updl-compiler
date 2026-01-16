#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Public orchestration helpers for example/test artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Tuple

from rich.console import Console

from .configs import (
    GenerationConfig,
    ModelConfig,
    build_generation_config,
    create_model_config,
)
from .data import extract_labeled_features, sample_tfds_dataset
from .io import write_c_array, write_header_file
from .workflows import LayerOutputGenerator, TestInputGenerator

__all__ = [
    "GenerationConfig",
    "ModelConfig",
    "build_generation_config",
    "create_model_config",
    "TestInputGenerator",
    "LayerOutputGenerator",
    "generate_artifacts",
    "generate_test_inputs_fp32",
]


def generate_artifacts(
    *,
    model_config: ModelConfig,
    example_dir: Path,
    mode: Literal["inputs", "layers"] = "inputs",
    console: Console | None = None,
    layout: str | None = None,
    model_path: Path | None = None,
) -> Tuple[Path, Path] | None:
    """
    Run the requested generation workflow and return emitted file paths.

    Args:
        model_config: Model configuration describing preprocessors/datasets
        example_dir: Root of the example project (contains datasets and uph5/)
        mode: "inputs" for test inputs or "layers" for intermediate activations
        console: Optional Rich console for interactive prompts
        layout: Optional layout override ("tf" or "updl")
    """
    console = console or Console()

    if mode == "inputs":
        generator = TestInputGenerator(model_config, example_dir, console=console)
        return generator.generate_test_inputs(layout_override=layout)

    if mode == "layers":
        if model_path is None:
            raise ValueError("model_path is required when mode='layers'")
        generator = LayerOutputGenerator(
            model_config, example_dir, model_path=model_path, console=console
        )
        generator.generate_layer_outputs(layout_override=layout)
        return None

    raise ValueError(f"Unsupported generation mode: {mode!r}")


def generate_test_inputs_fp32(
    config: GenerationConfig,
    preprocessor: Any,
    label_extractor: Callable[[dict], str],
    license_header: str | None = None,
) -> Tuple[Path, Path]:
    """Legacy helper that generates fp32 inputs without user interaction."""
    raw_samples = sample_tfds_dataset(config)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        label_extractor,
    )

    if not labeled_features:
        raise RuntimeError("No samples were collected; check dataset configuration.")

    labels = []
    flat_samples = []
    for label, features in labeled_features:
        labels.append(label)
        flat_samples.append(features.astype("float32", copy=False).flatten())

    input_size = flat_samples[0].size

    output_path = write_c_array(
        config,
        flat_samples,
        labels,
        input_size,
        license_header=license_header,
        element_type="float",
        values_per_line=8,
    )
    header_path = write_header_file(
        config,
        len(flat_samples),
        input_size,
        license_header=license_header,
        element_type="float",
        header_includes=("#include <stddef.h>",),
    )
    return output_path, header_path
