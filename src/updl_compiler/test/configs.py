#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses and helpers for test artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple


@dataclass(frozen=True)
class GenerationConfig:
    """Describes how dataset samples map to emitted C artifacts."""

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
    element_type: str = "int16_t"
    header_includes: Tuple[str, ...] = ("#include <stdint.h>",)
    values_per_line: int = 16


@dataclass(frozen=True)
class ModelConfig:
    """Model-specific metadata used to drive test generation."""

    name: str
    display_name: str
    preprocessor_class: type
    label_extractor: Callable[[dict], str]
    dataset_name: str
    dataset_dir: Path
    sample_count: int = 5
    random_seed: int = 1234
    array_name_template: str = "g_{}_test_inputs_fp32"
    outer_dim_token_template: str = "kNum{}TestInputs"
    inner_dim_token_template: str = "k{}InputSize"
    element_type: str = "float"
    header_includes: Tuple[str, ...] = ("#include <stddef.h>",)
    values_per_line: int = 8
    data_loader: Callable[[Path, int, int], list[dict]] | None = None


MODEL_PRESETS = {
    "ic": {
        "name": "ic",
        "display_name": "Image Classification",
        "dataset_name": "cifar10",
        "sample_count": 5,
    },
    "kws": {
        "name": "kws",
        "display_name": "Keyword Spotting",
        "dataset_name": "speech_commands",
        "sample_count": 10,
    },
    "vww": {
        "name": "vww",
        "display_name": "Visual Wake Words",
        "dataset_name": "visual_wake_words",
        "sample_count": 5,
    },
    "danet": {
        "name": "danet",
        "display_name": "Dynamic Adaptive Network",
        "dataset_name": "imu_data",
        "sample_count": 1,
    },
}


def create_model_config(model_type: str, **overrides) -> ModelConfig:
    """Return a copy of the requested model preset with the provided overrides."""
    if model_type not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_PRESETS.keys())}"
        )

    base = dict(MODEL_PRESETS[model_type])
    base.update(overrides)
    return ModelConfig(**base)


def build_generation_config(
    model_config: ModelConfig,
    *,
    example_dir: Path,
    output_dir: Path,
    array_suffix: str = "test_inputs_fp32",
    include_header: bool = True,
) -> GenerationConfig:
    """Construct a GenerationConfig with consistent naming conventions."""
    name_cap = model_config.name.capitalize()
    base_name = f"{model_config.name}_{array_suffix}"

    header_name = f"{base_name}.h"
    include_directive = f'#include "{header_name}"' if include_header else None

    return GenerationConfig(
        dataset_dir=model_config.dataset_dir,
        dataset_name=model_config.dataset_name,
        sample_count=model_config.sample_count,
        random_seed=model_config.random_seed,
        quant_params_path=example_dir / ".updlc_cache" / "unused_fp32_params.json",
        output_c_path=output_dir / f"{base_name}.c",
        output_header_path=output_dir / header_name if include_header else None,
        array_name=model_config.array_name_template.format(model_config.name),
        outer_dim_token=model_config.outer_dim_token_template.format(name_cap),
        inner_dim_token=model_config.inner_dim_token_template.format(name_cap),
        include_directive=include_directive,
        header_guard=f"{model_config.name.upper()}_{array_suffix.upper()}_H"
        if include_header
        else None,
        element_type=model_config.element_type,
        header_includes=model_config.header_includes,
        values_per_line=model_config.values_per_line,
    )
