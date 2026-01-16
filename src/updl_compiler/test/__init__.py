#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared across UPDL compiler example generators."""

from .configs import GenerationConfig, ModelConfig, build_generation_config, create_model_config
from .data import (
    collect_features_and_labels,
    extract_labeled_features,
    load_input_scale_zero_point,
    load_layer_quant_params,
    quantize_features,
    sample_tfds_dataset,
)
from .generation import generate_artifacts
from .io import (
    format_float_array_to_c,
    save_results,
    write_c_array,
    write_header_file,
)
from .layers import (
    capture_layer_outputs,
    find_major_computational_layer,
    get_layer_layout_info,
    list_capture_layers,
    requires_activation_layout_transform,
    transform_activation_layout,
)
from .prompts import prompt_layer_indices, prompt_layout_selection

__all__ = [
    "GenerationConfig",
    "ModelConfig",
    "build_generation_config",
    "create_model_config",
    "generate_artifacts",
    "capture_layer_outputs",
    "extract_labeled_features",
    "load_input_scale_zero_point",
    "load_layer_quant_params",
    "format_float_array_to_c",
    "prompt_layer_indices",
    "prompt_layout_selection",
    "list_capture_layers",
    "quantize_features",
    "sample_tfds_dataset",
    "write_c_array",
    "write_header_file",
    "save_results",
    "transform_activation_layout",
    "requires_activation_layout_transform",
    "find_major_computational_layer",
    "get_layer_layout_info",
    "collect_features_and_labels",
]
