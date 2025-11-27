#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared across UPDL compiler example generators."""

from .generation import (
    GenerationConfig,
    capture_layer_outputs,
    extract_labeled_features,
    load_input_scale_zero_point,
    load_layer_quant_params,
    format_float_array_to_c,
    prompt_layer_indices,
    list_capture_layers,
    quantize_features,
    sample_tfds_dataset,
    write_c_array,
    write_header_file,
    save_results,
    transform_activation_layout,
    requires_activation_layout_transform,
    find_major_computational_layer,
    get_layer_layout_info,
    prompt_layout_selection,
    collect_features_and_labels,
    generate_test_inputs_fp32,
)

__all__ = [
    "GenerationConfig",
    "capture_layer_outputs",
    "extract_labeled_features",
    "load_input_scale_zero_point",
    "load_layer_quant_params",
    "format_float_array_to_c",
    "prompt_layer_indices",
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
    "prompt_layout_selection",
    "collect_features_and_labels",
    "generate_test_inputs_fp32",
]
