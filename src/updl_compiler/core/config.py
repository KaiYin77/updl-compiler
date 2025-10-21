#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Legacy configuration module - now imports from centralized format specifications.
This module maintains backward compatibility while using the new centralized specs.
"""

from .formats.uph5_format import (
    DESCRIPTION_LENGTH,
    TAG_LENGTH,
    STRING_LENGTH,
    DTYPE_LIST,
    LTYPE_LIST,
    PTYPE_LIST,
    ATYPE_LIST,
)

# ============================================================================
# CENTRALIZED LAYER MANAGEMENT CONFIGURATION
# ============================================================================

# Layer categories for centralized management
LAYER_CONFIG = {
    # Supported layers that get serialized to UPH5
    "SUPPORTED": {
        "Conv1D",
        "Conv2D",
        "DepthwiseConv2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "Dense",
        "Flatten",
        "Lambda",
        "Softmax",
    },
    # Layers that are automatically skipped (not serialized)
    "SKIPPED": {
        "Dropout",
        "BatchNormalization",
    },  # Dropout not needed for inference, BatchNorm fused with Conv
    # Layers that can accept fused activations
    "FUSEABLE": {"Conv1D", "Conv2D", "Dense", "DepthwiseConv2D", "BatchNormalization"},
    # Layers that are used for activation fusion (will be skipped after fusion)
    "ACTIVATION": {"Activation"},
}

# ============================================================================
# CENTRALIZED LAYER MANAGEMENT FUNCTIONS
# ============================================================================


def is_layer_supported(layer_type):
    """Check if a layer type is supported for UPH5 serialization"""
    return layer_type in LAYER_CONFIG["SUPPORTED"]


def is_layer_skipped(layer_type):
    """Check if a layer type should be automatically skipped"""
    return layer_type in LAYER_CONFIG["SKIPPED"]


def is_layer_fuseable(layer_type):
    """Check if a layer type can accept fused activations"""
    return layer_type in LAYER_CONFIG["FUSEABLE"]


def is_activation_layer(layer_type):
    """Check if a layer type is an activation layer"""
    return layer_type in LAYER_CONFIG["ACTIVATION"]


def get_layer_category(layer_type):
    """Get the category of a layer type for logging purposes"""
    if is_layer_supported(layer_type):
        return "SUPPORTED"
    elif is_layer_skipped(layer_type):
        return "SKIPPED"
    elif is_activation_layer(layer_type):
        return "ACTIVATION"
    else:
        return "UNSUPPORTED"


def analyze_model_layers(model):
    """Analyze all layers in a model and categorize them"""
    layer_analysis = {
        "supported": [],
        "skipped": [],
        "activation": [],
        "unsupported": [],
    }

    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        category = get_layer_category(layer_type)

        layer_info = {
            "index": i,
            "name": layer.name,
            "type": layer_type,
            "layer": layer,
        }

        if category == "SUPPORTED":
            layer_analysis["supported"].append(layer_info)
        elif category == "SKIPPED":
            layer_analysis["skipped"].append(layer_info)
        elif category == "ACTIVATION":
            layer_analysis["activation"].append(layer_info)
        else:
            layer_analysis["unsupported"].append(layer_info)

    return layer_analysis


def validate_layer_configuration():
    """Validate that LTYPE_LIST matches SUPPORTED layers configuration"""
    from .logger import log_trace

    supported_set = LAYER_CONFIG["SUPPORTED"].copy()
    ltype_set = set(LTYPE_LIST)

    # Handle special mapping: BatchNormalization -> BatchNorm
    if "BatchNormalization" in supported_set:
        supported_set.remove("BatchNormalization")
        supported_set.add("BatchNorm")

    if supported_set != ltype_set:
        missing_in_ltype = supported_set - ltype_set
        extra_in_ltype = ltype_set - supported_set

        error_msg = "Layer configuration mismatch detected:\n"
        if missing_in_ltype:
            error_msg += f"  - Missing in LTYPE_LIST: {missing_in_ltype}\n"
        if extra_in_ltype:
            error_msg += f"  - Extra in LTYPE_LIST: {extra_in_ltype}\n"

        raise ValueError(error_msg)

    log_trace("Layer configuration validation passed")
