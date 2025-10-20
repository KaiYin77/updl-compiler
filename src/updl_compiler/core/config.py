#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

# Constants for file format
DESCRIPTION_LENGTH = 32
TAG_LENGTH = 16
STRING_LENGTH = 16

# Type definitions (matching C implementation)
DTYPE_LIST = [
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "bool",
    "char",
    "dtype_t",
    "ltype_t",
    "ptype_t",
    "atype_t",
]

# CRITICAL: This order MUST match the C enum ltype_t in updl_interpreter.h exactly
LTYPE_LIST = [
    "Conv1D",  # 0: Ltype_conv_1d
    "Conv2D",  # 1: Ltype_conv_2d
    "DepthwiseConv2D",  # 2: Ltype_depthwise_conv_2d
    "MaxPooling2D",  # 3: Ltype_max_pooling_2d
    "AveragePooling2D",  # 4: Ltype_average_pooling_2d
    "Dense",  # 5: Ltype_dense
    "Flatten",  # 6: Ltype_flatten
    "Lambda",  # 7: Ltype_lambda
    "Softmax",  # 9: Ltype_softmax
]

# CRITICAL: This order MUST match the C enum ptype_t in updl_interpreter.h exactly
# Ptype_valid=0, Ptype_same=1
PTYPE_LIST = ["valid", "same"]  # 0: Ptype_valid  # 1: Ptype_same

# CRITICAL: This order MUST match the C enum atype_t in updl_interpreter.h exactly
# Atype_none=0, Atype_linear=1, ..., Atype_tanh=6
ATYPE_LIST = [
    "none",  # 0: Atype_none
    "linear",  # 1: Atype_linear
    "relu",  # 2: Atype_relu
    "leakyrelu",  # 3: Atype_leakyrelu
    "softmax",  # 4: Atype_softmax
    "sigmoid",  # 5: Atype_sigmoid
    "tanh",  # 6: Atype_tanh
]

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
