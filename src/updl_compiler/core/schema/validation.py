#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Schema Validation Functions

⚠️  SCHEMA SENSITIVE: These validation functions ensure compatibility
    with C runtime definitions. Modify with extreme care.
"""

from .uph5 import DTYPE_LIST, LTYPE_LIST, PTYPE_LIST, ATYPE_LIST


def validate_layer_type(layer_type: str) -> bool:
    """
    ⚠️  SCHEMA SENSITIVE: Validate layer type against LTYPE_LIST

    Args:
        layer_type: Layer type string to validate

    Returns:
        bool: True if layer type is supported
    """
    # Handle BatchNormalization mapping
    if layer_type == "BatchNormalization":
        layer_type = "BatchNorm"

    return layer_type in LTYPE_LIST


def validate_activation_type(activation: str) -> bool:
    """
    ⚠️  SCHEMA SENSITIVE: Validate activation type against ATYPE_LIST

    Args:
        activation: Activation type string to validate

    Returns:
        bool: True if activation type is supported
    """
    return activation in ATYPE_LIST


def validate_padding_type(padding: str) -> bool:
    """
    ⚠️  SCHEMA SENSITIVE: Validate padding type against PTYPE_LIST

    Args:
        padding: Padding type string to validate

    Returns:
        bool: True if padding type is supported
    """
    return padding in PTYPE_LIST


def validate_dtype(dtype: str) -> bool:
    """
    ⚠️  SCHEMA SENSITIVE: Validate data type against DTYPE_LIST

    Args:
        dtype: Data type string to validate

    Returns:
        bool: True if data type is supported
    """
    return dtype in DTYPE_LIST


def get_layer_type_index(layer_type: str) -> int:
    """
    ⚠️  SCHEMA CRITICAL: Get index of layer type in LTYPE_LIST
    This index MUST match the C enum values exactly.

    Args:
        layer_type: Layer type string

    Returns:
        int: Index in LTYPE_LIST, or -1 if not found
    """
    # Handle BatchNormalization mapping
    if layer_type == "BatchNormalization":
        layer_type = "BatchNorm"

    try:
        return LTYPE_LIST.index(layer_type)
    except ValueError:
        return -1


def get_activation_type_index(activation: str) -> int:
    """
    ⚠️  SCHEMA CRITICAL: Get index of activation type in ATYPE_LIST
    This index MUST match the C enum values exactly.

    Args:
        activation: Activation type string

    Returns:
        int: Index in ATYPE_LIST, or -1 if not found
    """
    try:
        return ATYPE_LIST.index(activation)
    except ValueError:
        return -1


def get_padding_type_index(padding: str) -> int:
    """
    ⚠️  SCHEMA CRITICAL: Get index of padding type in PTYPE_LIST
    This index MUST match the C enum values exactly.

    Args:
        padding: Padding type string

    Returns:
        int: Index in PTYPE_LIST, or -1 if not found
    """
    try:
        return PTYPE_LIST.index(padding)
    except ValueError:
        return -1