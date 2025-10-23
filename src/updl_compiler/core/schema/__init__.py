#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UPDL Schema - Centralized format specifications and validation.

⚠️  SCHEMA CRITICAL: These constants MUST match C enum definitions exactly.
    DO NOT MODIFY without corresponding changes to updl_interpreter.h
    Future development should be extremely careful with these definitions.
"""

from .uph5 import (
    # Binary format constants
    DESCRIPTION_LENGTH,
    TAG_LENGTH,
    STRING_LENGTH,
    ALIGNMENT_4_BYTE,

    # Schema enums - CRITICAL: Must match C enums exactly
    DTYPE_LIST,
    LTYPE_LIST,
    PTYPE_LIST,
    ATYPE_LIST,

    # Data structures
    UPH5QuantizationParams,
    UPH5LayerMetadata,
    UPH5FormatSpec,
    WeightLayoutSpec,
    UPH5Compatibility,
)

from .validation import (
    validate_layer_type,
    validate_activation_type,
    validate_padding_type,
    validate_dtype,
)

__all__ = [
    # Constants
    "DESCRIPTION_LENGTH", "TAG_LENGTH", "STRING_LENGTH", "ALIGNMENT_4_BYTE",

    # Schema enums
    "DTYPE_LIST", "LTYPE_LIST", "PTYPE_LIST", "ATYPE_LIST",

    # Data structures
    "UPH5QuantizationParams", "UPH5LayerMetadata", "UPH5FormatSpec",
    "WeightLayoutSpec", "UPH5Compatibility",

    # Validation
    "validate_layer_type", "validate_activation_type",
    "validate_padding_type", "validate_dtype",
]