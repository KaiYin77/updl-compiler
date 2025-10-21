#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UPH5 Format Specifications and Hardware Abstraction Layer
"""

from .uph5_format import (
    UPH5FormatSpec,
    UPH5LayerMetadata,
    UPH5QuantizationParams,
    ALIGNMENT_4_BYTE,
)

from .up301_hardware import (
    UP301HardwareSpec,
    INT16_RANGE,
    UDL_MAX_SHIFT,
    UDL_SAFETY_MARGIN,
)

__all__ = [
    # UPH5 Format
    "UPH5FormatSpec",
    "UPH5LayerMetadata",
    "UPH5QuantizationParams",
    "ALIGNMENT_4_BYTE",

    # Hardware Specs
    "UP301HardwareSpec",
    "INT16_RANGE",
    "UDL_MAX_SHIFT",
    "UDL_SAFETY_MARGIN",
]