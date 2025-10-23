#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Specifications Module

⚠️  HARDWARE CRITICAL: These specifications define quantization constraints
    and optimization parameters for embedded deployment.
    Modify with extreme care as changes affect hardware compatibility.
"""

from .up301 import (
    # Hardware constants
    INT16_RANGE,
    INT16_MIN,
    INT16_MAX,
    UDL_MAX_SHIFT,
    UDL_SAFETY_MARGIN,
    UDL_DEFAULT_MODE,

    # Hardware specification class
    UP301HardwareSpec,

    # Predefined configurations
    HardwareConfigs,

    # Utility functions
    validate_hardware_compatibility,
    optimize_scale_for_hardware,
)

__all__ = [
    # Constants
    "INT16_RANGE", "INT16_MIN", "INT16_MAX",
    "UDL_MAX_SHIFT", "UDL_SAFETY_MARGIN", "UDL_DEFAULT_MODE",

    # Classes
    "UP301HardwareSpec", "HardwareConfigs",

    # Functions
    "validate_hardware_compatibility", "optimize_scale_for_hardware",
]