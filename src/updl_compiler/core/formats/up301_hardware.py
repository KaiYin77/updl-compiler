#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UP301/UP201 Hardware Specifications

This module centralizes all hardware-specific constants for UP301 and UP201
embedded AI accelerators to ensure consistent quantization and optimization.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# === Hardware Quantization Constants ===
INT16_RANGE: Tuple[int, int] = (-32768, 32767)
INT16_MIN, INT16_MAX = INT16_RANGE

# UDL (Upbeat Deep Learning) specific constants
UDL_MAX_SHIFT = 15  # Maximum bit shift for power-of-2 scaling
UDL_SAFETY_MARGIN = 0.9  # Safety margin for int16 overflow prevention
UDL_DEFAULT_MODE = True  # Enable UDL mode by default


@dataclass
class UP301HardwareSpec:
    """Hardware specifications for UP301/UP201 embedded AI accelerators"""

    # Quantization specifications
    quantization_dtype: str = "int16_t"
    quantization_range: Tuple[int, int] = INT16_RANGE
    zero_point_symmetric: bool = True  # Symmetric quantization (zero_point = 0)

    # UDL optimization settings
    udl_mode_enabled: bool = UDL_DEFAULT_MODE
    udl_power_of_2_scales: bool = True  # Prefer power-of-2 scales for efficiency
    udl_max_shift_bits: int = UDL_MAX_SHIFT
    udl_safety_margin: float = UDL_SAFETY_MARGIN

    # Memory alignment requirements
    weight_alignment_bytes: int = 4  # 4-byte alignment for weight data
    data_alignment_bytes: int = 4   # 4-byte alignment for all data

    # Layer support matrix
    supported_layers: Tuple[str, ...] = (
        "Conv2D", "Conv1D", "DepthwiseConv2D", "SeparableConv2D",
        "Dense", "Flatten", "MaxPooling2D", "AveragePooling2D",
        "GlobalMaxPooling2D", "GlobalAveragePooling2D",
        "BatchNormalization", "Dropout", "Softmax", "Activation"
    )

    # Supported activations
    supported_activations: Tuple[str, ...] = (
        "linear", "relu", "sigmoid", "tanh", "softmax", "elu", "selu",
        "softplus", "softsign", "relu6", "leaky_relu", "prelu", "exponential"
    )

    def validate_quantization_range(self, min_val: float, max_val: float) -> bool:
        """Validate that quantization range fits in hardware specs"""
        if self.zero_point_symmetric:
            # For symmetric quantization, check the larger absolute value
            max_abs = max(abs(min_val), abs(max_val))
            return max_abs <= INT16_MAX
        else:
            # For asymmetric quantization, check both bounds
            return INT16_MIN <= min_val and max_val <= INT16_MAX

    def calculate_optimal_scale(self, min_val: float, max_val: float) -> float:
        """Calculate optimal quantization scale for hardware"""
        if self.zero_point_symmetric:
            max_abs = max(abs(min_val), abs(max_val))
            return max_abs / INT16_MAX
        else:
            value_range = max_val - min_val
            return value_range / (INT16_MAX - INT16_MIN)

    def is_power_of_2_scale(self, scale: float, tolerance: float = 1e-6) -> bool:
        """Check if scale is approximately a power of 2"""
        if scale <= 0:
            return False

        # Find the closest power of 2
        log2_scale = np.log2(scale)
        closest_power = round(log2_scale)
        closest_scale = 2 ** closest_power

        return abs(scale - closest_scale) < tolerance

    def get_power_of_2_approximation(self, scale: float) -> Tuple[float, int]:
        """Get the closest power-of-2 approximation and corresponding shift"""
        if scale <= 0:
            return scale, 0

        log2_scale = np.log2(scale)
        shift = round(log2_scale)

        # Clamp shift to valid range
        shift = max(-self.udl_max_shift_bits, min(self.udl_max_shift_bits, shift))

        power_of_2_scale = 2 ** shift
        return power_of_2_scale, shift

    def check_overflow_risk(self, values: np.ndarray, scale: float) -> bool:
        """Check if quantizing with given scale would cause int16 overflow"""
        if len(values) == 0:
            return False

        # Symmetric quantization: quantized = value / scale
        quantized = values / scale
        return np.any(quantized < INT16_MIN) or np.any(quantized > INT16_MAX)

    def get_safe_scale(self, values: np.ndarray) -> float:
        """Calculate a safe scale that prevents overflow"""
        if len(values) == 0:
            return 1.0

        max_abs = np.max(np.abs(values))
        safe_scale = max_abs / (INT16_MAX * self.udl_safety_margin)
        return max(safe_scale, 1e-8)  # Prevent division by zero


# === Predefined Hardware Configurations ===

class HardwareConfigs:
    """Predefined hardware configurations for different deployment targets"""

    @staticmethod
    def up301_config() -> UP301HardwareSpec:
        """Standard UP301 configuration"""
        return UP301HardwareSpec()

    @staticmethod
    def up201_config() -> UP301HardwareSpec:
        """UP201 configuration (same as UP301 for now)"""
        return UP301HardwareSpec()

    @staticmethod
    def development_config() -> UP301HardwareSpec:
        """Development configuration with relaxed constraints"""
        return UP301HardwareSpec(
            udl_safety_margin=0.95,  # Higher safety margin
            udl_max_shift_bits=12,   # More conservative shift range
        )

    @staticmethod
    def production_config() -> UP301HardwareSpec:
        """Production configuration with optimal performance"""
        return UP301HardwareSpec(
            udl_safety_margin=0.85,  # Lower safety margin for better precision
            udl_max_shift_bits=15,   # Full shift range
        )


# === Hardware Utility Functions ===

def validate_hardware_compatibility(values: np.ndarray,
                                   scale: float,
                                   zero_point: int = 0,
                                   hardware_spec: UP301HardwareSpec = None) -> bool:
    """Validate that quantization parameters are compatible with hardware"""
    if hardware_spec is None:
        hardware_spec = UP301HardwareSpec()

    # Check if quantized values fit in int16 range
    quantized = ((values - zero_point) / scale).astype(np.int32)

    return (np.all(quantized >= INT16_MIN) and
            np.all(quantized <= INT16_MAX))


def optimize_scale_for_hardware(values: np.ndarray,
                               target_scale: float,
                               hardware_spec: UP301HardwareSpec = None) -> Tuple[float, bool]:
    """Optimize scale for hardware efficiency"""
    if hardware_spec is None:
        hardware_spec = UP301HardwareSpec()

    # If UDL mode is enabled, try to find power-of-2 approximation
    if hardware_spec.udl_mode_enabled and hardware_spec.udl_power_of_2_scales:
        power_of_2_scale, shift = hardware_spec.get_power_of_2_approximation(target_scale)

        # Check if power-of-2 scale would cause overflow
        if not hardware_spec.check_overflow_risk(values, power_of_2_scale):
            return power_of_2_scale, True

    # Fall back to original scale if power-of-2 doesn't work
    safe_scale = hardware_spec.get_safe_scale(values)
    final_scale = max(target_scale, safe_scale)

    return final_scale, False