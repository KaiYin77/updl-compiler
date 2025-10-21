#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Backward-compatible quantizer interface using new modular classes.
"""

from .quantization.config import QuantizationConfig
from .quantization.udl_quantizer import UDLQuantizer
from .quantization.parameter_calculator import (
    calculate_symmetric_quantization_params,
    calculate_udl_power_of_2_scale,
)
from .logger import log_debug, log_info


# Global instance for backward compatibility
_global_config = QuantizationConfig()
_global_quantizer = UDLQuantizer(_global_config)


def set_udl_shift_only_mode(enabled):
    """Set UDL shift-only quantization mode globally"""
    _global_config.udl_shift_only_mode = enabled
    log_info(f"UDL shift-only mode: {'ENABLED' if enabled else 'DISABLED'}")


def is_udl_shift_only_mode():
    """Check if UDL shift-only mode is enabled"""
    return _global_config.udl_shift_only_mode


def initialize_params(json_file, udl_mode=True):
    """Initialize quantization parameters from JSON file"""
    return _global_quantizer.initialize_params(json_file, udl_mode)


def load_params_json(json_file=None):
    """Load quantization parameters from JSON file (legacy interface)"""
    success = _global_config.load_params_from_json(json_file)
    return _global_config.params if success else None


def get_layer_params(layer_name, layer_type=None, layer_idx=None):
    """Get quantization parameters for a specific layer"""
    return _global_quantizer.get_layer_params(layer_name, layer_type, layer_idx)


def get_input_params():
    """Get input quantization parameters"""
    return _global_quantizer.get_input_params()


def calculate_params_symmetric(min_val, max_val):
    """Calculate symmetric quantization parameters (legacy interface)"""
    return calculate_symmetric_quantization_params(min_val, max_val, _global_config.udl_shift_only_mode)


def calculate_layer_activation(layer, layer_type, layer_idx, fused_activation=None):
    """Calculate quantization parameters for layer output based on activation type and layer characteristics

    Uses symmetric quantization (zero_point = 0) for embedded system efficiency.

    Args:
        layer: Keras layer instance
        layer_type: String type of the layer
        layer_idx: Index of the layer in the model
        fused_activation: Override activation for fused layers

    Returns:
        tuple: (scale, zero_point, min_val, max_val, activation_type)
    """
    # Get layer activation
    if fused_activation:
        activation = fused_activation
        log_info(f"Layer {layer_idx}: Using fused activation '{activation}'")
    else:
        activation = getattr(layer, 'activation', None)
        if hasattr(activation, '__name__'):
            activation = activation.__name__
        elif hasattr(activation, 'name'):
            activation = activation.name
        else:
            activation = str(activation)

    # Try to get parameters from JSON first
    layer_name = getattr(layer, 'name', f"layer_{layer_idx}")
    json_params = get_layer_params(layer_name, layer_type, layer_idx)

    if json_params:
        scale, zero_point = json_params

        # Calculate approximate min/max from scale and quantization range
        from .formats.up301_hardware import INT16_RANGE
        qmin, qmax = INT16_RANGE
        min_val = scale * qmin
        max_val = scale * qmax

        log_info(
            f"Layer {layer_idx}: Using JSON quantization params - "
            f"scale={scale:.8f}, zp={zero_point}, activation={activation}"
        )

        return scale, zero_point, min_val, max_val, activation

    # Default fallback values if no JSON params available
    log_info(
        f"Layer {layer_idx}: No JSON params found, using defaults for {layer_type} with {activation}"
    )

    # Use conservative defaults for different activation types
    if activation == 'relu':
        min_val, max_val = 0.0, 10.0
    elif activation == 'linear' or activation == 'softmax':
        min_val, max_val = -5.0, 5.0
    else:
        min_val, max_val = -1.0, 1.0

    scale, zero_point = calculate_params_symmetric(min_val, max_val)

    return scale, zero_point, min_val, max_val, activation


def calculate_weight_params(weights=None, layer_name=None, layer_type=None, layer_idx=None):
    """Calculate quantization parameters for layer weights"""
    return _global_quantizer.calculate_weight_params(weights, layer_type, layer_idx, layer_name)


def calculate_bias_params(bias=None, layer_name=None, layer_type=None, layer_idx=None):
    """Calculate quantization parameters for layer bias"""
    return _global_quantizer.calculate_bias_params(bias, layer_name, layer_type, layer_idx)