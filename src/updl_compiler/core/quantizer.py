#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import os
import json
import math
from .logger import log_debug, log_info

# Global variable to store loaded quantization parameters
_quantization_params = None

# Global variable to control quantization mode
_udl_shift_only_mode = False


def load_params_json(json_file=None):
    """Load quantization parameters from JSON file

    Args:
        json_file: Path to JSON file with quantization parameters.
                  If None, searches for 'quantization_params_int16.json' in current directory.

    Returns:
        dict: Loaded quantization parameters or None if file not found
    """
    global _quantization_params

    if json_file is None:
        json_file = "quantization_params_int16.json"

    if not os.path.exists(json_file):
        log_debug(
            f"Quantization params file not found: {json_file}, using default parameters"
        )
        return None

    try:
        with open(json_file, "r") as f:
            _quantization_params = json.load(f)
        log_debug(f"Loaded quantization parameters from: {json_file}")
        return _quantization_params
    except Exception as e:
        log_debug(f"Failed to load quantization parameters from {json_file}: {e}")
        return None


def set_udl_shift_only_mode(enabled=True):
    """Enable or disable UDL shift-only quantization mode
    
    In UDL shift-only mode, all quantization scales are forced to be power-of-2
    values to match UDL hardware constraints that only support shift operations.
    
    Args:
        enabled: True to enable UDL shift-only mode, False for standard TensorFlow Lite mode
    """
    global _udl_shift_only_mode
    _udl_shift_only_mode = enabled
    log_info(f"UDL shift-only mode: {'ENABLED' if enabled else 'DISABLED'}")


def is_udl_shift_only_mode():
    """Check if UDL shift-only quantization mode is enabled
    
    Returns:
        bool: True if UDL shift-only mode is enabled
    """
    global _udl_shift_only_mode
    return _udl_shift_only_mode


def initialize_params(json_file=None, udl_mode=False):
    """Initialize quantization parameters by loading from JSON file

    This should be called before starting the quantization process.

    Args:
        json_file: Path to JSON file with quantization parameters.
                  If None, searches for 'quantization_params_int16.json' in current directory.
        udl_mode: Enable UDL shift-only quantization mode

    Returns:
        bool: True if parameters loaded successfully, False otherwise
    """
    # Set UDL mode first
    set_udl_shift_only_mode(udl_mode)
    
    params = load_params_json(json_file)
    return params is not None


def get_layer_params(layer_name, layer_type, layer_idx):
    """Get quantization parameters for a layer from loaded JSON data

    Args:
        layer_name: Name of the layer
        layer_type: Type of the layer
        layer_idx: Index of the layer

    Returns:
        tuple: (scale, zero_point) or None if not found
    """
    global _quantization_params

    if _quantization_params is None:
        return None

    layers = _quantization_params.get("layers", {})

    # Try to find layer by name first
    if layer_name in layers:
        layer_data = layers[layer_name]
        scale = layer_data.get("scale")
        zero_point = layer_data.get("zero_point", 0)
        log_debug(
            f"Found JSON params for {layer_name}: scale={scale:.8f}, zp={zero_point}"
        )
        return scale, zero_point

    # Try to find by layer type and index if name not found
    for name, layer_data in layers.items():
        if (
            layer_data.get("layer_type") == layer_type
            and layer_data.get("layer_index") == layer_idx
        ):
            scale = layer_data.get("scale")
            zero_point = layer_data.get("zero_point", 0)
            log_debug(
                f"Found JSON params for {layer_type}[{layer_idx}]: scale={scale:.8f}, zp={zero_point}"
            )
            return scale, zero_point

    return None


def get_input_params():
    """Get input quantization parameters from loaded JSON data

    Returns:
        tuple: (scale, zero_point) or None if not found
    """
    global _quantization_params

    if _quantization_params is None:
        return None

    input_data = _quantization_params.get("input", {})
    if input_data:
        scale = input_data.get("scale")
        zero_point = input_data.get("zero_point", 0)
        log_debug(f"Found JSON input params: scale={scale:.8f}, zp={zero_point}")
        return scale, zero_point

    return None


def get_weight_params(layer_name=None, layer_type=None, layer_idx=None):
    """Get weight quantization parameters from loaded JSON data

    Args:
        layer_name: Name of the layer (optional)
        layer_type: Type of the layer (optional)
        layer_idx: Index of the layer (optional)

    Returns:
        tuple: (scale, zero_point) or None if not found

    Note: Currently weight params are not stored in JSON, but this function
          provides symmetric API for potential future use.
    """
    global _quantization_params

    if _quantization_params is None:
        return None

    # Check if there's a separate weights section in JSON (future extensibility)
    weights_data = _quantization_params.get("weights", {})
    if weights_data and layer_name:
        weight_data = weights_data.get(layer_name, {})
        if weight_data:
            scale = weight_data.get("scale")
            zero_point = weight_data.get("zero_point", 0)
            log_debug(
                f"Found JSON weight params for {layer_name}: scale={scale:.8f}, zp={zero_point}"
            )
            return scale, zero_point

    # Currently no weight params stored separately in JSON
    return None


def calculate_udl_power_of_2_scale(original_scale, max_shift=15, max_abs_value=None):
    """Convert arbitrary scale to nearest power-of-2 for UDL hardware
    
    Uses improved error minimization algorithm with overflow protection that matches C implementation:
    - Tries both floor and ceil approximations
    - Chooses the option with smaller absolute relative error
    - Prevents int16 overflow by checking quantized values
    - Ensures consistency with updl_kernels_support.h
    
    Args:
        original_scale: Original floating-point scale
        max_shift: Maximum allowed shift value (typically 15 for UDL hardware)
        max_abs_value: Maximum absolute value to be quantized (for overflow checking)
        
    Returns:
        tuple: (power_of_2_scale, shift_value, absolute_relative_error)
    """
    if original_scale <= 0:
        return 1.0, 0, 1.0
    
    # Helper function to check for int16 overflow
    def check_overflow(scale, max_val):
        if max_val is None:
            return False
        quantized = max_val / scale
        return abs(quantized) > 32767
    
    # Improved power-of-2 approximation with error minimization (matches C implementation)
    log2_scale = math.log2(original_scale)
    
    # Try both floor and ceil to find better approximation
    shift_floor = int(math.floor(-log2_scale))
    shift_ceil = int(math.ceil(-log2_scale))
    
    # Calculate scales for both options
    if shift_floor >= 0:
        scale_floor = 1.0 / (1 << shift_floor)  # Right shift: scale = 1/2^shift
    else:
        scale_floor = 1 << (-shift_floor)       # Left shift: scale = 2^(-shift)
        
    if shift_ceil >= 0:
        scale_ceil = 1.0 / (1 << shift_ceil)   # Right shift: scale = 1/2^shift  
    else:
        scale_ceil = 1 << (-shift_ceil)        # Left shift: scale = 2^(-shift)
    
    # Calculate absolute relative errors for both options (matches C implementation)
    error_floor = abs(scale_floor - original_scale) / original_scale
    error_ceil = abs(scale_ceil - original_scale) / original_scale
    
    # Check for overflow and choose the option with smaller error (matches C implementation with overflow protection)
    floor_overflow = check_overflow(scale_floor, max_abs_value)
    ceil_overflow = check_overflow(scale_ceil, max_abs_value)
    
    # If both would overflow, use original scale and warn
    if floor_overflow and ceil_overflow:
        log_info(f"WARNING: Both power-of-2 approximations would cause int16 overflow. Using original scale.")
        return original_scale, 0, 0.0
        
    # If one would overflow, use the other
    elif floor_overflow:
        calculated_shift = shift_ceil
        power_of_2_scale = scale_ceil
        absolute_relative_error = error_ceil
    elif ceil_overflow:
        calculated_shift = shift_floor
        power_of_2_scale = scale_floor
        absolute_relative_error = error_floor
    # If neither overflows, choose the option with smaller error
    elif error_floor < error_ceil:
        calculated_shift = shift_floor
        power_of_2_scale = scale_floor
        absolute_relative_error = error_floor
    else:
        calculated_shift = shift_ceil
        power_of_2_scale = scale_ceil
        absolute_relative_error = error_ceil
    
    # Clamp shift to UDL hardware limits (matches C implementation constants)
    UDL_MIN_SHIFT = -max_shift
    UDL_MAX_SHIFT = max_shift
    
    if calculated_shift < UDL_MIN_SHIFT:
        final_shift = UDL_MIN_SHIFT
    elif calculated_shift > UDL_MAX_SHIFT:
        final_shift = UDL_MAX_SHIFT
    else:
        final_shift = calculated_shift
    
    # Recalculate final scale if shift was clamped
    if final_shift != calculated_shift:
        if final_shift >= 0:
            power_of_2_scale = 1.0 / (1 << final_shift)
        else:
            power_of_2_scale = 1 << (-final_shift)
        # Recalculate error for clamped shift
        absolute_relative_error = abs(power_of_2_scale - original_scale) / original_scale
    
    return power_of_2_scale, final_shift, absolute_relative_error


def calculate_params_symmetric(min_val, max_val):
    """Calculate symmetric quantization parameters for int16 range

    Args:
        min_val: Minimum expected float value
        max_val: Maximum expected float value

    Returns:
        tuple: (scale, zero_point) for symmetric quantization
        Note: zero_point is always 0 for symmetric quantization
    """
    import numpy as np

    # int16 range: [-32768, 32767]
    qmin, qmax = -32768, 32767

    # Handle edge cases
    if min_val == max_val:
        return 1.0 / qmax, 0

    # For symmetric quantization, use the maximum absolute value
    max_abs = max(abs(min_val), abs(max_val))

    # Calculate scale based on symmetric range
    scale = (2.0 * max_abs) / (qmax - qmin)

    # Zero point is always 0 for symmetric quantization
    zero_point = 0
    
    # If UDL shift-only mode is enabled, convert to power-of-2 scale
    if is_udl_shift_only_mode():
        original_scale = scale
        power_of_2_scale, shift, absolute_relative_error = calculate_udl_power_of_2_scale(original_scale, max_abs_value=max_abs)
        
        # Debug logging that matches C implementation format
        log_info(f"UDL Power-of-2 conversion: {original_scale:.8f} -> {power_of_2_scale:.8f} "
                 f"(shift={shift}, abs_rel_error={absolute_relative_error:.6f})")
        
        # Warn if approximation error is significant (>10% error)
        if absolute_relative_error > 0.1:
            log_info(f"WARNING: Significant power-of-2 approximation error: "
                    f"{original_scale:.8f} -> {power_of_2_scale:.8f} "
                    f"(abs_rel_error={absolute_relative_error:.6f})")
        
        scale = power_of_2_scale

    return scale, zero_point


def calculate_layer_activation(layer, layer_type, layer_idx, fused_activation=None):
    """Calculate quantization parameters for layer output based on activation type and layer characteristics

    Uses symmetric quantization (zero_point = 0) for embedded system efficiency.
    This simplifies computations and reduces memory requirements.

    Args:
        layer: The Keras layer object
        layer_type: Layer type string
        layer_idx: Layer index
        fused_activation: Override activation if fusion was applied (e.g., "relu" for fused Conv2D+ReLU)
    """

    # First, try to get parameters from JSON file if available
    layer_name = getattr(layer, "name", f"{layer_type}_{layer_idx}")
    print(layer_name)
    json_params = get_layer_params(layer_name, layer_type, layer_idx)

    if json_params is not None:
        scale, zero_point = json_params
        log_info(
            f"Layer {layer_idx} ({layer_type}): Using JSON parameters, "
            f"act_scale={scale:.8f}, act_zp={zero_point} (from analysis)"
        )
        return scale, zero_point

    # Default values for symmetric quantization
    # We'll calculate optimal scale based on expected activation ranges

    # Pass-through layers (inherit input quantization) - but still no JSON data available
    if layer_type in ["MaxPooling2D", "AveragePooling2D", "Flatten", "Lambda"]:
        log_info(
            f"Layer {layer_idx} ({layer_type}): Pass-through layer with no JSON data, "
            f"using conservative default range"
        )
        # Use conservative symmetric range for pass-through layers when no JSON data
        min_val, max_val = -16.0, 16.0
        act_scale, act_zp = calculate_params_symmetric(min_val, max_val)
        log_info(
            f"Layer {layer_idx} ({layer_type}): FALLBACK pass-through range=[{min_val:.1f}, {max_val:.1f}], "
            f"act_scale={act_scale:.8f}, act_zp={act_zp} (conservative default)"
        )
        return act_scale, act_zp

    # Get activation function - prioritize fused activation over layer's original activation
    if fused_activation is not None:
        activation = fused_activation
        log_info(f"Layer {layer_idx}: Using fused activation '{activation}'")
    else:
        activation = "linear"  # default
        if hasattr(layer, "get_config"):
            try:
                activation = layer.get_config().get("activation", "linear")
            except:
                pass

    # Fall back to conservative default range when JSON data is not available
    log_info(
        f"Layer {layer_idx} ({layer_type}, {activation}): No JSON data available, "
        f"using conservative default range for symmetric quantization"
    )

    # Use a very conservative symmetric range that should accommodate most layer outputs
    # This is much safer than hardcoded layer-specific ranges which may not match actual data
    min_val, max_val = -32.0, 32.0

    # Calculate symmetric quantization parameters
    act_scale, act_zp = calculate_params_symmetric(min_val, max_val)

    log_info(
        f"Layer {layer_idx} ({layer_type}, {activation}): FALLBACK range=[{min_val:.1f}, {max_val:.1f}], "
        f"act_scale={act_scale:.8f}, act_zp={act_zp} (conservative default)"
    )
    return act_scale, act_zp


def calculate_weight_params(
    weights=None, layer_name=None, layer_type=None, layer_idx=None
):
    """Get optimal symmetric weight quantization parameters
    
    In UDL shift-only mode, ensures weight scales are power-of-2 values.
    """
    import numpy as np

    if weights is not None:
        # Calculate optimal symmetric quantization based on actual weight distribution
        min_val  = np.min(weights)
        max_val = np.max(weights)

        # Ensure minimum range to avoid division by zero
        if max_val - min_val < 0.01:
            center = (max_val + min_val) / 2
            min_val = center - 0.005
            max_val = center + 0.005

        # Calculate symmetric quantization parameters (UDL mode handled in calculate_params_symmetric)
        weight_scale, weight_zp = calculate_params_symmetric(min_val, max_val)
        
        if is_udl_shift_only_mode():
            log_debug(f"Weight quantization (UDL mode): layer {layer_idx} ({layer_type}), "
                     f"range=[{min_val:.6f}, {max_val:.6f}], scale={weight_scale:.8f}")

        return weight_scale, weight_zp
    else:
        # Default for layers without weights (pooling, flatten, etc.)
        # Conservative symmetric range that accommodates most weight distributions
        weight_scale, weight_zp = calculate_params_symmetric(-4.0, 4.0)
        return weight_scale, weight_zp


def calculate_bias_params(
    bias=None, layer_name=None, layer_type=None, layer_idx=None
):
    """Get optimal symmetric bias quantization parameters based on bias range
    
    In UDL shift-only mode, ensures bias scales are power-of-2 values.
    """
    import numpy as np
    
    if bias is not None:
        # Calculate optimal symmetric quantization based on actual bias distribution
        min_val = np.min(bias)
        max_val = np.max(bias)
        
        # Ensure minimum range to avoid division by zero
        if max_val - min_val < 1e-8:
            center = (max_val + min_val) / 2
            min_val = center - 5e-9
            max_val = center + 5e-9
            
        # Calculate symmetric quantization parameters for bias (UDL mode handled in calculate_params_symmetric)
        bias_scale, bias_zp = calculate_params_symmetric(min_val, max_val)
        
        if is_udl_shift_only_mode():
            log_debug(f"Bias quantization (UDL mode): layer {layer_idx} ({layer_type}), "
                     f"range=[{min_val:.6f}, {max_val:.6f}], scale={bias_scale:.8f}")
         
        return bias_scale, bias_zp
    
    else:
        bias_scale, bias_zp = calculate_params_symmetric(-4.0, 4.0)
        return bias_scale, bias_zp

def get_udl_shift_value(scale):
    """Get the shift value that would be used for a given scale in UDL mode
    
    Args:
        scale: Floating-point scale value
        
    Returns:
        int: Shift value that UDL hardware would use
    """
    if scale <= 0:
        return 0
    
    log2_scale = math.log2(scale)
    shift = int(round(-log2_scale))
    
    # Clamp to UDL hardware limits
    return max(-15, min(15, shift))
