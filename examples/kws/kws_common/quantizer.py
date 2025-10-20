#!/usr/bin/env python
"""
Quantization utilities for KWS models

Contains functions for calculating quantization parameters for int16 neural networks.
Supports both TensorFlow Lite and UDL power-of-2 quantization modes.
"""

import math


def calculate_udl_power_of_2_scale(original_scale, max_shift=15, max_abs_value=None):
    """Convert arbitrary scale to nearest power-of-2 for UDL hardware
    
    Uses improved error minimization algorithm with overflow protection:
    - Tries both floor and ceil approximations
    - Chooses the option with smaller absolute relative error
    - Prevents int16 overflow by checking quantized values
    - Ensures consistency with C implementation in updl_kernels_support.h
    
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
    
    # Improved power-of-2 approximation with error minimization
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
    
    # Calculate absolute relative errors for both options
    error_floor = abs(scale_floor - original_scale) / original_scale
    error_ceil = abs(scale_ceil - original_scale) / original_scale
    
    # Check for overflow and choose the option with smaller error
    floor_overflow = check_overflow(scale_floor, max_abs_value)
    ceil_overflow = check_overflow(scale_ceil, max_abs_value)
    
    # If both would overflow, use original scale and warn
    if floor_overflow and ceil_overflow:
        print(f"WARNING: Both power-of-2 approximations would cause int16 overflow. Using original scale.")
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
    
    # Clamp shift to UDL hardware limits
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


def calculate_symmetric_quantization_params(min_val, max_val, udl_mode=True):
    """Calculate symmetric quantization parameters for int16 range

    Args:
        min_val: Minimum expected float value
        max_val: Maximum expected float value
        udl_mode: If True, force scales to be power-of-2 for UDL hardware compatibility

    Returns:
        tuple: (scale, zero_point) for symmetric quantization
        Note: zero_point is always 0 for symmetric quantization
    """
    # int16 range: [-32768, 32767]
    qmin, qmax = -32768, 32767

    # Handle edge cases
    if min_val == max_val:
        base_scale = 1.0 / qmax
        if udl_mode:
            power_of_2_scale, _, _ = calculate_udl_power_of_2_scale(base_scale, max_abs_value=abs(min_val))
            return power_of_2_scale, 0
        return base_scale, 0

    # For symmetric quantization, use the maximum absolute value
    max_abs = max(abs(min_val), abs(max_val))

    # Calculate scale based on symmetric range
    scale = (2.0 * max_abs) / (qmax - qmin)

    # Zero point is always 0 for symmetric quantization
    zero_point = 0
    
    # If UDL mode is enabled, convert to power-of-2 scale
    if udl_mode:
        original_scale = scale
        power_of_2_scale, shift, absolute_relative_error = calculate_udl_power_of_2_scale(original_scale, max_abs_value=max_abs)
        
        # Warn if approximation error is significant (>10% error)
        if absolute_relative_error > 0.1:
            print(f"WARNING: Significant power-of-2 approximation error: "
                  f"{original_scale:.8f} -> {power_of_2_scale:.8f} "
                  f"(abs_rel_error={absolute_relative_error:.6f})")
        
        return power_of_2_scale, zero_point

    return scale, zero_point
