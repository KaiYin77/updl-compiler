#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import struct
import numpy as np
from .logger import log_error, log_debug, log_trace, log_info
from .formats.uph5_format import (
    TAG_LENGTH,
    STRING_LENGTH,
    ALIGNMENT_4_BYTE,
    WeightLayoutSpec,
)
from .formats.up301_hardware import INT16_RANGE


def write_string(f, string, length=STRING_LENGTH, debug=False):
    """Write a fixed-length string with null padding"""
    # Truncate if needed
    string_trimmed = string[:length]

    # Calculate padding
    padding_length = length - len(string_trimmed)

    # Create result with explicit null padding
    result = bytearray(string_trimmed.encode("ascii"))
    result.extend(bytearray(padding_length))  # This creates zero bytes

    # Ensure exactly the right length
    if len(result) != length:
        log_error(f"Serializer: String padding failed for '{string_trimmed}' - expected {length} bytes but generated {len(result)} bytes. This indicates a binary format specification error.")
        raise ValueError(f"Binary format error: string padding length mismatch")

    # Write to file
    f.write(result)

    log_trace(
        f"String: '{string_trimmed}' padded to {length} bytes (pos: {f.tell() - length})"
    )

    return result


def write_tag(f, tag, debug=False):
    """Write a tag (16 bytes)"""
    result = write_string(f, tag, TAG_LENGTH, debug)
    log_trace(f"Tag: '{tag}' (pos: {f.tell() - TAG_LENGTH})")
    return result


def write_uint16(f, value, debug=False):
    """Write a 2-byte unsigned integer"""
    result = value.to_bytes(2, byteorder="little", signed=False)
    f.write(result)
    log_trace(f"uint16: {value} (pos: {f.tell() - 2})")
    return result


def write_int32(f, value, debug=False):
    """Write a 4-byte signed integer"""
    result = value.to_bytes(4, byteorder="little", signed=True)
    f.write(result)
    log_trace(f"int32: {value} (pos: {f.tell() - 4})")
    return result


def write_int16(f, value, debug=False):
    """Write a 2-byte signed integer"""
    result = value.to_bytes(2, byteorder="little", signed=True)
    f.write(result)
    log_trace(f"int16: {value} (pos: {f.tell() - 2})")
    return result


def write_float32(f, value, debug=False):
    """Write a 4-byte float"""
    result = struct.pack("<f", value)  # little-endian float
    f.write(result)
    log_trace(f"float32: {value} (pos: {f.tell() - 4})")
    return result


def write_shape(f, shape, dim_count=4, debug=False):
    """Write a shape (array of uint16)"""
    for i in range(dim_count):
        # Default value for missing dimensions
        dim_value = 0
        if i < len(shape) and shape[i] is not None:
            dim_value = int(shape[i])  # Ensure integer conversion
        write_uint16(f, dim_value, debug)

    log_trace(f"Shape: {shape} (as {dim_count} dimensions)")


def write_enum(f, string_value, enum_list, debug=False):
    """Write a type enum (lookup string in list)"""
    if string_value not in enum_list:
        log_error(f"Serializer: Unknown layer type '{string_value}' encountered during serialization. Valid types are: {enum_list}. This indicates an unsupported layer type.")
        enum_value = -1
    else:
        enum_value = enum_list.index(string_value)

    write_int32(f, enum_value, debug)
    log_trace(f"Enum: {string_value} -> {enum_value}")
    return enum_value


def write_quantization_parameters(
    f,
    act_scale,
    act_zp,
    weight_scale,
    weight_zp,
    bias_scale,
    bias_zp,
    debug=False,
):
    """Write quantization parameters to UPH5 file with separate weight, activation, and bias parameters"""

    # Write activation quantization parameters
    write_tag(f, "act_scale", debug)
    write_float32(f, act_scale, debug)
    write_tag(f, "act_zp", debug)
    write_int16(f, act_zp, debug)

    # Write weight quantization parameters
    write_tag(f, "weight_scale", debug)
    write_float32(f, weight_scale, debug)
    write_tag(f, "weight_zp", debug)
    write_int16(f, weight_zp, debug)

    # Write bias quantization parameters
    write_tag(f, "bias_scale", debug)
    write_float32(f, bias_scale, debug)
    write_tag(f, "bias_zp", debug)
    write_int16(f, bias_zp, debug)

    if debug:
        log_debug(
            f"Wrote quantization parameters: act_scale={act_scale:.6f}, weight_scale={weight_scale:.6f}"
        )
        log_debug(f"  Activation: scale={act_scale:.8f}, zp={act_zp}")
        log_debug(f"  Weight: scale={weight_scale:.8f}, zp={weight_zp}")
        log_debug(f"  Bias: scale={bias_scale:.8f}, zp={bias_zp}")


def scale_to_16(in_matrix, safety_margin=0.9):
    """Calculate shift factor to scale matrix values to int16 range with safety margin"""
    abs_val = np.abs(in_matrix)
    max_value = np.max(abs_val)
    INT16_MIN, INT16_MAX = INT16_RANGE
    max_allowed = INT16_MAX * safety_margin  # Add safety margin
    shift = 0

    while max_value * 2 < max_allowed and shift < 14:  # Add upper limit to shift
        max_value *= 2
        shift += 1

    log_debug(
        f"Matrix scaling: original_max={np.max(abs_val)}, scaled_max={max_value}, shift={shift}"
    )

    return shift


def optimize_weights_layout(weights, layer_type, weight_name):
    """
    Unified weight layout optimization for embedded C runtime performance.
    Uses centralized WeightLayoutSpec for consistent optimization rules.

    Args:
        weights: Input weight tensor/array
        layer_type: Type of layer ("Conv2D", "DepthwiseConv2D", "Dense", etc.)
        weight_name: Name of weight ("weight", "bias")

    Returns:
        Optimized weight tensor with layout suited for C runtime performance
    """

    # Skip optimization for bias vectors (always 1D, no layout change needed)
    if weight_name == "bias":
        log_debug(f"Bias weights kept unchanged: {weights.shape}")
        return weights

    # Check if optimization is needed using centralized spec
    if not WeightLayoutSpec.requires_transpose(layer_type, weight_name):
        log_debug(
            f"{layer_type} {weight_name}: No layout optimization needed, shape: {weights.shape}"
        )
        return weights

    # === CONVOLUTIONAL LAYER OPTIMIZATIONS ===
    if layer_type == "Conv2D" and weight_name == "weight":
        # TensorFlow: [kernel_h, kernel_w, input_ch, output_ch] (HWIO)
        # C-optimal: [output_ch, input_ch, kernel_h, kernel_w] (OIHW)
        log_debug(f"Conv2D: Converting {WeightLayoutSpec.CONV2D_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.CONV2D_OPTIMAL_FORMAT} layout")

        transpose_axes = WeightLayoutSpec.get_conv2d_transpose_axes()
        optimized_weights = np.transpose(weights, transpose_axes)

        log_debug(f"Conv2D: {WeightLayoutSpec.CONV2D_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.CONV2D_OPTIMAL_FORMAT} {optimized_weights.shape}")
        return optimized_weights

    elif layer_type == "DepthwiseConv2D" and weight_name == "weight":
        # TensorFlow: [kernel_h, kernel_w, input_ch, depth_multiplier] (HWID)
        # C-optimal: [input_ch, depth_multiplier, kernel_h, kernel_w] (I1HW)
        log_debug(f"DepthwiseConv2D: Converting {WeightLayoutSpec.DEPTHWISE_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.DEPTHWISE_OPTIMAL_FORMAT} layout")

        transpose_axes = WeightLayoutSpec.get_depthwise_transpose_axes()
        optimized_weights = np.transpose(weights, transpose_axes)

        log_debug(
            f"DepthwiseConv2D: {WeightLayoutSpec.DEPTHWISE_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.DEPTHWISE_OPTIMAL_FORMAT} {optimized_weights.shape}"
        )
        return optimized_weights

    # === DENSE LAYER OPTIMIZATIONS ===
    elif layer_type == "Dense" and weight_name == "weight":
        # TensorFlow: [input_features, output_features]
        # C-optimal: [output_features, input_features] (transposed for cache efficiency)
        log_debug(
            f"{layer_type}: Converting {WeightLayoutSpec.DENSE_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.DENSE_OPTIMAL_FORMAT} for matrix multiplication optimization"
        )

        optimized_weights = weights.T

        log_debug(
            f"{layer_type}: {WeightLayoutSpec.DENSE_TF_FORMAT} {weights.shape} -> {WeightLayoutSpec.DENSE_OPTIMAL_FORMAT} {optimized_weights.shape} (transposed)"
        )
        return optimized_weights

    # === FALLBACK ===
    else:
        log_debug(
            f"{layer_type} {weight_name}: No layout optimization rule found, shape: {weights.shape}"
        )
        return weights


def write_alignment_padding(f, alignment=ALIGNMENT_4_BYTE, debug=False):
    """Write padding bytes to align the current file position to specified alignment"""
    current_pos = f.tell()
    padding_needed = (alignment - (current_pos % alignment)) % alignment

    if padding_needed > 0:
        padding_bytes = bytearray(padding_needed)  # Zero-filled padding
        f.write(padding_bytes)
        log_trace(f"Added {padding_needed} padding bytes for {alignment}-byte alignment (pos: {current_pos} -> {f.tell()})")

    return padding_needed

def write_weights(
    f,
    weights,
    name,
    debug=False,
    layer_type=None,
    weight_scale=None,
    weight_zp=None,
):
    """Write weight data (including shape and values) with optimized layout for embedded C runtime"""
    log_debug(f"\n--- Processing {name} weights for {layer_type} ---")
    if weight_scale is None:
        raise ValueError(
            f"weight_scale must be provided for {name} weights. Internal scale calculation removed."
        )

    # Apply unified layout optimization based on layer type and weight name
    if layer_type is not None:
        weights = optimize_weights_layout(weights, layer_type, name)

    # Quantize weights using correct symmetric quantization formula:
    # quantized_value = (float_value - zero_point) / scale
    # For symmetric quantization with zero_point=0: quantized_value = float_value / scale
    log_debug(
        f"Using precomputed weight_scale: {weight_scale:.8f}, weight_zp: {weight_zp}"
    )
    scaled_weights = ((weights - weight_zp) / weight_scale).astype(np.int16)

    # Write shape dimension
    write_tag(f, "weight_shape_d", debug)
    write_uint16(f, len(scaled_weights.shape), debug)

    # Write shape values
    write_tag(f, "weight_shape", debug)
    for dim in scaled_weights.shape:
        write_uint16(f, dim, debug)

    # Ensure 4-byte alignment before writing weight data for hardware compatibility
    write_alignment_padding(f, alignment=ALIGNMENT_4_BYTE, debug=debug)

    # Write weight data
    weight_bytes = scaled_weights.tobytes()
    f.write(weight_bytes)

    log_trace(f"Wrote {len(weight_bytes)} bytes of {name} data (pos: {f.tell()})")
