#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, Optional, Sequence
from .logger import (
    log_error,
    log_info,
)
from .schema import (
    DESCRIPTION_LENGTH,
    DTYPE_LIST,
    LTYPE_LIST,
    PTYPE_LIST,
    ATYPE_LIST,
    UPH5Compatibility,
)
from .serialize_util import (
    write_string,
    write_tag,
    write_uint16,
    write_float32,
    write_shape,
    write_enum,
    write_quantization_parameters,
    write_weights,
    optimize_weights_layout,
)
import numpy as np
import json
from .license import MLPERF_APACHE_LICENSE_HEADER

def serialize_input_feature_to_c_array(
    quantized_samples: Sequence[np.ndarray],
    labels: Sequence[str],
    input_size: int,
    *,
    array_name: str = "g_model_inputs_int16",
    element_type: str = "int16_t",
    outer_dim_token: Optional[str] = None,
    inner_dim_token: Optional[str] = None,
    include_directive: Optional[str] = None,
    license_header: str = MLPERF_APACHE_LICENSE_HEADER,
    value_formatter: Optional[Callable[[float], str]] = None,
    values_per_line: int = 16,
) -> str:
    """
    Format numeric input samples into a C array definition suitable for tests.

    Args:
        quantized_samples: Collection of flattened numeric feature vectors.
        labels: Labels used for per-sample comments (falls back to `sample_{i}`).
        input_size: Expected length of each sample (validated for consistency).
        array_name: Symbol name used in the emitted C code.
        element_type: C type for array elements (defaults to `int16_t`).
        outer_dim_token: Optional identifier for number of samples (defaults to literal).
        inner_dim_token: Optional identifier for feature dimension (defaults to literal).
        include_directive: Optional `#include` to insert before the array definition.
        license_header: Optional license header prepended to the output.
        values_per_line: Number of values emitted per row for readability.

    Returns:
        str: C source representation of the input feature array.

    Raises:
        ValueError: If no samples are provided or sample lengths are inconsistent.
    """
    samples = [np.asarray(sample) for sample in quantized_samples]
    if not samples:
        raise ValueError("No samples provided for input feature serialization.")

    for idx, sample in enumerate(samples):
        if sample.size != input_size:
            raise ValueError(
                f"Sample {idx} has size {sample.size}, expected {input_size}"
            )

    outer_dim = outer_dim_token or str(len(samples))
    inner_dim = inner_dim_token or str(input_size)

    formatter: Callable[[float], str]
    if value_formatter is not None:
        formatter = value_formatter
    else:
        sample_dtype = next(
            (sample.dtype for sample in samples if hasattr(sample, "dtype")), None
        )
        is_float_values = False
        if sample_dtype is not None and np.issubdtype(sample_dtype, np.floating):
            is_float_values = True
        elif element_type and "float" in element_type:
            is_float_values = True

        if is_float_values:
            def formatter(value: float) -> str:
                return f"{float(value):.8f}f"
        else:
            def formatter(value: float) -> str:
                return str(int(value))

    lines: list[str] = []
    if license_header:
        lines.append(license_header.strip("\n"))
        lines.append("")

    if include_directive:
        lines.append(include_directive)
        lines.append("")

    lines.append(
        f"const {element_type} {array_name}[{outer_dim}][{inner_dim}] = {{"
    )

    for idx, sample in enumerate(samples):
        label = labels[idx] if idx < len(labels) else f"sample_{idx}"
        lines.append(f"    {{ // {label}")
        for offset in range(0, input_size, values_per_line):
            chunk = sample[offset : offset + values_per_line]
            values = ", ".join(formatter(value) for value in chunk)
            trailing_comma = "," if offset + values_per_line < input_size else ""
            lines.append(f"        {values}{trailing_comma}")
        array_comma = "," if idx + 1 < len(samples) else ""
        lines.append(f"    }}{array_comma}")

    lines.append("};")
    return "\n".join(lines) + "\n"


def serialize_uph5_metadata_to_json(fused_data, output_file):
    """Step 6: Focus on serialization - serialize metadata from fused_data only

    Input: fused_data (contains 'input' and 'layers' sections)
    Output: JSON metadata file without weights/bias data
    """

    # Extract input information
    input_data = fused_data.get("input", {})
    layers_data = fused_data.get("layers", {})

    # Create metadata structure
    metadata = {
        "model_info": {
            "num_layers": len(layers_data),
            "input_quantization": {
                "scale": input_data.get("scale", 1.0),
                "zero_point": input_data.get("zero_point", 0),
            },
        },
        "layers": {},
    }

    # Process each fused layer
    for layer_idx, (layer_name, layer_data) in enumerate(layers_data.items()):
        layer_metadata = {
            "layer_type": layer_data.get("layer_type", "Unknown"),
            "activation": layer_data.get("activation", "linear"),
            "input_shape": layer_data.get("input_shape", []),
            "output_shape": layer_data.get("output_shape", []),
            "quantization": {
                "act_scale": layer_data.get("act_scale", 1.0),
                "act_zp": layer_data.get("act_zp", 0),
                "act_min_val": layer_data.get("act_min_val", 0.0),
                "act_max_val": layer_data.get("act_max_val", 1.0),
                "weight_scale": layer_data.get("weight_scale", 1.0),
                "weight_zp": layer_data.get("weight_zp", 0),
                "bias_scale": layer_data.get("bias_scale"),
                "bias_zp": layer_data.get("bias_zp"),
            },
        }

        # Add weight shape information (metadata only, no actual weights)
        if layer_data.get("weight_shape") is not None:
            layer_metadata["weight_shape"] = layer_data["weight_shape"]

        if layer_data.get("bias") is not None:
            layer_metadata["has_bias"] = True

        metadata["layers"][f"layer_{layer_idx}"] = layer_metadata

    # Write to file
    try:
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)
        log_info(f"UPH5 metadata written to {output_file}")
        log_info(f"Metadata contains {len(metadata['layers'])} layer descriptions")
    except Exception as e:
        log_error(f"Serializer: Cannot write UPH5 metadata to {output_file} - {e}. Check file permissions and disk space.")
        raise

def serialize_uph5_weight_to_json(fused_data, output_file):
    
    # Extract layers data
    layers_data = fused_data.get("layers", {})
    
    # Create weights debug structure
    weights_debug = {
        "description": "CHW-optimized weights as stored in C array (for debugging)",
        "format_info": {
            "conv2d": "OIHW format: [output_ch, input_ch, kernel_h, kernel_w]",
            "depthwise_conv2d": "I1HW format: [input_ch, 1, kernel_h, kernel_w]", 
            "dense": "Original format: [input_features, output_features]"
        },
        "layers": {}
    }
    
    # Process each layer
    for layer_idx, (layer_name, layer_data) in enumerate(layers_data.items()):
        layer_type = layer_data.get("layer_type", "Unknown")
        
        layer_weights_info = {
            "layer_type": layer_type,
            "layer_index": layer_idx,
            "weight_scale": layer_data.get("weight_scale", 1.0),
            "weight_zp": layer_data.get("weight_zp", 0),
            "has_weights": False,
            "has_bias": False
        }
        
        # Process weights if they exist
        if layer_data.get("weight") is not None:
            weights = layer_data["weight"]
            
            # Apply the same layout optimization that will be used during binary serialization
            converted_weights = optimize_weights_layout(weights, layer_type, "weight")
            if not np.array_equal(weights, converted_weights):
                log_info(f"Applied layout optimization for {layer_type}: {weights.shape} -> {converted_weights.shape}")
            
            # Store weight information
            layer_weights_info["has_weights"] = True
            layer_weights_info["original_shape"] = list(weights.shape)
            layer_weights_info["converted_shape"] = list(converted_weights.shape)
            
            # For Conv2D layers, save detailed first filter for debugging
            if layer_type == "Conv2D" and len(converted_weights.shape) == 4:
                out_ch, in_ch, h, w = converted_weights.shape
                layer_weights_info["weight_tensor_format"] = "OIHW"
                
                # Save first filter (or as much as available)
                filter_h = min(h, 10)
                filter_w = min(w, 4) 
                if filter_h > 0 and filter_w > 0:
                    first_filter = converted_weights[0, 0, :filter_h, :filter_w]
                    layer_weights_info["first_filter_debug"] = {
                        "shape": [filter_h, filter_w],
                        "format": "OIHW[0,0,:,:]",
                        "float32_values": first_filter.tolist(),
                        "description": f"First filter ({filter_h}x{filter_w}) from OIHW weights[0,0,:,:]"
                    }
                    
                # Save weight statistics
                layer_weights_info["weight_stats"] = {
                    "min": float(np.min(converted_weights)),
                    "max": float(np.max(converted_weights)),
                    "mean": float(np.mean(converted_weights)),
                    "std": float(np.std(converted_weights))
                }
                
            elif layer_type == "DepthwiseConv2D" and len(converted_weights.shape) == 4:
                in_ch, depth_mult, h, w = converted_weights.shape
                layer_weights_info["weight_tensor_format"] = "I1HW"
                
                # Save first filter
                filter_h = min(h, 10)
                filter_w = min(w, 4)
                if filter_h > 0 and filter_w > 0:
                    first_filter = converted_weights[0, 0, :filter_h, :filter_w]
                    layer_weights_info["first_filter_debug"] = {
                        "shape": [filter_h, filter_w],
                        "format": "I1HW[0,0,:,:]", 
                        "float32_values": first_filter.tolist(),
                        "description": f"First filter ({filter_h}x{filter_w}) from I1HW weights[0,0,:,:]"
                    }
                    
            elif layer_type == "Dense":
                layer_weights_info["weight_tensor_format"] = "2D"
                # Save first few weights for debugging
                if len(converted_weights.shape) == 2:
                    rows, cols = converted_weights.shape
                    sample_rows = min(rows, 4)
                    sample_cols = min(cols, 8)
                    weight_sample = converted_weights[:sample_rows, :sample_cols]
                    layer_weights_info["weight_sample_debug"] = {
                        "shape": [sample_rows, sample_cols],
                        "format": "[input_features, output_features]",
                        "float32_values": weight_sample.tolist(),
                        "description": f"Sample weights ({sample_rows}x{sample_cols})"
                    }
        
        # Process bias if it exists  
        if layer_data.get("bias") is not None:
            bias = layer_data["bias"]
            layer_weights_info["has_bias"] = True
            layer_weights_info["bias_shape"] = list(bias.shape)
            
            # Save first few bias values for debugging
            num_bias_to_save = min(len(bias), 8)
            layer_weights_info["bias_debug"] = {
                "shape": [num_bias_to_save],
                "float32_values": bias[:num_bias_to_save].tolist(),
                "description": f"First {num_bias_to_save} bias values"
            }
        
        weights_debug["layers"][f"layer_{layer_idx}_{layer_name}"] = layer_weights_info
    
    # Write to file
    try:
        with open(output_file, "w") as f:
            json.dump(weights_debug, f, indent=2)
    except Exception as e:
        log_error(f"Serializer: Cannot write weights debug data to {output_file} - {e}. Check file permissions and disk space.")
        raise


def serialize_uph5_to_binary(
    fused_data, output_file, model_name="model", description="no_description"
):
    """Step 6: Focus on serialization - serialize binary data from fused_data only

    Input: fused_data (contains 'input' and 'layers' sections)
    Output: Binary UPH5 file
    """

    input_data = fused_data.get("input", {})
    layers_data = fused_data.get("layers", {})

    with open(output_file, "wb") as f:
        # Write header section
        write_string(f, description, DESCRIPTION_LENGTH, debug=False)

        write_tag(f, "model_name", debug=False)
        write_string(f, model_name, debug=False)

        # Write model info
        write_tag(f, "num_layers", debug=False)
        write_uint16(f, len(layers_data), debug=False)

        # Write input shape (derive from first layer or use default)
        write_tag(f, "batch_inputshape", debug=False)
        if layers_data:
            first_layer = list(layers_data.values())[0]
            input_shape = first_layer.get("input_shape", [1, 6, 1, 1])
        else:
            input_shape = [1, 6, 1, 1]  # Default shape

        # Ensure 4D shape
        while len(input_shape) < 4:
            input_shape.append(1)
        write_shape(f, input_shape[:4], 4, debug=False)

        # Data type
        write_tag(f, "dtype", debug=False)
        dtype = UPH5Compatibility.DEFAULT_DTYPE
        write_string(f, dtype, debug=False)
        write_enum(f, dtype, DTYPE_LIST, debug=False)

        # Input scale (global input quantization scale)
        write_tag(f, "input_scale", debug=False)
        # Get scale from input.scale field (JSON structure: {"input": {"scale": value}})
        input_scale = input_data.get("scale")  # Default fallback
        write_float32(f, input_scale, debug=False)

        # Write each fused layer
        for layer_idx, (layer_name, layer_data) in enumerate(layers_data.items()):
            layer_type = layer_data.get("layer_type", "Unknown")

            # Map layer type for storage
            stored_layer_type = (
                "BatchNorm" if layer_type == "BatchNormalization" else layer_type
            )

            log_info(f"Serializing layer {layer_idx}: {layer_name} ({layer_type})")

            # Write layer header
            write_tag(f, "name", debug=False)
            write_string(f, layer_name, debug=False)

            write_tag(f, "type", debug=False)
            write_string(f, stored_layer_type, debug=False)
            write_enum(f, stored_layer_type, LTYPE_LIST, debug=False)

            # Write layer-specific data based on fused_data
            _write_layer_data_from_fused(f, layer_data, layer_type, layer_idx, layers_data)

    file_size = os.path.getsize(output_file)
    log_info(f"UPH5 binary file written: {file_size} bytes")
    return file_size


def _write_layer_data_from_fused(f, layer_data, layer_type, layer_idx=0, all_layers=None):
    """Helper function to write layer-specific data from fused_data"""
    # Write shapes
    write_tag(f, "input_shape", debug=False)
    input_shape = layer_data.get("input_shape", [1, 1, 1, 1])

    # Handle Add layer shape processing specifically
    if layer_type == "Add":
        # Convert tuple to list and handle None values (dynamic batch size)
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)

        # Flatten nested structures: if input_shape contains tuples, flatten them
        flattened_shape = []
        for item in input_shape:
            if isinstance(item, (tuple, list)):
                # Take the first tuple if there are multiple identical ones
                flattened_shape = list(item)
                break
            else:
                flattened_shape.append(item)

        # If we found a nested structure, use it; otherwise use original
        if any(isinstance(item, (tuple, list)) for item in input_shape):
            input_shape = flattened_shape

    # Replace None with 1 (common for dynamic batch size) - only if None values exist
    if any(dim is None for dim in input_shape):
        input_shape = [1 if dim is None else dim for dim in input_shape]

    # Dense layer expects 2D input shape in C loader
    if layer_type == "Dense":
        # Dense input should be 2D: [batch_size, input_features]
        if len(input_shape) >= 2:
            input_shape_2d = input_shape[:2]  # Take first 2 dimensions
        else:
            input_shape_2d = [1, 64]  # Default fallback
        write_shape(f, input_shape_2d, 2, debug=False)
    else:
        # All other layers use 4D input shape (including Softmax)
        while len(input_shape) < 4:
            input_shape.append(1)
        write_shape(f, input_shape[:4], 4, debug=False)

    write_tag(f, "output_shape", debug=False)
    output_shape = layer_data.get("output_shape", [1, 1, 1, 1])

    # Handle Add layer shape processing specifically
    if layer_type == "Add":
        # Convert tuple to list and handle None values (dynamic batch size)
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)

        # Flatten nested structures: if output_shape contains tuples, flatten them
        flattened_shape = []
        for item in output_shape:
            if isinstance(item, (tuple, list)):
                # Take the first tuple if there are multiple identical ones
                flattened_shape = list(item)
                break
            else:
                flattened_shape.append(item)

        # If we found a nested structure, use it; otherwise use original
        if any(isinstance(item, (tuple, list)) for item in output_shape):
            output_shape = flattened_shape

    # Replace None with 1 (common for dynamic batch size) - only if None values exist
    if any(dim is None for dim in output_shape):
        output_shape = [1 if dim is None else dim for dim in output_shape]

    # Flatten and Dense layers expect 2D output shape in C loader
    if layer_type in ["Flatten", "Dense"]:
        # Output should be 2D: [batch_size, features]
        if len(output_shape) >= 2:
            output_shape_2d = output_shape[:2]  # Take first 2 dimensions
        else:
            output_shape_2d = [1, 64]  # Default fallback
        write_shape(f, output_shape_2d, 2, debug=False)
    else:
        # All other layers use 4D output shape (including Softmax)
        while len(output_shape) < 4:
            output_shape.append(1)
        write_shape(f, output_shape[:4], 4, debug=False)

    # Write layer-specific parameters from fused_data
    if layer_type in ["Conv2D", "Conv1D"]:
        # Conv layer parameters from fused_data
        write_tag(f, "filters", debug=False)
        write_uint16(f, int(layer_data.get("filters", 32)), debug=False)

        write_tag(f, "kernel_size", debug=False)
        kernel_size = layer_data.get("kernel_size", [3, 3])
        if isinstance(kernel_size, (list, tuple)):
            write_uint16(f, int(kernel_size[0]), debug=False)
            if layer_type != "Conv1D":
                write_uint16(
                    f,
                    (
                        int(kernel_size[1])
                        if len(kernel_size) > 1
                        else int(kernel_size[0])
                    ),
                    debug=False,
                )
        else:
            write_uint16(f, int(kernel_size), debug=False)
            if layer_type != "Conv1D":
                write_uint16(f, int(kernel_size), debug=False)

        write_tag(f, "strides", debug=False)
        strides = layer_data.get("strides", [1, 1])
        if isinstance(strides, (list, tuple)):
            write_uint16(f, int(strides[0]), debug=False)
            if layer_type != "Conv1D":
                write_uint16(
                    f,
                    int(strides[1]) if len(strides) > 1 else int(strides[0]),
                    debug=False,
                )
        else:
            write_uint16(f, int(strides), debug=False)
            if layer_type != "Conv1D":
                write_uint16(f, int(strides), debug=False)

        write_tag(f, "padding", debug=False)
        padding = layer_data.get("padding", "valid")
        write_string(f, padding, debug=False)
        write_enum(f, padding, PTYPE_LIST, debug=False)

    elif layer_type == "DepthwiseConv2D":
        # DepthwiseConv2D parameters from fused_data
        write_tag(f, "depth_multiplier", debug=False)
        write_uint16(f, int(layer_data.get("depth_multiplier", 1)), debug=False)

        write_tag(f, "kernel_size", debug=False)
        kernel_size = layer_data.get("kernel_size", [3, 3])
        if isinstance(kernel_size, (list, tuple)):
            write_uint16(f, int(kernel_size[0]), debug=False)
            write_uint16(
                f,
                int(kernel_size[1]) if len(kernel_size) > 1 else int(kernel_size[0]),
                debug=False,
            )
        else:
            write_uint16(f, int(kernel_size), debug=False)
            write_uint16(f, int(kernel_size), debug=False)

        write_tag(f, "strides", debug=False)
        strides = layer_data.get("strides", [1, 1])
        if isinstance(strides, (list, tuple)):
            write_uint16(f, int(strides[0]), debug=False)
            write_uint16(
                f, int(strides[1]) if len(strides) > 1 else int(strides[0]), debug=False
            )
        else:
            write_uint16(f, int(strides), debug=False)
            write_uint16(f, int(strides), debug=False)

        write_tag(f, "padding", debug=False)
        padding = layer_data.get("padding", "valid")
        write_string(f, padding, debug=False)
        write_enum(f, padding, PTYPE_LIST, debug=False)

    elif layer_type == "Dense":
        # Dense layer parameters from fused_data
        write_tag(f, "units", debug=False)
        write_uint16(f, int(layer_data.get("units", 10)), debug=False)

    elif layer_type in ["MaxPooling2D", "AveragePooling2D"]:
        # Pooling layer parameters from fused_data
        write_tag(f, "pool_size", debug=False)
        pool_size = layer_data.get("pool_size", [2, 2])
        if isinstance(pool_size, (list, tuple)):
            write_uint16(f, int(pool_size[0]), debug=False)
            write_uint16(
                f,
                int(pool_size[1]) if len(pool_size) > 1 else int(pool_size[0]),
                debug=False,
            )
        else:
            write_uint16(f, int(pool_size), debug=False)
            write_uint16(f, int(pool_size), debug=False)

        write_tag(f, "strides", debug=False)
        strides = layer_data.get("strides", [2, 2])
        if isinstance(strides, (list, tuple)):
            write_uint16(f, int(strides[0]), debug=False)
            write_uint16(
                f, int(strides[1]) if len(strides) > 1 else int(strides[0]), debug=False
            )
        else:
            write_uint16(f, int(strides), debug=False)
            write_uint16(f, int(strides), debug=False)

        write_tag(f, "padding", debug=False)
        padding = layer_data.get("padding", "valid")
        write_string(f, padding, debug=False)
        write_enum(f, padding, PTYPE_LIST, debug=False)

    elif layer_type == "Flatten":
        # Flatten layer has no specific parameters beyond input/output shapes
        # The shapes are already written above, so nothing additional needed
        pass

    elif layer_type == "Add":
        # Add layer has no specific parameters beyond input/output shapes
        # The shapes are already written above, so nothing additional needed
        pass

    elif layer_type == "Softmax":
        # Softmax layer has no specific parameters beyond input/output shapes
        # The shapes are already written above, so nothing additional needed
        pass

    # Write activation
    write_tag(f, "activation", debug=False)
    activation = layer_data.get("activation", "linear")
    write_string(f, activation, debug=False)
    write_enum(f, activation, ATYPE_LIST, debug=False)

    # Write quantization parameters
    def extract_scalar(value, default):
        """Extract scalar from potentially tuple/array value"""
        if isinstance(value, (tuple, list)):
            return value[0] if len(value) > 0 else default
        return value if value is not None else default

    act_scale = extract_scalar(layer_data.get("act_scale"), 1.0)
    act_zp = extract_scalar(layer_data.get("act_zp"), 0)
    weight_scale = extract_scalar(layer_data.get("weight_scale"), 1.0)
    weight_zp = extract_scalar(layer_data.get("weight_zp"), 0)
    bias_scale = extract_scalar(layer_data.get("bias_scale"), 1.0)
    bias_zp = extract_scalar(layer_data.get("bias_zp"), 0)        

    write_quantization_parameters(
        f, act_scale, act_zp, weight_scale, weight_zp, bias_scale, bias_zp, debug=False
    )

    # Write weights and bias if they exist
    if layer_data.get("weight") is not None:
        # Pass the layer's weight_scale and weight_zp to ensure consistency
        layer_weight_scale = layer_data.get("weight_scale")
        layer_weight_zp = layer_data.get("weight_zp", 0)
        write_weights(
            f, layer_data["weight"], "weight", debug=False, layer_type=layer_type,
            weight_scale=layer_weight_scale, weight_zp=layer_weight_zp
        )

    if layer_data.get("bias") is not None:
        # Use precomputed bias quantization parameters from fusion step
        bias_scale = layer_data.get("bias_scale")
        bias_zp = layer_data.get("bias_zp", 0)
        write_weights(f, layer_data["bias"], "bias", debug=False, layer_type=layer_type,
                     weight_scale=bias_scale, weight_zp=bias_zp)


def serialize_uph5_to_c_array(fused_data, model_name, description="no_description", output_dir="./uph5"):
    """Step 6: Convert fused_data to C array format"""
    import tempfile


    # Generate temp binary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".uph5") as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Generate binary data from fused_data
        file_size = serialize_uph5_to_binary(
            fused_data, tmp_path, model_name, description
        )

        # Read the binary data
        with open(tmp_path, "rb") as f:
            data_bytes = f.read()

        # Generate C array files using UPH5 compatibility specs
        base_filename = f"{model_name}_int16"
        h_file_name = f"{base_filename}{UPH5Compatibility.HEADER_EXTENSION}"
        c_file_name = f"{base_filename}{UPH5Compatibility.SOURCE_EXTENSION}"

        # Output paths - now configurable
        h_file_path = os.path.join(output_dir, h_file_name)
        c_file_path = os.path.join(output_dir, c_file_name)

        # Ensure directories exist
        os.makedirs(os.path.dirname(h_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(c_file_path), exist_ok=True)

        # Write C file
        with open(c_file_path, "w") as f:
            f.write(f'#include "{h_file_name}"\n\n')
            f.write(f"// Generated UPH5 model data for: {description}\n")
            f.write(f"// Model name: {model_name}\n")
            f.write(f"// Data size: {len(data_bytes)} bytes\n")
            f.write(f"// Aligned for {UPH5Compatibility.UDL_VERSION} compatibility ({UPH5Compatibility.MEMORY_ALIGNMENT}-byte alignment)\n\n")

            f.write(f"__attribute__((aligned(4))) const unsigned char {model_name}_data[] = {{\n")

            # Write bytes in rows of 16 for readability
            for i in range(0, len(data_bytes), 16):
                row = data_bytes[i : i + 16]
                hex_values = ", ".join(f"0x{b:02x}" for b in row)
                f.write(f"    {hex_values}")
                if i + 16 < len(data_bytes):
                    f.write(",")
                f.write("\n")

            f.write("};\n\n")
            f.write(f"const unsigned int {model_name}_data_size = {len(data_bytes)};\n")

        # Write header file
        header_guard = h_file_name.replace(".", "_").upper()
        with open(h_file_path, "w") as f:
            f.write(f"#ifndef {header_guard}\n")
            f.write(f"#define {header_guard}\n\n")
            f.write(f"// Generated UPH5 model data for: {description}\n")
            f.write(f"// Model name: {model_name}\n")
            f.write(f"// Aligned for {UPH5Compatibility.UDL_VERSION} compatibility ({UPH5Compatibility.MEMORY_ALIGNMENT}-byte alignment)\n\n")
            f.write(f"extern __attribute__((aligned(4))) const unsigned char {model_name}_data[];\n")
            f.write(f"extern const unsigned int {model_name}_data_size;\n\n")
            f.write(f"#endif // {header_guard}\n")

        log_info(f"Generated C array files: {c_file_path}, {h_file_path}")
        return file_size

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
