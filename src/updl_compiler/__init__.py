#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UPDL Compiler - Convert ML models to UPH5 format for embedded systems.

This package provides tools to convert Keras/TensorFlow models into
UPH5 (Upbeat Portable HDF5) format optimized for embedded deployment.
"""

__version__ = "0.1.0"
__author__ = "Upbeat Inc"
__license__ = "Apache-2.0"

from .core.loader import load_model
from .core.quantizer import initialize_params, set_udl_shift_only_mode
from .core.fuser import fuse_layers_from_json, fuse_to_uph5_layer, combine_fused_data_step5
from .core.serializer import serialize_uph5_to_c_array, serialize_uph5_metadata_to_json
from .core.config import validate_layer_configuration

def compile_model(
    model: str = None,
    quant_config: str = None,
    # Legacy parameters (for backward compatibility)
    input_path: str = None,
    quantization_params: str = None,
    model_name: str = None,
    description: str = "no_description",
    output_dir: str = None,
    udl_mode: bool = True  # Always enabled by default
):
    """
    Main API function to compile a Keras model to UPH5 format.

    Simplified interface:
        model: Path to model file (e.g., "tf_model/kws_model_split.h5")
        quant_config: Path to quantization config (e.g., "quant_config/kws_quantize_int16_ref.json")

    Legacy interface (backward compatibility):
        input_path: Path to input HDF5 model file
        quantization_params: Path to JSON file with quantization parameters
        model_name: Name for the generated model (max 16 chars)
        description: Model description (max 32 chars)
        output_dir: Directory to write output artifacts
        udl_mode: Enable UDL shift-only quantization mode (always True)

    Returns:
        dict: Compilation results with file paths and sizes
    """
    import os
    from .core.logger import log_info, set_log_level, LOG_LEVEL_INFO

    # Handle simplified interface
    if model is not None:
        input_path = model
        # Extract model name from filename (e.g., "kws_model_split.h5" -> "kws_model_split")
        model_name = os.path.splitext(os.path.basename(model))[0]

    if quant_config is not None:
        quantization_params = quant_config

    # Set defaults for simplified interface
    if output_dir is None:
        output_dir = "."  # Current directory (will create uph5/ subdirectory)

    # Set up logging
    set_log_level(LOG_LEVEL_INFO)

    try:
        # Validate configuration
        validate_layer_configuration()

        # Load model
        log_info("Loading Keras model...")
        model = load_model(input_path)

        # Initialize quantization parameters
        log_info("Loading quantization parameters...")
        if not initialize_params(quantization_params, udl_mode=udl_mode):
            raise RuntimeError(f"Failed to load quantization parameters from {quantization_params}")

        # Create fusion data
        log_info("Processing layer fusion...")
        base_name = os.path.splitext(os.path.basename(quantization_params))[0]
        fusion_json = os.path.join(output_dir, f"fusable_{base_name}.json")
        fusable_data = fuse_layers_from_json(quantization_params, fusion_json)
        fusable_layers = fusable_data["layers"]

        # Process fusion
        fused_layers = fuse_to_uph5_layer(model, fusable_layers)
        fused_data = combine_fused_data_step5(fusable_data, fused_layers)

        # Generate outputs
        log_info("Serializing outputs...")
        os.makedirs(output_dir, exist_ok=True)

        # Metadata JSON
        metadata_json = os.path.join(output_dir, ".updlc_cache", f"uph5_metadata_{base_name}.json")
        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
        serialize_uph5_metadata_to_json(fused_data, metadata_json)

        # C Array - save to uph5 directory
        uph5_dir = os.path.join(output_dir, "uph5")
        os.makedirs(uph5_dir, exist_ok=True)
        file_size = serialize_uph5_to_c_array(fused_data, model_name, description=description, output_dir=uph5_dir)

        log_info(f"Compilation completed successfully! Generated {file_size} bytes")

        return {
            "model_name": model_name,
            "file_size": file_size,
            "metadata_json": metadata_json,
            "uph5_dir": uph5_dir,
            "output_dir": output_dir
        }

    except Exception as e:
        from .core.logger import log_error
        log_error(f"Compilation failed: {e}")
        raise

__all__ = [
    "compile_model",
    "load_model",
    "initialize_params",
    "set_udl_shift_only_mode",
    "fuse_layers_from_json",
    "fuse_to_uph5_layer",
    "combine_fused_data_step5",
    "serialize_uph5_to_c_array",
    "serialize_uph5_metadata_to_json",
    "validate_layer_configuration",
]