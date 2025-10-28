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
from .core.serializer import (
    serialize_uph5_to_c_array,
    serialize_uph5_metadata_to_json,
    serialize_flatbuffers_to_c_array,
    serialize_uph5_to_flatbuffers
)
from .core.config import validate_layer_configuration
from .core.quantization_analyzer import QuantizationAnalyzer
from .core.preprocessors import DataPreprocessor

def compile_model(
    model: str,
    preprocessor: DataPreprocessor,
    calibration_data=None,
    model_name: str = None,
    description: str = "no_description",
    output_dir: str = None,
    format: str = "uph5",  # "uph5" or "flatbuffers"
):
    """
    Compile a model with automatic quantization analysis (streamlined workflow)

    Args:
        model: Path to model file or directory
        preprocessor: DataPreprocessor instance for the specific model type
        calibration_data: Calibration data source (dataset path, data directory, or sample list)
        model_name: Name for the generated model (auto-extracted if not provided)
        description: Model description
        output_dir: Directory to write output artifacts
        format: Output format - "uph5" (default) or "flatbuffers"

    Returns:
        dict: Compilation results with file paths and sizes
    """
    import os
    from .core.logger import log_info, set_log_level, LOG_LEVEL_INFO

    # Set defaults
    if output_dir is None:
        output_dir = "."

    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model))[0]

    # Set up logging
    set_log_level(LOG_LEVEL_INFO)

    try:
        # Validate configuration
        validate_layer_configuration()

        # Step 1: Load model
        log_info(f"Model Loader: Loading {model}")
        loaded_model = load_model(model)

        # Step 2: Run quantization analysis
        if calibration_data:
            calibration_samples = preprocessor.load_calibration_data(calibration_data)
        else:
            raise ValueError("Quantization Analyzer: calibration_data is required for quantization analysis")

        log_info(f"Quantization Analyzer: Starting analysis with {len(calibration_samples)} samples")
        analyzer = QuantizationAnalyzer(preprocessor=preprocessor)

        # Generate quantization parameters and save to .updlc_cache
        cache_dir = os.path.join(output_dir, ".updlc_cache")
        os.makedirs(cache_dir, exist_ok=True)

        quant_params_file = os.path.join(cache_dir, f"{model_name}_quantize_params.json")
        quantization_params = analyzer.analyze_model_quantization(
            loaded_model,
            calibration_samples,
            quant_params_file,
            udl_mode=True
        )

        # Step 3: Continue with compilation using generated parameters
        if not initialize_params(quant_params_file, udl_mode=True):
            raise RuntimeError(f"Parameter Initializer: Failed to load quantization parameters from {quant_params_file}")

        # Create fusion data
        log_info(f"Layer Fuser: Processing {len(loaded_model.layers)} layers")
        base_name = os.path.splitext(os.path.basename(quant_params_file))[0]
        fusion_json = os.path.join(output_dir, ".updlc_cache", f"fusable_{base_name}.json")
        fusable_data = fuse_layers_from_json(quant_params_file, fusion_json)
        fusable_layers = fusable_data["layers"]

        # Process fusion
        fused_layers = fuse_to_uph5_layer(loaded_model, fusable_layers)
        fused_data = combine_fused_data_step5(fusable_data, fused_layers)

        # Generate outputs
        log_info(f"Serializer: Generating {format} C arrays for {model_name}")

        # Metadata JSON
        metadata_json = os.path.join(output_dir, ".updlc_cache", f"{format}_metadata_{base_name}.json")
        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
        serialize_uph5_metadata_to_json(fused_data, metadata_json)

        # C Array - choose format
        output_subdir = "uph5" if format == "uph5" else "flatbuffers"
        output_format_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(output_format_dir, exist_ok=True)

        if format == "flatbuffers":
            file_size = serialize_flatbuffers_to_c_array(fused_data, model_name, description=description, output_dir=output_format_dir)
            log_info(f"FlatBuffers: Generated C arrays in {output_format_dir}")
        else:
            file_size = serialize_uph5_to_c_array(fused_data, model_name, description=description, output_dir=output_format_dir)
            log_info(f"UPH5: Generated C arrays in {output_format_dir}")

        result = {
            "model_name": model_name,
            "file_size": file_size,
            "metadata_json": metadata_json,
            "output_format_dir": output_format_dir,
            "output_dir": output_dir,
            "format": format
        }

        # Add quantization info to result
        result["quantization_params"] = quant_params_file
        result["calibration_samples"] = len(calibration_samples)

        log_info(f"Compilation: Successfully generated {file_size} bytes")
        return result

    except Exception as e:
        from .core.logger import log_error
        log_error(f"Compilation: Failed during model compilation - {e}")
        raise


__all__ = [
    "compile_model",
    "DataPreprocessor",
    "QuantizationAnalyzer",
]