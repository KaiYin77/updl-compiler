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
from .core.quantization_analyzer import QuantizationAnalyzer
from .core.preprocessors import DataPreprocessor

def compile_model(
    model: str,
    preprocessor: DataPreprocessor,
    calibration_data=None,
    model_name: str = None,
    description: str = "no_description",
    output_dir: str = None,
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
        log_info("Step 1: Loading model...")
        loaded_model = load_model(model)

        # Step 2: Run quantization analysis
        log_info("Step 2: Performing quantization analysis...")

        # Load calibration data using preprocessor
        if calibration_data:
            calibration_samples = preprocessor.load_calibration_data(calibration_data)
        else:
            raise ValueError("calibration_data is required for quantization analysis")

        # Create quantization analyzer
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
        log_info("Step 3: Compiling model with quantization parameters...")

        # Initialize quantization parameters
        if not initialize_params(quant_params_file, udl_mode=True):
            raise RuntimeError(f"Failed to load generated quantization parameters from {quant_params_file}")

        # Create fusion data
        log_info("Processing layer fusion...")
        base_name = os.path.splitext(os.path.basename(quant_params_file))[0]
        fusion_json = os.path.join(output_dir, ".updlc_cache", f"fusable_{base_name}.json")
        fusable_data = fuse_layers_from_json(quant_params_file, fusion_json)
        fusable_layers = fusable_data["layers"]

        # Process fusion
        fused_layers = fuse_to_uph5_layer(loaded_model, fusable_layers)
        fused_data = combine_fused_data_step5(fusable_data, fused_layers)

        # Generate outputs
        log_info("Serializing outputs...")

        # Metadata JSON
        metadata_json = os.path.join(output_dir, ".updlc_cache", f"uph5_metadata_{base_name}.json")
        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
        serialize_uph5_metadata_to_json(fused_data, metadata_json)

        # C Array - save to uph5 directory
        uph5_dir = os.path.join(output_dir, "uph5")
        os.makedirs(uph5_dir, exist_ok=True)
        file_size = serialize_uph5_to_c_array(fused_data, model_name, description=description, output_dir=uph5_dir)

        result = {
            "model_name": model_name,
            "file_size": file_size,
            "metadata_json": metadata_json,
            "uph5_dir": uph5_dir,
            "output_dir": output_dir
        }

        # Add quantization info to result
        result["quantization_params"] = quant_params_file
        result["calibration_samples"] = len(calibration_samples)

        log_info(f"Compilation completed successfully!")
        return result

    except Exception as e:
        from .core.logger import log_error
        log_error(f"Compilation failed: {e}")
        raise


__all__ = [
    "compile_model",
    "DataPreprocessor",
    "QuantizationAnalyzer",
]