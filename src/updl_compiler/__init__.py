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
from .core.codegen import serialize_uph5_to_c_array, serialize_uph5_metadata_to_json
from .core.config import validate_layer_configuration
from .core.calibrator import Calibrator
from .core.preprocessors import DataPreprocessor
from .core.pipeline import CompilationSession, CompilationResult

def compile_model(
    model: str,
    preprocessor: DataPreprocessor,
    calibration_data=None,
    calibration_count: int | None = None,
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
    from .core.logger import log_error, log_info, LOG_LEVEL_INFO

    try:
        session = CompilationSession(
            model_path=model,
            preprocessor=preprocessor,
            calibration_source=calibration_data,
            calibration_count=calibration_count,
            model_name=model_name,
            description=description,
            output_dir=output_dir,
            log_level=LOG_LEVEL_INFO,
        )
        session_result = session.run()

        result = {
            "model_name": session_result.model_name,
            "file_size": session_result.file_size,
            "metadata_json": session_result.metadata_json,
            "uph5_dir": session_result.uph5_dir,
            "output_dir": session_result.output_dir,
            "quantization_params": session_result.quantization_params,
            "calibration_samples": session_result.calibration_samples,
            "cache_dir": session_result.cache_dir,
        }

        log_info(f"Compilation: Successfully generated {session_result.file_size} bytes")
        return result

    except Exception as e:
        log_error(f"Compilation: Failed during model compilation - {e}")
        raise


__all__ = [
    "compile_model",
    "DataPreprocessor",
    "Calibrator",
]
