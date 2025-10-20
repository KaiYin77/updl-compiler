#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
from ..core.logger import (
    LOG_LEVEL_OFF,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_TRACE,
    set_log_level,
    log_info,
    log_error,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="UPDL Compiler - Convert ML models to UPH5 format with automatic quantization"
    )
    parser.add_argument("--model", required=True, help="Path to model file or directory")
    parser.add_argument(
        "--preprocessor",
        required=True,
        help="Preprocessor type (e.g., 'kws') or path to custom preprocessor"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory or calibration data"
    )
    parser.add_argument(
        "--description",
        default="no_description",
        help="Model description (max 32 chars)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (auto-extracted from filename if not provided)"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=120,
        help="Number of calibration samples to use for quantization"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--log-level",
        choices=["off", "error", "warn", "info", "debug", "trace"],
        default="info",
        help="Set log level (default: info)",
    )
    return parser.parse_args()


def load_preprocessor(preprocessor_arg):
    """Load preprocessor based on argument"""
    if preprocessor_arg == "kws":
        # Import KWS preprocessor from examples
        examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "kws")
        sys.path.insert(0, examples_dir)
        try:
            from kws_preprocessor import KWSPreprocessor
            return KWSPreprocessor()
        except ImportError as e:
            raise RuntimeError(f"Failed to load KWS preprocessor: {e}")
    else:
        # Try to import custom preprocessor
        try:
            if os.path.exists(preprocessor_arg):
                # Load from file path
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_preprocessor", preprocessor_arg)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Assume the module has a main preprocessor class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "__bases__") and any("DataPreprocessor" in str(base) for base in attr.__bases__):
                        return attr()
                raise RuntimeError("No DataPreprocessor subclass found in custom preprocessor file")
            else:
                raise RuntimeError(f"Preprocessor file not found: {preprocessor_arg}")
        except Exception as e:
            raise RuntimeError(f"Failed to load custom preprocessor: {e}")


def main():
    """Main entry point - streamlined workflow with automatic quantization"""
    args = parse_args()

    # Set log level
    log_level_map = {
        "off": LOG_LEVEL_OFF,
        "error": LOG_LEVEL_ERROR,
        "warn": LOG_LEVEL_WARN,
        "info": LOG_LEVEL_INFO,
        "debug": LOG_LEVEL_DEBUG,
        "trace": LOG_LEVEL_TRACE,
    }
    set_log_level(log_level_map[args.log_level])

    try:
        log_info("=== UPDL Model Compilation with Automatic Quantization ===")

        # Load preprocessor
        log_info(f"Loading preprocessor: {args.preprocessor}")
        preprocessor = load_preprocessor(args.preprocessor)

        # Import compile_model from main package
        from .. import compile_model

        # Auto-extract model name if not provided
        if args.model_name is None:
            args.model_name = os.path.splitext(os.path.basename(args.model))[0]
            log_info(f"Auto-extracted model name: {args.model_name}")

        # Compile model with automatic quantization
        result = compile_model(
            model=args.model,
            preprocessor=preprocessor,
            calibration_data=args.dataset,
            calibration_count=args.calibration_samples,
            model_name=args.model_name,
            description=args.description,
            output_dir=args.output_dir
        )

        log_info("=== Compilation Results ===")
        log_info(f"Model name: {result['model_name']}")
        log_info(f"File size: {result['file_size']} bytes")
        log_info(f"Quantization parameters: {result['quantization_params']}")
        log_info(f"Calibration samples: {result['calibration_samples']}")
        log_info(f"Generated files in: {result['uph5_dir']}")
        log_info("=== Conversion completed successfully! ===")

    except Exception as e:
        log_error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()