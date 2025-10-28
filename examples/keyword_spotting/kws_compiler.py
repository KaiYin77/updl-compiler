#!/usr/bin/env python3
"""
KWS model compilation with automatic quantization analysis.

This script demonstrates how to use the compile_model() function that automatically
performs quantization analysis and generates parameters in .updlc_cache directory,
eliminating the need for manual preprocessing.
"""

import sys
import os

# Add the updl_compiler to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler import compile_model
from kws_preprocessor import KWSPreprocessor


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KWS Model Compiler')
    parser.add_argument('--format', choices=['uph5', 'flatbuffers'], default='uph5',
                      help='Output format: uph5 (default) or flatbuffers')
    parser.add_argument('--model-name', default='kws_uph5_model',
                      help='Model name for output files')
    args = parser.parse_args()

    print(f"Compiling KWS model in {args.format} format...")

    # Configuration
    model_path = "ref_model/"
    dataset_dir = "/home/kaiyin-upbeat/data"
    calibration_indices_file = "kws_quantize_calibrate_idxs.txt"

    try:
        # Create preprocessor
        preprocessor = KWSPreprocessor(
            sample_rate=16000,
            clip_duration_ms=1000,
            window_size_ms=30,
            window_stride_ms=20,
            dct_coefficient_count=10
        )

        # Setup calibration data
        calibration_data = {
            'dataset_dir': dataset_dir,
            'indices_file': calibration_indices_file if os.path.exists(calibration_indices_file) else None
        }

        # Compile model with selected format
        result = compile_model(
            model=model_path,
            preprocessor=preprocessor,
            calibration_data=calibration_data,
            model_name=args.model_name,
            description="no_description",
            format=args.format
        )

        print(f"✅ Compilation successful!")
        print(f"   Format: {result['format']}")
        print(f"   Model: {result['model_name']}")
        print(f"   Size: {result['file_size']} bytes")
        print(f"   Output: {result['output_format_dir']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())