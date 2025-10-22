#!/usr/bin/env python3
"""
VWW model compilation with automatic quantization analysis.

This script demonstrates how to use the compile_model() function that automatically
performs quantization analysis and generates parameters in .updlc_cache directory,
eliminating the need for manual preprocessing.
"""

import sys
import os

# Add the updl_compiler to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler import compile_model
from vww_preprocessor import VWWPreprocessor


def main():
    print("Compiling VWW model...")

    # Configuration
    model_path = "ref_model/"
    dataset_dir = "/home/kaiyin-upbeat/data"
    calibration_indices_file = "vww_quantize_calibrate_idxs.txt"

    try:
        # Create preprocessor
        preprocessor = VWWPreprocessor(
            image_size=96
        )

        # Setup calibration data
        calibration_data = {
            'dataset_dir': dataset_dir,
            'indices_file': calibration_indices_file if os.path.exists(calibration_indices_file) else None
        }

        # Compile model
        _ = compile_model(
            model=model_path,
            preprocessor=preprocessor,
            calibration_data=calibration_data,
            model_name="vww_uph5_model",
            description="no_description"
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())