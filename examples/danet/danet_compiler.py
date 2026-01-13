#!/usr/bin/env python3
"""
DANet model compilation with automatic quantization analysis.

This script demonstrates how to use the compile_model() function that automatically
performs quantization analysis and generates parameters in .updlc_cache directory,
eliminating the need for manual preprocessing.
"""

import sys
import os

# Add the updl_compiler to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler import compile_model
from danet_preprocessor import DANetPreprocessor


def main():
    print("Compiling DANet model...")

    # Configuration
    model_path = "ref_model/"
    dataset_path = "../../references/ready_for_training.csv"
    calibration_indices_file = "danet_quantize_calibrate_idxs.txt"

    try:
        # Create preprocessor
        preprocessor = DANetPreprocessor(
            acc_factor=9.81,
            gyro_factor=4.0,
            diff_factor=5.0,
            lpf_alpha_rise=0.5,
            lpf_alpha_fall=0.01
        )

        # Setup calibration data
        calibration_data = {
            'dataset_path': dataset_path,
            'indices_file': calibration_indices_file if os.path.exists(calibration_indices_file) else None
        }

        # Compile model
        _ = compile_model(
            model=model_path,
            preprocessor=preprocessor,
            calibration_data=calibration_data,
            model_name="danet_uph5_model",
            description="DANet model for adaptive IMU sensor fusion"
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())