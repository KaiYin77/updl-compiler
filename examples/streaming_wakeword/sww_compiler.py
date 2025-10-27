#!/usr/bin/env python3
"""
Streaming Wake Word Model Compiler

Compiles the trained streaming wake word model to UPH5 format for embedded deployment.
"""
import sys
import os

# Add the updl_compiler to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from updl_compiler import compile_model
from sww_preprocessor import SWWPreprocessor


def main():
    print("Compiling sww model...")

    # Configuration
    model_path = "ref_model/"
    dataset_dir = "/home/kaiyin-upbeat/data/speech_commands/0.0.2"
    calibration_indices_file = "sww_quantize_calibrate_idxs.txt"

    try:
        # Create preprocessor
        preprocessor = SWWPreprocessor()

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
            model_name="sww_uph5_model",
            description="no_description"
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())