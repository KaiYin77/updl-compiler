#!/usr/bin/env python3
"""
KWS LiteRT Compiler: End-to-end pipeline from model quantization to C array generation.

This script combines model quantization and C array generation into a single pipeline,
taking a trained Keras model and producing a quantized TFLite model and corresponding
C array files for embedded deployment.
"""

import tensorflow as tf
import os
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path


class KWSLiteRTCompiler:
    def __init__(self, model_path, output_dir="./", model_name="kws_model"):
        """
        Initialize the KWS LiteRT Compiler.

        Args:
            model_path (str): Path to the trained Keras model
            output_dir (str): Output directory for generated files
            model_name (str): Base name for output files
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.tflite_file = self.output_dir / f"{model_name}_int8.tflite"
        self.c_data_file = self.output_dir / f"{model_name}_data.cc"
        self.c_header_file = self.output_dir / f"{model_name}_data.h"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_calibration_indices(self, indices_file="quant_cal_idxs.txt"):
        """Load calibration indices from file."""
        if os.path.exists(indices_file):
            with open(indices_file) as fpi:
                cal_indices = [int(line.strip()) for line in fpi if line.strip()]
            return sorted(cal_indices)
        else:
            print(f"Warning: {indices_file} not found. Using default calibration.")
            return list(range(100))  # Default to first 100 samples

    def create_representative_dataset(self, cal_indices, dataset_generator=None):
        """
        Create representative dataset for quantization calibration.

        Args:
            cal_indices (list): List of calibration indices
            dataset_generator: Optional custom dataset generator function
        """
        if dataset_generator is None:
            # Create a dummy dataset generator for demonstration
            # In practice, this should be replaced with actual data loading
            def dummy_generator():
                for _ in range(len(cal_indices)):
                    # Generate dummy audio features (typically 49x10 for KWS)
                    yield [np.random.randn(1, 49, 10, 1).astype(np.float32)]
            return dummy_generator
        else:
            return dataset_generator

    def quantize_model(self, calibration_indices_file="quant_cal_idxs.txt",
                      dataset_generator=None):
        """
        Quantize the Keras model to int8 TFLite format.

        Args:
            calibration_indices_file (str): Path to calibration indices file
            dataset_generator: Optional custom dataset generator for calibration
        """
        print(f"Loading model from {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)

        # Load calibration indices
        cal_indices = self.load_calibration_indices(calibration_indices_file)
        print(f"Using {len(cal_indices)} calibration samples")

        # Create representative dataset
        representative_dataset_gen = self.create_representative_dataset(
            cal_indices, dataset_generator)

        # Convert to quantized int8 model
        print(f"Converting to quantized int8 TFLite: {self.tflite_file}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_quant_model = converter.convert()
        with open(self.tflite_file, "wb") as fpo:
            num_bytes_written = fpo.write(tflite_quant_model)
        print(f"Wrote {num_bytes_written} bytes to quantized int8 tflite file")

        return str(self.tflite_file)

    def generate_c_array(self):
        """
        Generate C array files from the TFLite model using xxd.
        Creates both .cc and .h files.
        """
        if not self.tflite_file.exists():
            raise FileNotFoundError(f"TFLite file not found: {self.tflite_file}")

        print(f"Generating C array from {self.tflite_file}")

        # Use xxd to convert tflite to C array
        try:
            # Generate the C data file
            cmd = ['xxd', '-i', str(self.tflite_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            c_content = result.stdout

            # Apply transformations similar to make_model_c_file
            header_include = f'#include "{self.c_header_file.name}"\n\n'
            c_content = header_include + c_content

            # Replace variable names and types
            tflite_var_name = str(self.tflite_file).replace('/', '_').replace('.', '_')
            new_var_name = f"g_{self.model_name}_data"

            c_content = c_content.replace('unsigned char', 'const unsigned char')
            c_content = c_content.replace('unsigned int', 'const unsigned int')
            c_content = c_content.replace(tflite_var_name, new_var_name)

            # Write the C data file
            with open(self.c_data_file, 'w') as f:
                f.write(c_content)

            # Generate the header file
            header_content = f"""#ifndef {self.model_name.upper()}_MODEL_DATA_H_
#define {self.model_name.upper()}_MODEL_DATA_H_

extern const unsigned char g_{self.model_name}_data[];
extern const unsigned int g_{self.model_name}_data_len;

#endif  // {self.model_name.upper()}_MODEL_DATA_H_
"""

            with open(self.c_header_file, 'w') as f:
                f.write(header_content)

            print(f"Generated C files:")
            print(f"  Data: {self.c_data_file}")
            print(f"  Header: {self.c_header_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error running xxd: {e}")
            raise
        except FileNotFoundError:
            print("Error: xxd command not found. Please install xxd.")
            raise

    def compile(self, calibration_indices_file="quant_cal_idxs.txt",
                dataset_generator=None):
        """
        Complete compilation pipeline: quantization + C array generation.

        Args:
            calibration_indices_file (str): Path to calibration indices file
            dataset_generator: Optional custom dataset generator for calibration

        Returns:
            dict: Paths to generated files
        """
        print("=== KWS LiteRT Compilation Pipeline ===")

        # Step 1: Quantize model
        tflite_path = self.quantize_model(calibration_indices_file, dataset_generator)

        # Step 2: Generate C array
        self.generate_c_array()

        print("=== Compilation Complete ===")

        return {
            'tflite_quantized': str(self.tflite_file),
            'c_data_file': str(self.c_data_file),
            'c_header_file': str(self.c_header_file)
        }


def main():
    parser = argparse.ArgumentParser(description='KWS LiteRT Compiler')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained Keras model')
    parser.add_argument('--output_dir', type=str, default='./litert',
                       help='Output directory for generated files')
    parser.add_argument('--model_name', type=str, default='kws_model',
                       help='Base name for output files')
    parser.add_argument('--calibration_indices', type=str, default='quant_cal_idxs.txt',
                       help='Path to calibration indices file')

    args = parser.parse_args()

    try:
        # Create compiler instance
        compiler = KWSLiteRTCompiler(
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_name=args.model_name
        )

        # Run compilation pipeline
        output_files = compiler.compile(calibration_indices_file=args.calibration_indices)

        print("\nGenerated files:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type}: {file_path}")

    except Exception as e:
        print(f"Compilation failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())