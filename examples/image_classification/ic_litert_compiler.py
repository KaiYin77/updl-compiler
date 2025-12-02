#!/usr/bin/env python3
"""
Image Classification LiteRT Compiler: End-to-end pipeline from model quantization to C array generation.

This script combines model quantization and C array generation into a single pipeline,
taking a trained Keras model and producing a quantized TFLite model and corresponding
C array files for embedded deployment.

Based on TinyMLPerf image classification benchmark and UPDL compiler patterns.
"""

import tensorflow as tf
import os
import sys
import subprocess
import numpy as np
from pathlib import Path


class ICLiteRTCompiler:
    def __init__(self, model_path, output_dir="./", model_name="ic_model"):
        """
        Initialize the Image Classification LiteRT Compiler.

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

    def load_calibration_data(self, calibration_samples_file="ic_quantize_calibrate_idxs.txt",
                            cifar_10_dir="cifar-10-batches-py"):
        """
        Load calibration data for quantization.

        Args:
            calibration_samples_file (str): Path to calibration sample indices
            cifar_10_dir (str): Path to CIFAR-10 data directory

        Returns:
            function: Representative dataset generator
        """
        if os.path.exists(calibration_samples_file):
            if calibration_samples_file.endswith('.txt'):
                # Load from text file (one index per line)
                with open(calibration_samples_file, 'r') as f:
                    cal_indices = [int(line.strip()) for line in f if line.strip()]
                cal_indices = np.array(cal_indices)
            else:
                # Load from numpy file
                cal_indices = np.load(calibration_samples_file)
        else:
            print(f"Warning: {calibration_samples_file} not found. Using default calibration.")
            cal_indices = np.arange(100)  # Default to first 100 samples

        def representative_dataset_generator():
            # Load CIFAR-10 test data if available
            if os.path.exists(cifar_10_dir):
                try:
                    # Try to import and use CIFAR-10 data loading
                    import train
                    train_data, _, _, test_data, _, _, _ = train.load_cifar_10_data(cifar_10_dir)
                    for i in cal_indices:
                        sample_img = np.expand_dims(np.array(test_data[i], dtype=np.float32), axis=0)
                        yield [sample_img]
                except (ImportError, AttributeError):
                    print("Warning: Could not load CIFAR-10 data. Using dummy calibration data.")
                    for _ in cal_indices:
                        # Generate dummy CIFAR-10 sized images (32x32x3)
                        yield [np.random.randn(1, 32, 32, 3).astype(np.float32)]
            else:
                print(f"Warning: CIFAR-10 directory {cifar_10_dir} not found. Using dummy calibration data.")
                for _ in cal_indices:
                    # Generate dummy CIFAR-10 sized images (32x32x3)
                    yield [np.random.randn(1, 32, 32, 3).astype(np.float32)]

        return representative_dataset_generator

    def quantize_model(self, calibration_samples_file="ic_quantize_calibrate_idxs.txt",
                      cifar_10_dir="cifar-10-batches-py"):
        """
        Quantize the Keras model to int8 TFLite format.

        Args:
            calibration_samples_file (str): Path to calibration sample indices
            cifar_10_dir (str): Path to CIFAR-10 data directory
        """
        print(f"Loading model from {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)

        # Create representative dataset
        representative_dataset_gen = self.load_calibration_data(
            calibration_samples_file, cifar_10_dir)

        # Convert to quantized int8 model
        print(f"Converting to quantized int8 TFLite: {self.tflite_file}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Set optimization flags for int8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset_gen
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the model
        tflite_quant_model = converter.convert()

        # Save the quantized model
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

            # Apply transformations for proper C array format
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

    def compile(self, calibration_samples_file="ic_quantize_calibrate_idxs.txt",
                cifar_10_dir="cifar-10-batches-py"):
        """
        Complete compilation pipeline: quantization + C array generation.

        Args:
            calibration_samples_file (str): Path to calibration sample indices
            cifar_10_dir (str): Path to CIFAR-10 data directory

        Returns:
            dict: Paths to generated files
        """
        print("=== Image Classification LiteRT Compilation Pipeline ===")

        # Step 1: Quantize model
        tflite_path = self.quantize_model(calibration_samples_file, cifar_10_dir)

        # Step 2: Generate C array
        self.generate_c_array()

        print("=== Compilation Complete ===")

        return {
            'tflite_quantized': str(self.tflite_file),
            'c_data_file': str(self.c_data_file),
            'c_header_file': str(self.c_header_file)
        }


def main():
    # Configuration
    model_path = "ref_model/"
    dataset_dir = "/home/kaiyin-upbeat/data"
    calibration_indices_file = "ic_quantize_calibrate_idxs.txt"
    cifar_10_dir = f"{dataset_dir}/cifar-10-batches-py"
    output_dir = "./litert"
    model_name = "ic_model"

    try:
        # Create compiler instance
        compiler = ICLiteRTCompiler(
            model_path=model_path,
            output_dir=output_dir,
            model_name=model_name
        )

        # Run compilation pipeline
        output_files = compiler.compile(
            calibration_samples_file=calibration_indices_file,
            cifar_10_dir=cifar_10_dir
        )

        print("\nGenerated files:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type}: {file_path}")

    except Exception as e:
        print(f"Compilation failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())