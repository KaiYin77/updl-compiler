#!/usr/bin/env python
"""
Script to convert WAV files to C array format for KWS model testing using proper int16 quantization.
Based on wav_to_c_array.py but using UPDL quantization method with input_scale = 0.00377110f.
"""

import tensorflow as tf
import numpy as np
import os
import argparse
import json

# Import common KWS utilities
from kws_common import (
    convert_wav_to_mfcc_features_tensor,
    convert_wav_to_tensor,
    parse_command,
)


def load_quantization_params(json_file):
    """Load quantization parameters from JSON file"""
    try:
        with open(json_file, "r") as f:
            params = json.load(f)
        return params["input"]["scale"], params["input"]["zero_point"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading quantization parameters: {e}")
        return None, None


def wav_to_c_array_int16(wav_file, flags, output_file=None, json_params_file=None):
    """Convert WAV file to properly quantized int16 MFCC features and output as C array"""

    # Load audio
    wav = convert_wav_to_tensor(wav_file, flags)
    if wav is None:
        print("Failed to load audio file")
        return None

    # Extract MFCC features
    features = convert_wav_to_mfcc_features_tensor(wav, flags)

    # Quantize features to int16 with proper rounding (UPDL method)
    features_np = features.numpy()

    # Load quantization parameters from JSON file or calculate dynamically
    if json_params_file and os.path.exists(json_params_file):
        input_scale, zero_point = load_quantization_params(json_params_file)
        if input_scale is not None:
            print(f"Using quantization parameters from {json_params_file}")
            print(f"Input scale: {input_scale}, zero_point: {zero_point}")
        else:
            print("Failed to load quantization parameters, calculating dynamically")
            input_scale = None
    else:
        print(
            f"JSON params file not found: {json_params_file}, calculating dynamically"
        )
        input_scale = None

    # Fall back to dynamic calculation if JSON params not available
    if input_scale is None:
        max_abs_value = np.max(np.abs(features_np))
        if max_abs_value == 0:
            input_scale = 1.0  # Avoid division by zero
        else:
            input_scale = max_abs_value / (32767)
        zero_point = 0
        print(f"Calculated input_scale: {input_scale}, zero_point: {zero_point}")

    # Apply UPDL quantization: quantized_value = round(original_float / scale) + zero_point
    quantized_float = features_np / input_scale + zero_point
    features_q = np.round(quantized_float).astype(np.int16)

    # Verify no overflow occurred
    if np.any(np.abs(features_q) > 32767):
        print(
            f"WARNING: Overflow detected! Max quantized value: {np.max(np.abs(features_q))}"
        )
        # Adjust scale if needed
        actual_max = np.max(np.abs(features_q))
        input_scale = input_scale * (actual_max / 32767)
        quantized_float = features_np / input_scale + zero_point
        features_q = np.round(quantized_float).astype(np.int16)
        print(f"Adjusted input_scale to: {input_scale}")

    # Debug: Show quantization process for first 5 values
    print(f"[QUANTIZATION] Original float (first 5): {features_np.flatten()[:5]}")
    print(f"[QUANTIZATION] After scale division: {quantized_float.flatten()[:5]}")
    print(f"[QUANTIZATION] After rounding to int16: {features_q.flatten()[:5]}")
    print(
        f"[QUANTIZATION] As hex: {[hex(x & 0xFFFF) for x in features_q.flatten()[:5]]}"
    )

    # Verify round-trip conversion
    reconstructed_float = (features_q.flatten()[:5] - zero_point) * input_scale
    print(f"[VERIFICATION] Reconstructed float: {reconstructed_float}")
    print(
        f"[VERIFICATION] Quantization error: {reconstructed_float - features_np.flatten()[:5]}"
    )

    # Flatten the array
    flat_features = features_q.flatten()

    # Get the word from filename
    word = os.path.splitext(os.path.basename(wav_file))[0].lower()

    # Generate C array
    c_array_size = len(flat_features)

    # Create C array string in the requested format with proper copyright header
    c_code = f"""/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kws_mock_input_data_int16.h"
// {word}
const int16_t g_kws_inputs_int16[kNumKwsTestInputs][kKwsInputSize] = {{
    {{
"""

    # Add array elements as signed decimal values (8 per line for int16)
    for i in range(0, len(flat_features), 8):
        line_elements = flat_features[i : i + 8]
        line_str = "        " + ", ".join(f"{int(val)}" for val in line_elements)
        if i + 8 < len(flat_features):
            line_str += ","
        c_code += line_str + "\n"

    c_code += """    }
};
"""

    # Output to file or stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write(c_code)
        print(f"C array written to {output_file}")
    else:
        print(c_code)

    print(f"Array size: {c_array_size}")
    print(f"Feature shape: {features_q.shape}")
    print(
        f"UPDL input scale: {input_scale} (symmetric quantization, zero_point={zero_point})"
    )

    return c_code


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Convert WAV files to int16 C arrays for KWS testing using UPDL quantization"
    )
    parser.add_argument("--wav_file", help="Path to WAV file to convert")
    parser.add_argument(
        "--output",
        "-o",
        default="../c/src/kws/kws_input_data_int16.c",
        help="Output C file (default: stdout)",
    )
    parser.add_argument(
        "--json_params",
        default="artifacts/kws_quantize_int16.json",
        help="Path to JSON file with quantization parameters (default: artifacts/kws_quantize_int16.json)",
    )

    args = parser.parse_args()

    # Get default flags from kws_common
    flags, _ = parse_command()

    # Convert WAV to C array using quantization parameters from JSON
    wav_to_c_array_int16(args.wav_file, flags, args.output, args.json_params)


if __name__ == "__main__":
    main()
