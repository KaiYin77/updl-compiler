#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate int16-quantized C arrays for DANet test inputs."""

from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from danet_preprocessor import DANetPreprocessor
from updl_compiler.test import (
    GenerationConfig,
    extract_labeled_features,
    load_input_scale_zero_point,
    quantize_features,
    write_c_array,
    write_header_file,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXAMPLE_DIR / "uph5"
ARRAY_NAME = "g_danet_test_inputs_int16"
OUTER_DIM_TOKEN = "kNumDanetTestInputs"
INNER_DIM_TOKEN = "kDanetInputSize"

DEFAULT_CONFIG = GenerationConfig(
    dataset_dir=Path("/home/kaiyin-upbeat/data"),
    quant_params_path=EXAMPLE_DIR / ".updlc_cache" / "danet_uph5_model_quantize_params.json",
    output_c_path=BACKEND_DIR / "danet_test_inputs_int16.c",
    dataset_name="imu_data",  # Custom dataset name for DANet
    sample_count=1,  # Only 1 test case as requested
    random_seed=1234,
    array_name=ARRAY_NAME,
    outer_dim_token=OUTER_DIM_TOKEN,
    inner_dim_token=INNER_DIM_TOKEN,
    include_directive='#include "danet_test_inputs_int16.h"',
    output_header_path=BACKEND_DIR / "danet_test_inputs_int16.h",
    header_guard="DANET_TEST_INPUTS_INT16_H",
)


def danet_label_extractor(sample: dict) -> str:
    """Extract label from DANet sample - use sample index as label."""
    return f"imu_sample_{sample.get('index', 0)}"


def sample_imu_dataset(config: GenerationConfig) -> List[dict]:
    """Sample IMU data from the CSV file."""
    # Use the reference CSV file
    csv_path = EXAMPLE_DIR.parent.parent / "references" / "ready_for_training.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at {csv_path}")

    # Load IMU dataset
    df = pd.read_csv(csv_path)

    # Sample the first row as our test case
    sample_idx = 0
    row = df.iloc[sample_idx]

    # Convert to the expected format
    sample = {
        'ax': float(row['ax']),
        'ay': float(row['ay']),
        'az': float(row['az']),
        'gx': float(row['gx']),
        'gy': float(row['gy']),
        'gz': float(row['gz']),
        'dt': 0.005,  # Default 200Hz sampling rate
        'index': sample_idx
    }

    return [sample]


def main(config: GenerationConfig = DEFAULT_CONFIG) -> None:
    # Create uph5 directory if it doesn't exist
    BACKEND_DIR.mkdir(exist_ok=True)

    scale, zero_point = load_input_scale_zero_point(config)

    preprocessor = DANetPreprocessor(
        acc_factor=9.81,
        gyro_factor=4.0,
        diff_factor=5.0,
        lpf_alpha_rise=0.5,
        lpf_alpha_fall=0.01
    )

    # Sample IMU data instead of using TFDS
    raw_samples: List[dict] = sample_imu_dataset(config)

    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        danet_label_extractor,
    )
    quantized_samples, labels = quantize_features(labeled_features, scale, zero_point)

    if not quantized_samples:
        raise RuntimeError("No samples were quantized; check dataset configuration.")

    input_size = quantized_samples[0].size
    if any(vec.size != input_size for vec in quantized_samples[1:]):
        raise RuntimeError("Quantized samples have inconsistent sizes.")

    output_path = write_c_array(
        config,
        quantized_samples,
        labels,
        input_size,
    )
    header_path = write_header_file(
        config,
        len(quantized_samples),
        input_size,
    )

    print(f"Wrote {len(quantized_samples)} samples to {output_path}")
    print(f"Input size per sample: {input_size}")
    print(f"Quantization scale: {scale}, zero_point: {zero_point}")
    print(f"CSV data source: {EXAMPLE_DIR.parent.parent / 'references' / 'ready_for_training.csv'}")
    print(f"Header written to {header_path}")


if __name__ == "__main__":
    main()