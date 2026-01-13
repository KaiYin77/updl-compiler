#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for DANet test inputs."""

from pathlib import Path
import pandas as pd
from typing import List

from danet_preprocessor import DANetPreprocessor
from updl_compiler.test.generation import TestInputGenerator, create_model_config


def danet_label_extractor(sample: dict) -> str:
    """Extract label from DANet IMU sample."""
    return f"imu_sample_{sample.get('index', 0)}"


def danet_data_loader(dataset_dir: Path, sample_count: int = 1, random_seed: int = 1234) -> List[dict]:
    """Load IMU data from CSV file instead of TFDS."""
    # Use the reference CSV file
    csv_path = Path(__file__).parent.parent.parent / "references" / "ready_for_training.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at {csv_path}")

    # Load IMU dataset
    df = pd.read_csv(csv_path)

    # Sample the first row as our test case
    samples = []
    for i in range(min(sample_count, len(df))):
        row = df.iloc[i]
        sample = {
            'ax': float(row['ax']),
            'ay': float(row['ay']),
            'az': float(row['az']),
            'gx': float(row['gx']),
            'gy': float(row['gy']),
            'gz': float(row['gz']),
            'dt': 0.005,  # Default 200Hz sampling rate
            'index': i
        }
        samples.append(sample)

    return samples


def main() -> None:
    """Main entry point for DANet test input generation."""
    example_dir = Path(__file__).resolve().parent
    dataset_dir = Path("/home/kaiyin-upbeat/data")

    # Create model-specific configuration for DANet
    config = create_model_config(
        "danet",
        preprocessor_class=DANetPreprocessor,
        label_extractor=danet_label_extractor,
        dataset_dir=dataset_dir,
        sample_count=1,  # Only 1 test case as requested
        data_loader=danet_data_loader,  # Custom data loader for IMU CSV data
    )

    # Generate test inputs using the framework
    generator = TestInputGenerator(config, example_dir)
    output_path, header_path = generator.generate_test_inputs()

    print(f"\n✓ Generated: {output_path}")
    print(f"✓ Header: {header_path}")


if __name__ == "__main__":
    main()