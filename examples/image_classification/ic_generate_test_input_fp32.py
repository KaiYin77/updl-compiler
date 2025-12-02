#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for image classification test inputs."""

from pathlib import Path

from ic_preprocessor import ICPreprocessor, CIFAR10_LABELS
from updl_compiler.test.generation import TestInputGenerator, create_model_config


def ic_label_extractor(sample: dict) -> str:
    """Extract label from CIFAR-10 sample."""
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(CIFAR10_LABELS):
        return CIFAR10_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    """Main entry point for IC test input generation."""
    example_dir = Path(__file__).resolve().parent
    dataset_dir = Path("/home/kaiyin-upbeat/data/cifar-10-batches-py")

    # Create model-specific configuration
    config = create_model_config(
        "ic",
        preprocessor_class=ICPreprocessor,
        label_extractor=ic_label_extractor,
        dataset_dir=dataset_dir,
    )

    # Generate test inputs using the framework
    generator = TestInputGenerator(config, example_dir)
    output_path, header_path = generator.generate_test_inputs()

    print(f"\n✓ Generated: {output_path}")
    print(f"✓ Header: {header_path}")


if __name__ == "__main__":
    main()