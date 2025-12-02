#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for keyword spotting test inputs."""

from pathlib import Path

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test.generation import TestInputGenerator, create_model_config


def kws_label_extractor(sample: dict) -> str:
    """Extract label from speech command sample."""
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(WORD_LABELS):
        return WORD_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    """Main entry point for KWS test input generation."""
    example_dir = Path(__file__).resolve().parent
    dataset_dir = Path("/home/kaiyin-upbeat/data")

    # Create model-specific configuration
    config = create_model_config(
        "kws",
        preprocessor_class=KWSPreprocessor,
        label_extractor=kws_label_extractor,
        dataset_dir=dataset_dir,
    )

    # Generate test inputs using the framework
    generator = TestInputGenerator(config, example_dir)
    output_path, header_path = generator.generate_test_inputs()

    print(f"\n✓ Generated: {output_path}")
    print(f"✓ Header: {header_path}")


if __name__ == "__main__":
    main()