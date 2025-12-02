#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Interactive CLI for exporting fp32 image classification layer activations to C arrays."""

from pathlib import Path

from ic_preprocessor import ICPreprocessor, CIFAR10_LABELS
from updl_compiler.test.generation import LayerOutputGenerator, create_model_config


def ic_label_extractor(sample: dict) -> str:
    """Extract label from CIFAR-10 sample."""
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(CIFAR10_LABELS):
        return CIFAR10_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    """Main entry point for IC layer output generation."""
    example_dir = Path(__file__).resolve().parent
    model_path = example_dir / "ref_model"
    dataset_dir = Path("/home/kaiyin-upbeat/data/cifar-10-batches-py")

    # Create model-specific configuration
    config = create_model_config(
        "ic",
        preprocessor_class=ICPreprocessor,
        label_extractor=ic_label_extractor,
        dataset_dir=dataset_dir,
    )

    # Generate layer outputs using the framework
    generator = LayerOutputGenerator(config, example_dir, model_path)
    generator.generate_layer_outputs()


if __name__ == "__main__":
    main()