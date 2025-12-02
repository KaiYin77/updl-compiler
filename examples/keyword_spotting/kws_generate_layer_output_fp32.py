#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Interactive CLI for exporting fp32 keyword spotting layer activations to C arrays."""

from pathlib import Path

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test.generation import LayerOutputGenerator, create_model_config


def kws_label_extractor(sample: dict) -> str:
    """Extract label from speech command sample."""
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(WORD_LABELS):
        return WORD_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    """Main entry point for KWS layer output generation."""
    example_dir = Path(__file__).resolve().parent
    model_path = example_dir / "ref_model"
    dataset_dir = Path("/home/kaiyin-upbeat/data")

    # Create model-specific configuration
    config = create_model_config(
        "kws",
        preprocessor_class=KWSPreprocessor,
        label_extractor=kws_label_extractor,
        dataset_dir=dataset_dir,
    )

    # Generate layer outputs using the framework
    generator = LayerOutputGenerator(config, example_dir, model_path)
    generator.generate_layer_outputs()


if __name__ == "__main__":
    main()