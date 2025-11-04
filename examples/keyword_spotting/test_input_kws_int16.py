#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate int16-quantized C arrays for keyword spotting test inputs."""

from pathlib import Path
from typing import List

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test import (
    GenerationConfig,
    build_config_with_tokens,
    extract_labeled_features,
    load_input_scale_zero_point,
    quantize_features,
    sample_tfds_dataset,
    write_c_array,
    write_header_file,
)

EXAMPLE_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = build_config_with_tokens(
    base_dir=EXAMPLE_DIR,
    dataset_dir=Path("/home/kaiyin-upbeat/data"),
    model_token="kws",
    backend_token="uph5",
    dtype_token="int16",
)


def kws_label_extractor(sample: dict) -> str:
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(WORD_LABELS):
        return WORD_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main(config: GenerationConfig = DEFAULT_CONFIG) -> None:
    scale, zero_point = load_input_scale_zero_point(config)

    preprocessor = KWSPreprocessor()
    raw_samples: List[dict] = sample_tfds_dataset(config)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        kws_label_extractor,
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
    print(f"Dataset directory: {config.dataset_dir}")
    print(f"Header written to {header_path}")


if __name__ == "__main__":
    main()
