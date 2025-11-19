#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for keyword spotting test inputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test import (
    GenerationConfig,
    extract_labeled_features,
    sample_tfds_dataset,
    write_c_array,
    write_header_file,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path("/home/kaiyin-upbeat/data")

ARRAY_NAME = "g_kws_test_inputs_fp32"
OUTER_DIM_TOKEN = "kNumKwsTestInputs"
INNER_DIM_TOKEN = "kKwsInputSize"
SAMPLE_COUNT = 10
RANDOM_SEED = 1234

FLOAT_CONFIG = GenerationConfig(
    dataset_dir=DATASET_DIR,
    quant_params_path=EXAMPLE_DIR / ".updlc_cache" / "unused_fp32_params.json",
    output_c_path=EXAMPLE_DIR / "uph5" / "kws_test_inputs_fp32.c",
    dataset_name="speech_commands",
    sample_count=SAMPLE_COUNT,
    random_seed=RANDOM_SEED,
    array_name=ARRAY_NAME,
    outer_dim_token=OUTER_DIM_TOKEN,
    inner_dim_token=INNER_DIM_TOKEN,
    include_directive='#include "kws_test_inputs_fp32.h"',
    output_header_path=EXAMPLE_DIR / "uph5" / "kws_test_inputs_fp32.h",
    header_guard="KWS_TEST_INPUTS_FP32_H",
    element_type="float",
    header_includes=("#include <stddef.h>",),
    values_per_line=8,
)


def kws_label_extractor(sample: dict) -> str:
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(WORD_LABELS):
        return WORD_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    preprocessor = KWSPreprocessor()
    raw_samples: List[dict] = sample_tfds_dataset(FLOAT_CONFIG)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        kws_label_extractor,
    )

    if not labeled_features:
        raise RuntimeError("No samples were collected; check dataset configuration.")

    labels = []
    flat_samples = []
    for label, features in labeled_features:
        labels.append(label)
        flat_samples.append(features.astype("float32", copy=False).flatten())

    input_size = flat_samples[0].size
    if any(sample.size != input_size for sample in flat_samples[1:]):
        raise RuntimeError("Float samples have inconsistent sizes.")

    output_path = write_c_array(
        FLOAT_CONFIG,
        flat_samples,
        labels,
        input_size,
        license_header=None,
    )
    header_path = write_header_file(
        FLOAT_CONFIG,
        len(flat_samples),
        input_size,
        license_header=None,
    )

    print(f"Wrote {len(flat_samples)} samples to {output_path}")
    print(f"Input size per sample: {input_size}")
    print(f"Dataset directory: {FLOAT_CONFIG.dataset_dir}")
    print(f"Header written to {header_path}")


if __name__ == "__main__":
    main()
