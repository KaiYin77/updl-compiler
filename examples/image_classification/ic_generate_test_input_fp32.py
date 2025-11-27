#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for image classification test inputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ic_preprocessor import ICPreprocessor, CIFAR10_LABELS
from updl_compiler.test import (
    GenerationConfig,
    generate_test_inputs_fp32,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path("/home/kaiyin-upbeat/data/cifar-10-batches-py")

ARRAY_NAME = "g_ic_test_inputs_fp32"
OUTER_DIM_TOKEN = "kNumIcTestInputs"
INNER_DIM_TOKEN = "kIcInputSize"
SAMPLE_COUNT = 5
RANDOM_SEED = 1234

FLOAT_CONFIG = GenerationConfig(
    dataset_dir=DATASET_DIR,
    quant_params_path=EXAMPLE_DIR / ".updlc_cache" / "unused_fp32_params.json",
    output_c_path=EXAMPLE_DIR / "uph5" / "ic_test_inputs_fp32.c",
    dataset_name="cifar10",
    sample_count=SAMPLE_COUNT,
    random_seed=RANDOM_SEED,
    array_name=ARRAY_NAME,
    outer_dim_token=OUTER_DIM_TOKEN,
    inner_dim_token=INNER_DIM_TOKEN,
    include_directive='#include "ic_test_inputs_fp32.h"',
    output_header_path=EXAMPLE_DIR / "uph5" / "ic_test_inputs_fp32.h",
    header_guard="IC_TEST_INPUTS_FP32_H",
    element_type="float",
    header_includes=("#include <stddef.h>",),
    values_per_line=8,
)


def ic_label_extractor(sample: dict) -> str:
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(CIFAR10_LABELS):
        return CIFAR10_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    preprocessor = ICPreprocessor()
    output_path, header_path = generate_test_inputs_fp32(
        FLOAT_CONFIG,
        preprocessor,
        ic_label_extractor,
        license_header=None,
    )

    print(f"Wrote {FLOAT_CONFIG.sample_count} samples to {output_path}")
    print(f"Dataset directory: {FLOAT_CONFIG.dataset_dir}")
    print(f"Header written to {header_path}")


if __name__ == "__main__":
    main()
