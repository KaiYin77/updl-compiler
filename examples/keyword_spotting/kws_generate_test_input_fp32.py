#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Generate float32 C arrays for keyword spotting test inputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test.generation import (
    GenerationConfig,
    generate_test_inputs_fp32,
    collect_features_and_labels,
    prompt_quantization_cycle_choice,
    prompt_quantization_params,
    apply_quantization_cycle,
    write_c_array,
    write_header_file,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path("/home/kaiyin-upbeat/data")

console = Console()

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
    console.rule("[bold green]KWS Test Input Generator (fp32 with Quantization)")

    preprocessor = KWSPreprocessor()

    # Collect features and labels first
    features, labels = collect_features_and_labels(
        FLOAT_CONFIG, preprocessor, kws_label_extractor
    )
    console.print(
        f"Loaded [bold]{features.shape[0]}[/] samples with feature shape [bold]{features.shape[1:]}[/]",
        style="green",
    )

    # Ask user if they want to apply quantization cycle
    apply_quantization = prompt_quantization_cycle_choice(console)

    if apply_quantization:
        # Prompt for quantization parameters
        try:
            quant_params = prompt_quantization_params(console, EXAMPLE_DIR)
            console.print(
                f"\nApplying quantization cycle: scale={quant_params['scale']}, zero_point={quant_params['zero_point']}",
                style="yellow",
            )

            # Apply quantization cycle to input features
            console.print("Processing input features through quantization cycle...", style="cyan")
            features = apply_quantization_cycle(features, quant_params['scale'], quant_params['zero_point'], console)
            console.print("✓ Input quantization cycle completed", style="green")

        except (ValueError, KeyboardInterrupt):
            console.print("[red]Error:[/] Invalid quantization parameters. Exiting.", style="red")
            return
    else:
        console.print("[cyan]Skipping quantization cycle - using original fp32 features[/]", style="cyan")

    # Save the processed features (quantized or original)
    # Prepare flat samples from features
    flat_samples = [sample.flatten() for sample in features]
    input_size = flat_samples[0].size

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

    feature_type = "quantized" if apply_quantization else "original fp32"
    console.print(f"Wrote {FLOAT_CONFIG.sample_count} {feature_type} samples to {output_path}", style="green")
    console.print(f"Dataset directory: {FLOAT_CONFIG.dataset_dir}", style="cyan")
    console.print(f"Header written to {header_path}", style="green")


if __name__ == "__main__":
    main()
