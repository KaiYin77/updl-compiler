#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Interactive CLI for exporting fp32 keyword spotting layer activations to C arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

from kws_preprocessor import KWSPreprocessor, WORD_LABELS
from updl_compiler.test import (
    GenerationConfig,
    capture_layer_outputs,
    extract_labeled_features,
    find_major_computational_layer,
    get_layer_layout_info,
    list_capture_layers,
    prompt_layer_indices,
    requires_activation_layout_transform,
    sample_tfds_dataset,
    save_results,
    transform_activation_layout,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "ref_model"
OUTPUT_DIR = EXAMPLE_DIR / "uph5"
DATASET_DIR = Path("/home/kaiyin-upbeat/data")
SAMPLE_COUNT = 10
RANDOM_SEED = 1234
ARRAY_PREFIX = "kws_test_layers_fp32"
OUTER_DIM_TOKEN = "kNumKwsTestInputs"
console = Console()

LAYER_CONFIG = GenerationConfig(
    dataset_dir=DATASET_DIR,
    quant_params_path=EXAMPLE_DIR / ".updlc_cache" / "unused_fp32_params.json",
    output_c_path=OUTPUT_DIR / "unused.c",
    dataset_name="speech_commands",
    sample_count=SAMPLE_COUNT,
    random_seed=RANDOM_SEED,
)


def kws_label_extractor(sample: dict) -> str:
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(WORD_LABELS):
        return WORD_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def prompt_layout_selection() -> str:
    """Prompt user to choose between TensorFlow and UPDL layout formats."""
    layout_options = {
        'tf': 'TensorFlow native layout (HWIO weights, NHWC activations)',
        'updl': 'UPDL C-optimized layout (OIHW weights, NCHW activations)'
    }

    console.print("\n[bold cyan]Available Layout Formats:[/]")
    for key, description in layout_options.items():
        console.print(f"  [bold]{key.upper()}[/]: {description}")

    while True:
        choice = console.input("\n[bold]Select layout format (tf/updl)[/]: ").strip().lower()
        if choice in layout_options:
            return choice
        elif choice == '':
            console.print("[yellow]Defaulting to TensorFlow layout[/]")
            return 'tf'
        else:
            console.print(f"[red]Invalid choice '{choice}'. Please enter 'tf' or 'updl'[/]")




def collect_features_and_labels() -> tuple[np.ndarray, List[str]]:
    """Sample TFDS data using the default config and return float features + labels."""
    preprocessor = KWSPreprocessor()
    raw_samples: List[dict] = sample_tfds_dataset(LAYER_CONFIG)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        kws_label_extractor,
    )

    features = np.concatenate([feat for _, feat in labeled_features], axis=0)
    labels = [label for label, _ in labeled_features]
    return features, labels


def main() -> None:
    console.rule("[bold green]KWS Layer Activation Exporter (fp32)")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    layers = list_capture_layers(model)
    layer_names = [layer.name for layer in layers]

    features, labels = collect_features_and_labels()
    console.print(
        f"Loaded [bold]{features.shape[0]}[/] samples with feature shape [bold]{features.shape[1:]}[/]",
        style="green",
    )

    # Interactive layout selection
    layout = prompt_layout_selection()
    console.print(
        f"\nUsing [bold]{layout.upper()}[/] layout format",
        style="cyan",
    )

    table = Table(title="Available Layers", show_lines=False)
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Layer Name", style="magenta")
    for idx, name in enumerate(layer_names):
        table.add_row(f"{idx:02d}", name)
    console.print(table)

    try:
        selected_indices = prompt_layer_indices(
            layer_names,
            input_fn=console.input,
            print_fn=console.print,
            show_layer_list=False,
            prompt_text="[bold]Selection[/]: ",
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        return

    if not selected_indices:
        console.print("No layers selected; exiting.", style="yellow")
        return

    chosen_layers = [layers[idx] for idx in selected_indices]
    chosen_names = [layer.name for layer in chosen_layers]
    console.print(
        "\nGenerating outputs for: [bold]" + ", ".join(chosen_names) + "[/]",
        style="cyan",
    )
    activations = capture_layer_outputs(model, features, chosen_layers)

    # Analyze layout requirements for selected layers
    layout_info = get_layer_layout_info(model, selected_indices, layout)

    if layout == "updl":
        console.print("Analyzing UPDL layout requirements...", style="yellow")
        for idx in selected_indices:
            info = layout_info[idx]
            layer_name = info['layer_name']
            layer_type = info['layer_type']
            major_layer_type = info['major_layer_type']
            requires_transformation = info['requires_transformation']

            if layer_type != major_layer_type:
                # This is a dependent layer
                if requires_transformation:
                    console.print(
                        f"  • {layer_name} ({layer_type}): Inherits {major_layer_type} UPDL layout",
                        style="cyan"
                    )
                else:
                    console.print(
                        f"  • {layer_name} ({layer_type}): Follows {major_layer_type} (no transformation)",
                        style="dim"
                    )
            else:
                # This is a major computational layer
                if requires_transformation:
                    console.print(
                        f"  • {layer_name} ({layer_type}): UPDL layout transformation required",
                        style="green"
                    )
                else:
                    console.print(
                        f"  • {layer_name} ({layer_type}): No layout transformation needed",
                        style="dim"
                    )

        # Transform activation layouts from TF (NHWC) to UPDL (NCHW)
        console.print("Transforming activation layouts from NHWC to NCHW...", style="yellow")
        transformed_activations = []
        for i, (layer, activation) in enumerate(zip(chosen_layers, activations)):
            layer_type = layer.__class__.__name__
            major_layer_type = find_major_computational_layer(model, selected_indices[i])

            # Check if this layer needs activation layout transformation
            needs_transform = requires_activation_layout_transform(major_layer_type)

            if needs_transform:
                original_shape = activation.shape
                transformed_activation = transform_activation_layout(activation, major_layer_type, layout)
                console.print(
                    f"  • {layer.name}: {original_shape} → {transformed_activation.shape} (NHWC→NCHW)",
                    style="green"
                )
                transformed_activations.append(transformed_activation)
            else:
                console.print(
                    f"  • {layer.name}: {activation.shape} (no transformation needed)",
                    style="dim"
                )
                transformed_activations.append(activation)

        activations = transformed_activations

    save_results(
        chosen_names,
        activations,
        labels,
        base_config=LAYER_CONFIG,
        output_dir=OUTPUT_DIR,
        array_prefix=ARRAY_PREFIX,
        outer_dim_token=OUTER_DIM_TOKEN,
        console=console,
        array_name_prefix="g_kws",
        array_name_suffix="_fp32",
        inner_dim_prefix="kKws",
        header_includes=(
            # '#include "kws_test_input_data_fp32.h"',
            "#include <stddef.h>",
        ),
        element_type="float",
        values_per_line=8,
        license_header=None,
    )

if __name__ == "__main__":
    main()
