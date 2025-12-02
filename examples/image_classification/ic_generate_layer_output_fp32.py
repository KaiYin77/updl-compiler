#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Interactive CLI for exporting fp32 image classification layer activations to C arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

from ic_preprocessor import ICPreprocessor, CIFAR10_LABELS
from updl_compiler.test.generation import (
    GenerationConfig,
    capture_layer_outputs,
    find_major_computational_layer,
    get_layer_layout_info,
    list_capture_layers,
    prompt_layer_indices,
    requires_activation_layout_transform,
    save_results,
    transform_activation_layout,
    prompt_layout_selection,
    collect_features_and_labels,
    prompt_quantization_cycle_choice,
    prompt_quantization_params,
    apply_quantization_cycle,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "ref_model"
DATASET_DIR = Path("/home/kaiyin-upbeat/data/cifar-10-batches-py")
SAMPLE_COUNT = 5
RANDOM_SEED = 1234
ARRAY_PREFIX = "ic_test_layers_fp32"
OUTER_DIM_TOKEN = "kNumIcTestInputs"
console = Console()




def create_layer_config(output_dir: Path) -> GenerationConfig:
    """Create layer configuration with dynamic output directory."""
    return GenerationConfig(
        dataset_dir=DATASET_DIR,
        quant_params_path=EXAMPLE_DIR / ".updlc_cache" / "unused_fp32_params.json",
        output_c_path=output_dir / "unused.c",
        dataset_name="cifar10",
        sample_count=SAMPLE_COUNT,
        random_seed=RANDOM_SEED,
    )


def ic_label_extractor(sample: dict) -> str:
    label_idx = int(sample["label"].numpy())
    if 0 <= label_idx < len(CIFAR10_LABELS):
        return CIFAR10_LABELS[label_idx].lower()
    return f"label_{label_idx}"


def main() -> None:
    console.rule("[bold green]IC Layer Activation Exporter (fp32)")

    # Interactive layout selection
    layout = prompt_layout_selection(console)
    console.print(
        f"\nUsing [bold]{layout.upper()}[/] layout format",
        style="cyan",
    )

    # Set output directory based on layout choice
    if layout == "updl":
        output_dir = EXAMPLE_DIR / "uph5"
        console.print(
            "Saving to [bold]uph5/[/] directory for UPDL layout", style="cyan"
        )
    else:
        output_dir = EXAMPLE_DIR / "litert"
        console.print(
            "Saving to [bold]litert/[/] directory for TensorFlow layout", style="cyan"
        )

    # Create configuration with appropriate output directory
    layer_config = create_layer_config(output_dir)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    layers = list_capture_layers(model)
    layer_names = [layer.name for layer in layers]

    preprocessor = ICPreprocessor()
    features, labels = collect_features_and_labels(
        layer_config, preprocessor, ic_label_extractor
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
                f"\nApplying quantization cycle to input data: scale={quant_params['scale']}, zero_point={quant_params['zero_point']}",
                style="yellow",
            )

            # Apply fp32 -> int16 -> fp32 quantization cycle to input features
            console.print("Processing input features through quantization cycle...", style="cyan")
            features = apply_quantization_cycle(features, quant_params['scale'], quant_params['zero_point'], console)

            console.print("✓ Input quantization cycle completed", style="green")

        except (ValueError, KeyboardInterrupt):
            console.print("[red]Error:[/] Invalid quantization parameters. Exiting.", style="red")
            return
    else:
        console.print("[cyan]Skipping quantization cycle - using original fp32 features for layer outputs[/]", style="cyan")

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
            layer_name = info["layer_name"]
            layer_type = info["layer_type"]
            major_layer_type = info["major_layer_type"]
            requires_transformation = info["requires_transformation"]

            if layer_type != major_layer_type:
                # This is a dependent layer
                if requires_transformation:
                    console.print(
                        f"  • {layer_name} ({layer_type}): Inherits {major_layer_type} UPDL layout",
                        style="cyan",
                    )
                else:
                    console.print(
                        f"  • {layer_name} ({layer_type}): Follows {major_layer_type} (no transformation)",
                        style="dim",
                    )
            else:
                # This is a major computational layer
                if requires_transformation:
                    console.print(
                        f"  • {layer_name} ({layer_type}): UPDL layout transformation required",
                        style="green",
                    )
                else:
                    console.print(
                        f"  • {layer_name} ({layer_type}): No layout transformation needed",
                        style="dim",
                    )

        # Transform activation layouts from TF (NHWC) to UPDL (NCHW)
        console.print(
            "Transforming activation layouts from NHWC to NCHW...", style="yellow"
        )
        transformed_activations = []
        for i, (layer, activation) in enumerate(zip(chosen_layers, activations)):
            layer_type = layer.__class__.__name__
            major_layer_type = find_major_computational_layer(
                model, selected_indices[i]
            )

            # Check if this layer needs activation layout transformation
            needs_transform = requires_activation_layout_transform(major_layer_type)

            if needs_transform:
                original_shape = activation.shape
                transformed_activation = transform_activation_layout(
                    activation, major_layer_type, layout
                )
                console.print(
                    f"  • {layer.name}: {original_shape} → {transformed_activation.shape} (NHWC→NCHW)",
                    style="green",
                )
                transformed_activations.append(transformed_activation)
            else:
                console.print(
                    f"  • {layer.name}: {activation.shape} (no transformation needed)",
                    style="dim",
                )
                transformed_activations.append(activation)

        activations = transformed_activations

    save_results(
        chosen_names,
        activations,
        labels,
        base_config=layer_config,
        output_dir=output_dir,
        array_prefix=ARRAY_PREFIX,
        outer_dim_token=OUTER_DIM_TOKEN,
        console=console,
        array_name_prefix="g_ic",
        array_name_suffix="_fp32",
        inner_dim_prefix="kIc",
        header_includes=(
            # '#include "ic_test_input_data_fp32.h"',
            "#include <stddef.h>",
        ),
        element_type="float",
        values_per_line=8,
        license_header=None,
    )


if __name__ == "__main__":
    main()
