#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""High-level workflows for generating test inputs and layer activations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

from .configs import GenerationConfig, ModelConfig, build_generation_config
from .data import collect_features_and_labels
from .io import save_results, write_c_array, write_header_file
from .layers import (
    capture_layer_outputs,
    get_layer_layout_info,
    list_capture_layers,
    requires_activation_layout_transform,
    transform_activation_layout,
)
from .prompts import (
    prompt_layout_selection,
    prompt_layer_indices,
    prompt_fakequant_choice,
    prompt_quantization_params,
)
from .quantization import apply_fakequant_cycle


class TestInputGenerator:
    """Model-agnostic test input generator."""

    def __init__(
        self,
        model_config: ModelConfig,
        example_dir: Path,
        *,
        console: Console | None = None,
    ):
        self.config = model_config
        self.example_dir = example_dir
        self.console = console or Console()

    def generate_test_inputs(self, layout_override: str | None = None) -> Tuple[Path, Path]:
        """Entry point that orchestrates preprocessing, quantization, and emission."""
        console = self.console
        console.rule(
            f"[bold green]{self.config.display_name} Test Input Generator (fp32 with Quantization)"
        )

        layout = layout_override or prompt_layout_selection(console)
        output_dir = self._get_output_directory(layout)
        console.print(
            f"\nUsing [bold]{layout.upper()}[/] layout format → storing artifacts in {output_dir}",
            style="cyan",
        )

        generation_cfg = build_generation_config(
            self.config,
            example_dir=self.example_dir,
            output_dir=output_dir,
        )

        preprocessor = self.config.preprocessor_class()
        features, labels = collect_features_and_labels(
            generation_cfg,
            preprocessor,
            self.config.label_extractor,
            self.config.data_loader,
        )
        console.print(
            f"Loaded [bold]{features.shape[0]}[/] samples with feature shape [bold]{features.shape[1:]}[/]",
            style="green",
        )

        features = self._apply_layout_transformation(features, layout)

        if prompt_fakequant_choice(console):
            quant_params = prompt_quantization_params(console, self.example_dir)
            console.print(
                f"\nApplying FakeQuant node: scale={quant_params['scale']}, zero_point={quant_params['zero_point']}",
                style="yellow",
            )
            features = apply_fakequant_cycle(
                features, quant_params["scale"], quant_params["zero_point"], console
            )
        else:
            console.print("[cyan]Skipping FakeQuant simulation - using original fp32 features[/]")

        flat_samples = [sample.flatten() for sample in features]
        input_size = flat_samples[0].size

        output_path = write_c_array(
            generation_cfg,
            flat_samples,
            labels,
            input_size,
            license_header=None,
        )
        header_path = write_header_file(
            generation_cfg,
            len(flat_samples),
            input_size,
            license_header=None,
        )

        console.print(f"Wrote samples to {output_path}", style="green")
        console.print(f"Header written to {header_path}", style="green")
        return output_path, header_path

    def _get_output_directory(self, layout: str) -> Path:
        if layout == "updl":
            output_dir = self.example_dir / "uph5"
            self.console.print("Saving to [bold]uph5/[/] directory for UPDL layout", style="cyan")
        else:
            output_dir = self.example_dir / "litert"
            self.console.print("Saving to [bold]litert/[/] directory for TensorFlow layout", style="cyan")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _apply_layout_transformation(self, features: np.ndarray, layout: str) -> np.ndarray:
        if layout != "updl":
            return features

        if len(features.shape) == 4:
            transformed = np.transpose(features, (0, 3, 1, 2))
            self.console.print(
                f"Transformed input layout: {features.shape} → {transformed.shape} (NHWC→NCHW)",
                style="green",
            )
            return transformed
        if len(features.shape) == 3:
            transformed = np.transpose(features, (0, 2, 1))
            self.console.print(
                f"Transformed input layout: {features.shape} → {transformed.shape} (NLC→NCL)",
                style="green",
            )
            return transformed

        self.console.print(
            f"Input shape {features.shape}: no layout transformation needed",
            style="dim",
        )
        return features


class LayerOutputGenerator:
    """Model-agnostic layer output generator."""

    def __init__(
        self,
        model_config: ModelConfig,
        example_dir: Path,
        model_path: Path,
        *,
        console: Console | None = None,
    ):
        self.config = model_config
        self.example_dir = example_dir
        self.model_path = model_path
        self.console = console or Console()

    def generate_layer_outputs(self, layout_override: str | None = None) -> None:
        console = self.console
        console.rule(f"[bold green]{self.config.display_name} Layer Activation Exporter (fp32)")

        layout = layout_override or prompt_layout_selection(console)
        output_dir = self._get_output_directory(layout)

        model = tf.keras.models.load_model(self.model_path)
        layers = list_capture_layers(model)
        layer_names = [layer.name for layer in layers]

        self._show_layer_table(layer_names)
        indices = prompt_layer_indices(
            layer_names,
            input_fn=console.input,
            print_fn=console.print,
            show_layer_list=False,
            prompt_text="[bold]Selection[/]: ",
        )
        if not indices:
            console.print("No layers selected; exiting.", style="yellow")
            return

        preprocessor = self.config.preprocessor_class()
        generation_cfg = build_generation_config(
            self.config,
            example_dir=self.example_dir,
            output_dir=output_dir,
        )
        features, labels = collect_features_and_labels(
            generation_cfg,
            preprocessor,
            self.config.label_extractor,
            self.config.data_loader,
        )

        chosen_layers = [layers[idx] for idx in indices]
        activations = capture_layer_outputs(model, features, chosen_layers)

        if layout == "updl":
            layout_info = get_layer_layout_info(model, indices, layout)
            console.print("Analyzing UPDL layout requirements...", style="yellow")
            for idx in indices:
                info = layout_info[idx]
                note = "UPDL layout transformation required" if info["requires_transformation"] else "No layout transformation needed"
                console.print(f"  • {info['layer_name']} ({info['layer_type']}): {note}", style="cyan")

            activations = [
                transform_activation_layout(act, layer.__class__.__name__, layout)
                if requires_activation_layout_transform(layer.__class__.__name__)
                else act
                for act, layer in zip(activations, chosen_layers)
            ]

        save_results(
            [layer.name for layer in chosen_layers],
            activations,
            labels,
            base_config=generation_cfg,
            output_dir=output_dir,
            array_prefix=f"{self.config.name}_test_layers_fp32",
            outer_dim_token=f"kNum{self.config.name.capitalize()}TestInputs",
            console=console,
            array_name_prefix=f"g_{self.config.name}",
            array_name_suffix="_fp32",
            inner_dim_prefix=f"k{self.config.name.capitalize()}",
            header_includes=("#include <stddef.h>",),
            element_type="float",
            values_per_line=8,
            license_header=None,
        )

    def _get_output_directory(self, layout: str) -> Path:
        if layout == "updl":
            output_dir = self.example_dir / "uph5"
        else:
            output_dir = self.example_dir / "litert"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _show_layer_table(self, layer_names: List[str]) -> None:
        table = Table(title="Available Layers", show_lines=False)
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Layer Name", style="magenta")
        for idx, name in enumerate(layer_names):
            table.add_row(f"{idx:02d}", name)
        self.console.print(table)
