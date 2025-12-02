#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""
Reusable helpers for generating quantized int16 input fixtures.

The functions in this module are intentionally lightweight so that example
scripts (e.g. keyword spotting, sound detection) can share the same workflow:

1. Load input quantization parameters from the compiler cache.
2. Sample a dataset via TensorFlow Datasets.
3. Run the project-specific preprocessor to obtain float32 feature tensors.
4. Quantize features to int16 using the shared quantizer.
5. Emit a C array via the compiler serializer helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from rich.console import Console

from ..core.quantization import QuantizationConfig
from ..core.quantizer import quantize_input_data_fp32_to_int16
from ..core.serializer import serialize_input_feature_to_c_array
from ..core.license import MLPERF_APACHE_LICENSE_HEADER


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for dataset-driven feature generation."""

    dataset_dir: Path
    quant_params_path: Path
    output_c_path: Path
    dataset_name: str = "speech_commands"
    sample_count: int = 20
    random_seed: int = 1234
    array_name: str = "g_model_inputs_int16"
    outer_dim_token: str | None = None
    inner_dim_token: str | None = None
    include_directive: str | None = None
    output_header_path: Path | None = None
    header_guard: str | None = None
    element_type: str = "int16_t"
    header_includes: Tuple[str, ...] = ("#include <stdint.h>",)
    values_per_line: int = 16


# Model-Agnostic Test Generation Framework

@dataclass
class ModelConfig:
    """Model-specific configuration for test generation."""
    name: str                           # "ic", "kws", "vww"
    display_name: str                   # "Image Classification"
    preprocessor_class: type            # ICPreprocessor, KWSPreprocessor
    label_extractor: Callable[[dict], str]  # ic_label_extractor, kws_label_extractor
    dataset_name: str                   # "cifar10", "speech_commands"
    dataset_dir: Path                   # Model-specific dataset path
    sample_count: int = 5
    random_seed: int = 1234
    array_name_template: str = "g_{}_test_inputs_fp32"  # {} filled with name
    outer_dim_token_template: str = "kNum{}TestInputs"  # {} filled with capitalized name
    inner_dim_token_template: str = "k{}InputSize"      # {} filled with capitalized name
    element_type: str = "float"
    header_includes: Tuple[str, ...] = ("#include <stddef.h>",)
    values_per_line: int = 8


def create_model_config(model_type: str, **overrides) -> ModelConfig:
    """Factory function for creating model configurations."""
    base_configs = {
        "ic": {
            "name": "ic",
            "display_name": "Image Classification",
            "dataset_name": "cifar10",
            "sample_count": 5,
        },
        "kws": {
            "name": "kws",
            "display_name": "Keyword Spotting",
            "dataset_name": "speech_commands",
            "sample_count": 10,
        },
        "vww": {
            "name": "vww",
            "display_name": "Visual Wake Words",
            "dataset_name": "visual_wake_words",
            "sample_count": 5,
        }
    }

    if model_type not in base_configs:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(base_configs.keys())}")

    base = base_configs[model_type]

    # Apply overrides
    for key, value in overrides.items():
        base[key] = value

    # Create ModelConfig - required fields must be provided via overrides
    return ModelConfig(**base)


class TestInputGenerator:
    """Model-agnostic test input generator."""

    def __init__(self, model_config: ModelConfig, example_dir: Path):
        self.config = model_config
        self.example_dir = example_dir
        self.console = Console()
        self.generation_config = self._create_generation_config()

    def generate_test_inputs(self) -> Tuple[Path, Path]:
        """Main entry point - handles entire test input generation flow."""
        self._show_header()
        layout = self._prompt_layout_selection()
        output_dir = self._get_output_directory(layout)

        features, labels = self._load_features()

        # Apply layout transformation if needed
        if layout == "updl":
            features = self._apply_layout_transformation(features)

        if self._should_apply_quantization():
            features = self._apply_quantization_cycle(features)
            feature_type = "quantized"
        else:
            feature_type = "original fp32"

        return self._save_features(features, labels, feature_type, output_dir, layout)

    def _create_generation_config(self) -> GenerationConfig:
        """Create GenerationConfig from ModelConfig."""
        name_cap = self.config.name.capitalize()  # "ic" -> "Ic"

        return GenerationConfig(
            dataset_dir=self.config.dataset_dir,
            quant_params_path=self.example_dir / ".updlc_cache" / "unused_fp32_params.json",
            output_c_path=self.example_dir / "uph5" / f"{self.config.name}_test_inputs_fp32.c",
            dataset_name=self.config.dataset_name,
            sample_count=self.config.sample_count,
            random_seed=self.config.random_seed,
            array_name=self.config.array_name_template.format(self.config.name),
            outer_dim_token=self.config.outer_dim_token_template.format(name_cap),
            inner_dim_token=self.config.inner_dim_token_template.format(name_cap),
            include_directive=f'#include "{self.config.name}_test_inputs_fp32.h"',
            output_header_path=self.example_dir / "uph5" / f"{self.config.name}_test_inputs_fp32.h",
            header_guard=f"{self.config.name.upper()}_TEST_INPUTS_FP32_H",
            element_type=self.config.element_type,
            header_includes=self.config.header_includes,
            values_per_line=self.config.values_per_line,
        )

    def _show_header(self) -> None:
        """Display model-specific header."""
        title = f"{self.config.display_name} Test Input Generator (fp32 with Quantization)"
        self.console.rule(f"[bold green]{title}")

    def _prompt_layout_selection(self) -> str:
        """Prompt user for layout selection."""
        layout = prompt_layout_selection(self.console)
        self.console.print(
            f"\nUsing [bold]{layout.upper()}[/] layout format",
            style="cyan",
        )
        return layout

    def _get_output_directory(self, layout: str) -> Path:
        """Get output directory based on layout choice."""
        if layout == "updl":
            output_dir = self.example_dir / "uph5"
            self.console.print("Saving to [bold]uph5/[/] directory for UPDL layout", style="cyan")
        else:
            output_dir = self.example_dir / "litert"
            self.console.print("Saving to [bold]litert/[/] directory for TensorFlow layout", style="cyan")
        return output_dir

    def _apply_layout_transformation(self, features: np.ndarray) -> np.ndarray:
        """Transform input features from TensorFlow layout (NHWC) to UPDL layout (NCHW)."""
        if len(features.shape) == 4:  # 4D tensor: [batch, height, width, channels]
            # Transform from NHWC to NCHW: [N, H, W, C] -> [N, C, H, W]
            transformed_features = np.transpose(features, (0, 3, 1, 2))
            self.console.print(
                f"Transformed input layout: {features.shape} → {transformed_features.shape} (NHWC→NCHW)",
                style="green",
            )
            return transformed_features
        elif len(features.shape) == 3:  # 3D tensor: [batch, length, channels]
            # Transform from NLC to NCL: [N, L, C] -> [N, C, L]
            transformed_features = np.transpose(features, (0, 2, 1))
            self.console.print(
                f"Transformed input layout: {features.shape} → {transformed_features.shape} (NLC→NCL)",
                style="green",
            )
            return transformed_features
        else:
            # For 2D tensors (like Dense inputs) or other dimensions, no transformation needed
            self.console.print(
                f"Input shape {features.shape}: no layout transformation needed",
                style="dim",
            )
            return features

    def _load_features(self) -> Tuple[np.ndarray, List[str]]:
        """Load and preprocess features using model-specific preprocessor."""
        preprocessor = self.config.preprocessor_class()
        features, labels = collect_features_and_labels(
            self.generation_config, preprocessor, self.config.label_extractor
        )
        self.console.print(
            f"Loaded [bold]{features.shape[0]}[/] samples with feature shape [bold]{features.shape[1:]}[/]",
            style="green",
        )
        return features, labels

    def _should_apply_quantization(self) -> bool:
        """Prompt user for quantization choice."""
        return prompt_quantization_cycle_choice(self.console)

    def _apply_quantization_cycle(self, features: np.ndarray) -> np.ndarray:
        """Apply quantization cycle to features."""
        try:
            quant_params = prompt_quantization_params(self.console, self.example_dir)
            self.console.print(
                f"\nApplying quantization cycle: scale={quant_params['scale']}, zero_point={quant_params['zero_point']}",
                style="yellow",
            )

            self.console.print("Processing input features through quantization cycle...", style="cyan")
            quantized_features = apply_quantization_cycle(features, quant_params['scale'], quant_params['zero_point'], self.console)
            self.console.print("✓ Input quantization cycle completed", style="green")

            return quantized_features

        except (ValueError, KeyboardInterrupt):
            self.console.print("[red]Error:[/] Invalid quantization parameters. Exiting.", style="red")
            raise

    def _save_features(self, features: np.ndarray, labels: List[str], feature_type: str, output_dir: Path, layout: str) -> Tuple[Path, Path]:
        """Save processed features to C arrays."""
        if feature_type == "original fp32":
            self.console.print("[cyan]Skipping quantization cycle - using original fp32 features[/]", style="cyan")

        # Prepare flat samples from features
        flat_samples = [sample.flatten() for sample in features]
        input_size = flat_samples[0].size

        # Create layout-specific configuration
        layout_config = self._create_layout_specific_config(output_dir, layout)

        output_path = write_c_array(
            layout_config,
            flat_samples,
            labels,
            input_size,
            license_header=None,
        )

        header_path = write_header_file(
            layout_config,
            len(flat_samples),
            input_size,
            license_header=None,
        )

        layout_info = f"({layout.upper()} layout)" if layout == "updl" else "(TensorFlow layout)"
        self.console.print(f"Wrote {layout_config.sample_count} {feature_type} samples to {output_path} {layout_info}", style="green")
        self.console.print(f"Dataset directory: {layout_config.dataset_dir}", style="cyan")
        self.console.print(f"Header written to {header_path}", style="green")

        return output_path, header_path

    def _create_layout_specific_config(self, output_dir: Path, layout: str) -> GenerationConfig:
        """Create GenerationConfig with layout-specific paths."""
        name_cap = self.config.name.capitalize()

        return GenerationConfig(
            dataset_dir=self.config.dataset_dir,
            quant_params_path=self.example_dir / ".updlc_cache" / "unused_fp32_params.json",
            output_c_path=output_dir / f"{self.config.name}_test_inputs_fp32.c",
            dataset_name=self.config.dataset_name,
            sample_count=self.config.sample_count,
            random_seed=self.config.random_seed,
            array_name=self.config.array_name_template.format(self.config.name),
            outer_dim_token=self.config.outer_dim_token_template.format(name_cap),
            inner_dim_token=self.config.inner_dim_token_template.format(name_cap),
            include_directive=f'#include "{self.config.name}_test_inputs_fp32.h"',
            output_header_path=output_dir / f"{self.config.name}_test_inputs_fp32.h",
            header_guard=f"{self.config.name.upper()}_TEST_INPUTS_FP32_H",
            element_type=self.config.element_type,
            header_includes=self.config.header_includes,
            values_per_line=self.config.values_per_line,
        )


class LayerOutputGenerator:
    """Model-agnostic layer output generator."""

    def __init__(self, model_config: ModelConfig, example_dir: Path, model_path: Path):
        self.config = model_config
        self.example_dir = example_dir
        self.model_path = model_path
        self.console = Console()
        self.generation_config = self._create_layer_config()

    def generate_layer_outputs(self) -> None:
        """Main entry point - handles entire layer output generation flow."""
        self._show_header()
        layout = self._prompt_layout_selection()
        output_dir = self._get_output_directory(layout)

        # Load model and features
        model = tf.keras.models.load_model(self.model_path)
        layers = list_capture_layers(model)
        layer_names = [layer.name for layer in layers]

        features, labels = self._load_features()

        # Ask about quantization
        if self._should_apply_quantization():
            features = self._apply_quantization_cycle(features)

        # Layer selection and processing
        selected_indices = self._prompt_layer_selection(layer_names)
        if not selected_indices:
            self.console.print("No layers selected; exiting.", style="yellow")
            return

        chosen_layers = [layers[idx] for idx in selected_indices]
        chosen_names = [layer.name for layer in chosen_layers]

        self.console.print(
            "\nGenerating outputs for: [bold]" + ", ".join(chosen_names) + "[/]",
            style="cyan",
        )

        activations = capture_layer_outputs(model, features, chosen_layers)

        # Handle layout transformations if needed
        if layout == "updl":
            activations = self._handle_updl_layout(model, selected_indices, chosen_layers, activations)

        # Save results
        save_results(
            chosen_names,
            activations,
            labels,
            base_config=self.generation_config,
            output_dir=output_dir,
            array_prefix=f"{self.config.name}_test_layers_fp32",
            outer_dim_token=f"kNum{self.config.name.capitalize()}TestInputs",
            console=self.console,
            array_name_prefix=f"g_{self.config.name}",
            array_name_suffix="_fp32",
            inner_dim_prefix=f"k{self.config.name.capitalize()}",
            header_includes=("#include <stddef.h>",),
            element_type="float",
            values_per_line=8,
            license_header=None,
        )

    def _create_layer_config(self) -> GenerationConfig:
        """Create GenerationConfig for layer output."""
        return GenerationConfig(
            dataset_dir=self.config.dataset_dir,
            quant_params_path=self.example_dir / ".updlc_cache" / "unused_fp32_params.json",
            output_c_path=self.example_dir / "unused.c",
            dataset_name=self.config.dataset_name,
            sample_count=self.config.sample_count,
            random_seed=self.config.random_seed,
        )

    def _show_header(self) -> None:
        """Display model-specific header."""
        title = f"{self.config.display_name} Layer Activation Exporter (fp32)"
        self.console.rule(f"[bold green]{title}")

    def _prompt_layout_selection(self) -> str:
        """Prompt user for layout selection."""
        layout = prompt_layout_selection(self.console)
        self.console.print(
            f"\nUsing [bold]{layout.upper()}[/] layout format",
            style="cyan",
        )
        return layout

    def _get_output_directory(self, layout: str) -> Path:
        """Get output directory based on layout choice."""
        if layout == "updl":
            output_dir = self.example_dir / "uph5"
            self.console.print("Saving to [bold]uph5/[/] directory for UPDL layout", style="cyan")
        else:
            output_dir = self.example_dir / "litert"
            self.console.print("Saving to [bold]litert/[/] directory for TensorFlow layout", style="cyan")
        return output_dir

    def _load_features(self) -> Tuple[np.ndarray, List[str]]:
        """Load and preprocess features."""
        preprocessor = self.config.preprocessor_class()
        features, labels = collect_features_and_labels(
            self.generation_config, preprocessor, self.config.label_extractor
        )
        self.console.print(
            f"Loaded [bold]{features.shape[0]}[/] samples with feature shape [bold]{features.shape[1:]}[/]",
            style="green",
        )
        return features, labels

    def _should_apply_quantization(self) -> bool:
        """Prompt user for quantization choice."""
        return prompt_quantization_cycle_choice(self.console)

    def _apply_quantization_cycle(self, features: np.ndarray) -> np.ndarray:
        """Apply quantization cycle to features."""
        try:
            quant_params = prompt_quantization_params(self.console, self.example_dir)
            self.console.print(
                f"\nApplying quantization cycle to input data: scale={quant_params['scale']}, zero_point={quant_params['zero_point']}",
                style="yellow",
            )

            self.console.print("Processing input features through quantization cycle...", style="cyan")
            quantized_features = apply_quantization_cycle(features, quant_params['scale'], quant_params['zero_point'], self.console)
            self.console.print("✓ Input quantization cycle completed", style="green")

            return quantized_features

        except (ValueError, KeyboardInterrupt):
            self.console.print("[red]Error:[/] Invalid quantization parameters. Exiting.", style="red")
            raise

    def _prompt_layer_selection(self, layer_names: List[str]) -> List[int]:
        """Prompt user to select layers."""
        from rich.table import Table

        table = Table(title="Available Layers", show_lines=False)
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Layer Name", style="magenta")
        for idx, name in enumerate(layer_names):
            table.add_row(f"{idx:02d}", name)
        self.console.print(table)

        try:
            selected_indices = prompt_layer_indices(
                layer_names,
                input_fn=self.console.input,
                print_fn=self.console.print,
                show_layer_list=False,
                prompt_text="[bold]Selection[/]: ",
            )
            return selected_indices
        except ValueError as exc:
            self.console.print(f"[red]Error:[/] {exc}")
            return []

    def _handle_updl_layout(self, model, selected_indices: List[int], chosen_layers: List, activations: List[np.ndarray]) -> List[np.ndarray]:
        """Handle UPDL layout transformations."""
        layout_info = get_layer_layout_info(model, selected_indices, "updl")

        self.console.print("Analyzing UPDL layout requirements...", style="yellow")
        for idx in selected_indices:
            info = layout_info[idx]
            layer_name = info["layer_name"]
            layer_type = info["layer_type"]
            major_layer_type = info["major_layer_type"]
            requires_transformation = info["requires_transformation"]

            if layer_type != major_layer_type:
                if requires_transformation:
                    self.console.print(
                        f"  • {layer_name} ({layer_type}): Inherits {major_layer_type} UPDL layout",
                        style="cyan",
                    )
                else:
                    self.console.print(
                        f"  • {layer_name} ({layer_type}): Follows {major_layer_type} (no transformation)",
                        style="dim",
                    )
            else:
                if requires_transformation:
                    self.console.print(
                        f"  • {layer_name} ({layer_type}): UPDL layout transformation required",
                        style="green",
                    )
                else:
                    self.console.print(
                        f"  • {layer_name} ({layer_type}): No layout transformation needed",
                        style="dim",
                    )

        # Transform activation layouts from TF (NHWC) to UPDL (NCHW)
        self.console.print("Transforming activation layouts from NHWC to NCHW...", style="yellow")
        transformed_activations = []

        for i, (layer, activation) in enumerate(zip(chosen_layers, activations)):
            major_layer_type = find_major_computational_layer(model, selected_indices[i])
            needs_transform = requires_activation_layout_transform(major_layer_type)

            if needs_transform:
                original_shape = activation.shape
                transformed_activation = transform_activation_layout(activation, major_layer_type, "updl")
                self.console.print(
                    f"  • {layer.name}: {original_shape} → {transformed_activation.shape} (NHWC→NCHW)",
                    style="green",
                )
                transformed_activations.append(transformed_activation)
            else:
                self.console.print(
                    f"  • {layer.name}: {activation.shape} (no transformation needed)",
                    style="dim",
                )
                transformed_activations.append(activation)

        return transformed_activations


def load_layer_quant_params(
    path: Path,
) -> List[Tuple[str, List[str], Dict[str, float]]]:
    """
    Load per-layer quantization parameters from a compiler cache JSON file.

    Returns:
        A list of tuples `(fusable_layer_name, op_names, params_dict)` in the
        order they appear in the JSON. `op_names` preserves the metadata order,
        and callers can choose which Keras layer to target. The params dict
        includes `scale`, `zero_point`, `min_val`, and `max_val`.
    """

    data = json.loads(path.read_text())
    layers: List[Tuple[str, List[str], Dict[str, float]]] = []

    for fusable_layer, ops in data["layers"].items():
        op_order = list(ops.keys())
        terminal_op = op_order[-1]
        cfg = ops[terminal_op]
        layers.append(
            (
                fusable_layer,
                op_order,
                {
                    "scale": float(cfg["scale"]),
                    "zero_point": int(cfg.get("zero_point", 0)),
                    "min_val": float(cfg.get("min_val", 0.0)),
                    "max_val": float(cfg.get("max_val", 0.0)),
                },
            )
        )

    return layers


def load_input_scale_zero_point(config: GenerationConfig) -> Tuple[float, int]:
    """Return (scale, zero_point) from the compiler-generated JSON cache."""
    quant_config = QuantizationConfig()
    if not quant_config.load_params_from_json(str(config.quant_params_path)):
        raise RuntimeError(
            f"Cannot load quantization params from {config.quant_params_path}"
        )

    params = quant_config.get_input_params()
    if not params:
        raise RuntimeError(
            f"No input quantization parameters found in {config.quant_params_path}"
        )

    try:
        scale = float(params["scale"])
        zero_point = int(params.get("zero_point", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed input quantization parameters in {config.quant_params_path}: {exc}"
        ) from exc

    return scale, zero_point


def sample_tfds_dataset(config: GenerationConfig) -> List[dict]:
    """Sample `sample_count` elements from the configured TFDS dataset."""
    if not config.dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {config.dataset_dir}")

    dataset = tfds.load(
        config.dataset_name,
        split="train",
        data_dir=str(config.dataset_dir),
    )
    shuffled = dataset.shuffle(
        buffer_size=10_000,
        seed=config.random_seed,
        reshuffle_each_iteration=False,
    )
    samples = list(shuffled.take(config.sample_count))
    if len(samples) < config.sample_count:
        print(
            f"[WARN] Requested {config.sample_count} samples but dataset only yielded {len(samples)}; "
            "continuing with available samples."
        )
    return samples


LabelExtractor = Callable[[dict], str]


def extract_labeled_features(
    preprocessor: Callable[[dict], tf.Tensor],
    samples: Iterable[dict],
    label_extractor: LabelExtractor,
) -> List[Tuple[str, np.ndarray]]:
    """Run the model preprocessor over dataset samples and attach labels."""
    labeled_features: List[Tuple[str, np.ndarray]] = []
    for sample in samples:
        label = label_extractor(sample)
        features = preprocessor(sample).numpy().astype(np.float32)
        labeled_features.append((label, features))
    return labeled_features


def quantize_features(
    labeled_features: Sequence[Tuple[str, np.ndarray]],
    scale: float,
    zero_point: int,
) -> Tuple[List[np.ndarray], List[str]]:
    """Quantize float features to int16 and return flattened vectors plus labels."""
    quantized_vectors: List[np.ndarray] = []
    labels: List[str] = []

    for idx, (label, feature_np) in enumerate(labeled_features):
        quantized, overflow_count = quantize_input_data_fp32_to_int16(
            feature_np,
            scale,
            zero_point,
            return_overflow_count=True,
        )
        if overflow_count:
            print(
                f"[WARN] Clipped {overflow_count} values outside int16 range for sample {idx} ({label})."
            )
        quantized_vectors.append(quantized.flatten())
        labels.append(label)

    return quantized_vectors, labels


def list_capture_layers(
    model: tf.keras.Model,
    *,
    skip_input_layers: bool = True,
) -> List[tf.keras.layers.Layer]:
    """Return an ordered list of layers suitable for activation capture."""

    layers: List[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if skip_input_layers and isinstance(layer, tf.keras.layers.InputLayer):
            continue
        layers.append(layer)
    return layers


def capture_layer_outputs(
    model: tf.keras.Model,
    features: np.ndarray,
    target_layers: Sequence[str | tf.keras.layers.Layer],
    *,
    batch_size: int | None = None,
) -> List[np.ndarray]:
    """Run inference and return activations for the specified layers."""

    resolved_layers: List[tf.keras.layers.Layer] = []
    for layer_ref in target_layers:
        if isinstance(layer_ref, tf.keras.layers.Layer):
            resolved_layers.append(layer_ref)
        else:
            resolved_layers.append(model.get_layer(layer_ref))

    capture_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in resolved_layers],
        name="updl_activation_capture",
    )
    activations = capture_model.predict(
        features,
        batch_size=batch_size or features.shape[0],
        verbose=0,
    )
    if not isinstance(activations, list):
        activations = [activations]
    return activations


def format_float_array_to_c(
    samples_2d: np.ndarray,
    labels: Sequence[str],
    array_name: str,
    *,
    values_per_line: int = 8,
) -> str:
    """Serialize flattened float32 activations into a C array string."""

    sample_count, flat_size = samples_2d.shape
    lines: List[str] = []
    lines.append("// Generated activation data; values are float32.")
    lines.append(f"const float {array_name}[{sample_count}][{flat_size}] = {{")

    for idx, row in enumerate(samples_2d):
        label = labels[idx] if idx < len(labels) else f"sample_{idx}"
        lines.append(f"    {{ // {label}")
        row_values = [f"{value:.8f}f" for value in row]
        for offset in range(0, flat_size, values_per_line):
            chunk = row_values[offset : offset + values_per_line]
            trailing_comma = "," if offset + values_per_line < flat_size else ""
            lines.append("        " + ", ".join(chunk) + trailing_comma)
        comma = "," if idx + 1 < sample_count else ""
        lines.append(f"    }}{comma}")

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def prompt_layer_indices(
    layer_names: Sequence[str],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
    show_layer_list: bool = True,
    prompt_text: str = "Selection: ",
) -> List[int]:
    """Prompt the user to choose layer indices."""

    input_fn = input_fn or input
    print_fn = print_fn or print

    if show_layer_list:
        print_fn("Available layers:")
        for idx, name in enumerate(layer_names):
            print_fn(f"  [{idx:02d}] {name}")

    raw = input_fn(prompt_text).strip().lower()
    if not raw or raw == "all":
        return list(range(len(layer_names)))

    indices: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:  # pragma: no cover
                raise ValueError(f"Invalid range '{token}'") from exc
            if start > end:
                start, end = end, start
            for val in range(start, end + 1):
                indices.add(val)
        else:
            try:
                indices.add(int(token))
            except ValueError as exc:  # pragma: no cover
                raise ValueError(f"Invalid index '{token}'") from exc

    for idx in indices:
        if idx < 0 or idx >= len(layer_names):
            raise ValueError(
                f"Index {idx} is out of bounds (0-{len(layer_names) - 1})."
            )

    return sorted(indices)


def write_c_array(
    config: GenerationConfig,
    quantized_samples: Sequence[np.ndarray],
    labels: Sequence[str],
    input_size: int,
    *,
    license_header: str | None = None,
    element_type: str | None = None,
    value_formatter: Callable[[float], str] | None = None,
    values_per_line: int | None = None,
) -> Path:
    """Serialize quantized samples into a C source file."""
    resolved_element_type = element_type or getattr(config, "element_type", "int16_t")
    resolved_values_per_line = values_per_line or getattr(config, "values_per_line", 16)

    body = serialize_input_feature_to_c_array(
        quantized_samples,
        labels,
        input_size,
        array_name=config.array_name,
        outer_dim_token=config.outer_dim_token,
        inner_dim_token=config.inner_dim_token,
        include_directive=config.include_directive,
        license_header=license_header,
        element_type=resolved_element_type,
        value_formatter=value_formatter,
        values_per_line=resolved_values_per_line,
    )

    output_path = config.output_c_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def write_header_file(
    config: GenerationConfig,
    sample_count: int,
    input_size: int,
    *,
    license_header: str | None = MLPERF_APACHE_LICENSE_HEADER,
    element_type: str | None = None,
    header_includes: Sequence[str] | None = None,
) -> Path:
    """Create a header declaring the generated input array."""
    if config.output_header_path is None:
        raise ValueError("output_header_path is not configured.")

    header_path = config.output_header_path
    if not header_path.is_absolute():
        header_path = Path.cwd() / header_path
    header_path.parent.mkdir(parents=True, exist_ok=True)

    guard = config.header_guard
    if not guard:
        guard = header_path.name.replace(".", "_").upper()

    resolved_element_type = element_type or getattr(config, "element_type", "int16_t")
    resolved_includes = header_includes or getattr(
        config, "header_includes", ("#include <stdint.h>",)
    )

    lines: list[str] = []
    if license_header:
        lines.append(license_header.strip("\n"))
        lines.append("")

    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")

    includes = tuple(resolved_includes)
    if includes:
        lines.extend(includes)
        lines.append("")

    if config.outer_dim_token:
        lines.append(f"#define {config.outer_dim_token} {sample_count}")
    if config.inner_dim_token:
        lines.append(f"#define {config.inner_dim_token} {input_size}")

    outer = config.outer_dim_token or str(sample_count)
    inner = config.inner_dim_token or str(input_size)

    lines.append("")
    lines.append(
        f"extern const {resolved_element_type} {config.array_name}[{outer}][{inner}];"
    )
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")

    header_path.write_text("\n".join(lines), encoding="utf-8")
    return header_path


def _sanitize_token(value: str) -> str:
    """Return a C-friendly identifier token derived from the provided string."""
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", value)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "layer"


def _camelize_token(value: str) -> str:
    """Convert a sanitized identifier into CamelCase for macro generation."""
    value = value.lstrip("_")
    parts = [part for part in value.split("_") if part]
    if not parts:
        return "Layer"
    return "".join(part.capitalize() for part in parts)


def save_results(
    layer_names: Sequence[str],
    activations: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    base_config: GenerationConfig,
    output_dir: Path,
    array_prefix: str,
    outer_dim_token: str,
    console: Any | None = None,
    array_name_prefix: str | None = None,
    array_name_suffix: str = "",
    inner_dim_prefix: str = "k",
    inner_dim_suffix: str = "FeatureSize",
    header_includes: Sequence[str] | None = None,
    element_type: str | None = None,
    values_per_line: int | None = None,
    include_header_template: str = "{base_name}.h",
    header_guard_suffix: str = "_H",
    license_header: str | None = None,
) -> None:
    """
    Save layer activations to C source/header files using the provided config template.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_element_type = element_type or getattr(
        base_config, "element_type", "int16_t"
    )
    resolved_values_per_line = values_per_line or getattr(
        base_config, "values_per_line", 16
    )
    resolved_header_includes = (
        tuple(header_includes)
        if header_includes is not None
        else getattr(base_config, "header_includes", ())
    )

    for layer_name, act in zip(layer_names, activations):
        flattened = act.reshape(act.shape[0], -1).astype(np.float32)
        safe_token = _sanitize_token(layer_name)
        base_name = f"{array_prefix}_{safe_token}"

        array_name_base = array_name_prefix or array_prefix
        array_name = f"{array_name_base}_{safe_token}" if array_name_base else base_name
        if array_name_suffix:
            array_name = f"{array_name}{array_name_suffix}"

        inner_dim_token = None
        if inner_dim_prefix or inner_dim_suffix:
            inner_dim_token = (
                f"{inner_dim_prefix}{_camelize_token(safe_token)}{inner_dim_suffix}"
            )

        include_header_name = include_header_template.format(
            base_name=base_name,
            safe_token=safe_token,
            layer_name=layer_name,
        )
        include_directive = f'#include "{include_header_name}"'

        layer_config = replace(
            base_config,
            array_name=array_name,
            output_c_path=output_dir / f"{base_name}.c",
            output_header_path=output_dir / include_header_name,
            include_directive=include_directive,
            element_type=resolved_element_type,
            header_includes=resolved_header_includes,
            values_per_line=resolved_values_per_line,
            outer_dim_token=outer_dim_token,
            inner_dim_token=inner_dim_token,
            header_guard=f"{_sanitize_token(base_name).upper()}{header_guard_suffix}",
        )

        flat_rows = [row for row in flattened]
        input_size = flattened.shape[1]

        output_path = write_c_array(
            layer_config,
            flat_rows,
            labels,
            input_size,
            license_header=license_header,
        )
        header_path = write_header_file(
            layer_config,
            flattened.shape[0],
            input_size,
            license_header=license_header,
        )

        message = (
            "  • Saved "
            f"{output_path} and {header_path} "
            f"(samples={flattened.shape[0]}, flat_size={flattened.shape[1]})"
        )
        if console is not None and hasattr(console, "print"):
            console.print(message, style="green")
        else:
            print(message)


def transform_activation_layout(
    activation: np.ndarray, layer_type: str, layout: str
) -> np.ndarray:
    """
    Transform activation tensors between TensorFlow layout (NHWC) and UPDL layout (NCHW).

    Args:
        activation: Activation tensor from TensorFlow inference
        layer_type: Type of the layer that produced this activation
        layout: Either 'tf' for TensorFlow layout or 'updl' for UPDL layout

    Returns:
        Transformed activation tensor
    """
    if layout == "tf":
        # Return activation as-is for TensorFlow layout
        return activation

    # For UPDL layout, convert from NHWC (TensorFlow) to NCHW (C-optimized)
    if len(activation.shape) == 4:  # 4D tensor: [batch, height, width, channels]
        # Transform from NHWC to NCHW: [N, H, W, C] -> [N, C, H, W]
        return np.transpose(activation, (0, 3, 1, 2))
    elif len(activation.shape) == 3:  # 3D tensor: [batch, length, channels]
        # Transform from NLC to NCL: [N, L, C] -> [N, C, L]
        return np.transpose(activation, (0, 2, 1))
    elif len(activation.shape) == 2:  # 2D tensor: [batch, features] - no change needed
        return activation
    else:
        # For other dimensions, return as-is
        return activation


def requires_activation_layout_transform(layer_type: str) -> bool:
    """
    Check if a layer type produces activations that need layout transformation.

    Args:
        layer_type: The type of layer

    Returns:
        True if the layer produces spatial/channel activations that need NHWC->NCHW conversion
    """
    # Layers that produce spatial activations (need NHWC->NCHW conversion)
    spatial_layers = [
        "Conv1D",
        "Conv2D",
        "DepthwiseConv2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "BatchNormalization",
        "Activation",
        "Dropout",  # These inherit from previous spatial layers
    ]
    return layer_type in spatial_layers


def find_major_computational_layer(
    model: tf.keras.Model, target_layer_index: int
) -> str:
    """
    Find the major computational layer that influences the layout for a given layer.

    Args:
        model: The Keras model
        target_layer_index: Index of the target layer

    Returns:
        The type of the major computational layer that should determine the layout
    """
    from ..core.schema.uph5 import LTYPE_LIST

    layers = list_capture_layers(model)

    # Major computational layer types are defined in UPH5 LTYPE_LIST
    major_layer_types = LTYPE_LIST

    # Dependent layer types that inherit layout from major layers
    # These are layers not in LTYPE_LIST but commonly used in models
    dependent_layer_types = ["BatchNormalization", "Activation", "Dropout"]

    target_layer = layers[target_layer_index]
    target_layer_type = target_layer.__class__.__name__

    # If the target layer is already a major computational layer, return its type
    if target_layer_type in major_layer_types:
        return target_layer_type

    # If it's a dependent layer, look backwards to find the most recent major layer
    if target_layer_type in dependent_layer_types:
        for i in range(target_layer_index - 1, -1, -1):
            prev_layer = layers[i]
            prev_layer_type = prev_layer.__class__.__name__
            if prev_layer_type in major_layer_types:
                return prev_layer_type

    # Default to the layer's own type if no major layer found
    return target_layer_type


def get_layer_layout_info(
    model: tf.keras.Model, layer_indices: List[int], layout: str
) -> Dict[int, Dict[str, str]]:
    """
    Get layout information for selected layers, considering major computational layer dependencies.

    Args:
        model: The Keras model
        layer_indices: List of selected layer indices
        layout: Either 'tf' or 'updl'

    Returns:
        Dictionary mapping layer index to layout information
    """
    from ..core.schema.uph5 import WeightLayoutSpec

    layers = list_capture_layers(model)
    layout_info = {}

    for idx in layer_indices:
        layer = layers[idx]
        layer_type = layer.__class__.__name__
        major_layer_type = find_major_computational_layer(model, idx)

        layout_info[idx] = {
            "layer_name": layer.name,
            "layer_type": layer_type,
            "major_layer_type": major_layer_type,
            "requires_transformation": WeightLayoutSpec.requires_transpose(
                major_layer_type, "kernel"
            ),
            "layout_format": layout,
        }

    return layout_info


def prompt_layout_selection(console=None) -> str:
    """Prompt user to choose between TensorFlow and UPDL layout formats."""
    layout_options = {
        "tf": "TensorFlow native layout (HWIO weights, NHWC activations)",
        "updl": "UPDL C-optimized layout (OIHW weights, NCHW activations)",
    }

    if console and hasattr(console, "print"):
        console.print("\n[bold cyan]Available Layout Formats:[/]")
        for key, description in layout_options.items():
            console.print(f"  [bold]{key.upper()}[/]: {description}")
    else:
        print("\nAvailable Layout Formats:")
        for key, description in layout_options.items():
            print(f"  {key.upper()}: {description}")

    while True:
        if console and hasattr(console, "input"):
            choice = (
                console.input("\n[bold]Select layout format (tf/updl)[/]: ")
                .strip()
                .lower()
            )
        else:
            choice = input("\nSelect layout format (tf/updl): ").strip().lower()

        if choice in layout_options:
            return choice
        elif choice == "":
            if console and hasattr(console, "print"):
                console.print("[yellow]Defaulting to TensorFlow layout[/]")
            else:
                print("Defaulting to TensorFlow layout")
            return "tf"
        else:
            error_msg = f"Invalid choice '{choice}'. Please enter 'tf' or 'updl'"
            if console and hasattr(console, "print"):
                console.print(f"[red]{error_msg}[/]")
            else:
                print(error_msg)


def collect_features_and_labels(
    layer_config: GenerationConfig, preprocessor: Any, label_extractor: LabelExtractor
) -> Tuple[np.ndarray, List[str]]:
    """
    Sample TFDS data using the provided config and return float features + labels.

    Args:
        layer_config: Generation configuration
        preprocessor: Domain-specific preprocessor instance
        label_extractor: Function to extract label from sample dict

    Returns:
        Tuple of (features array, labels list)
    """
    raw_samples: List[dict] = sample_tfds_dataset(layer_config)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        label_extractor,
    )

    features = np.concatenate([feat for _, feat in labeled_features], axis=0)
    labels = [label for label, _ in labeled_features]
    return features, labels


def generate_test_inputs_fp32(
    config: GenerationConfig,
    preprocessor: Any,
    label_extractor: LabelExtractor,
    license_header: str | None = None,
) -> Tuple[Path, Path]:
    """
    Generate float32 test inputs following the standard workflow.

    Args:
        config: Generation configuration with paths and parameters
        preprocessor: Domain-specific preprocessor instance
        label_extractor: Function to extract labels from samples
        license_header: Optional license header for generated files

    Returns:
        Tuple of (output_path, header_path)
    """
    raw_samples: List[dict] = sample_tfds_dataset(config)
    labeled_features = extract_labeled_features(
        preprocessor.preprocess_sample,
        raw_samples,
        label_extractor,
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
        config,
        flat_samples,
        labels,
        input_size,
        license_header=license_header,
    )
    header_path = write_header_file(
        config,
        len(flat_samples),
        input_size,
        license_header=license_header,
    )

    return output_path, header_path


def load_quantization_params_from_uph5_metadata(metadata_file: Path) -> Dict[str, float] | None:
    """Load input quantization parameters from UPH5 metadata JSON file."""
    try:
        if metadata_file.exists():
            data = json.loads(metadata_file.read_text())
            input_quant = data.get("model_info", {}).get("input_quantization", {})
            if "scale" in input_quant and "zero_point" in input_quant:
                return {
                    "scale": float(input_quant["scale"]),
                    "zero_point": int(input_quant["zero_point"])
                }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def prompt_quantization_cycle_choice(console=None) -> bool:
    """Prompt user to choose whether to apply quantization cycle."""
    if console and hasattr(console, "print"):
        console.print("\n[bold cyan]Quantization Cycle Option[/]")
        console.print("Do you want to apply fp32->int16->fp32 quantization cycle to input features?")
        console.print("[dim]This simulates the quantization effects that would occur during inference.[/]")
    else:
        print("\nQuantization Cycle Option")
        print("Do you want to apply fp32->int16->fp32 quantization cycle to input features?")
        print("This simulates the quantization effects that would occur during inference.")

    while True:
        if console and hasattr(console, "input"):
            choice = console.input("Apply quantization cycle? ([bold]y[/]/n): ").strip().lower()
        else:
            choice = input("Apply quantization cycle? (y/n): ").strip().lower()

        if choice in ['y', 'yes', '']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            error_msg = "Please enter 'y' for yes or 'n' for no"
            if console and hasattr(console, "print"):
                console.print(f"[yellow]{error_msg}[/]")
            else:
                print(error_msg)


def prompt_quantization_params(console=None, cache_dir: Path | str | None = None) -> Dict[str, float]:
    """Prompt user for input quantization parameters, loading from UPH5 metadata if available."""

    # Set up UPH5 metadata file path - try to find any UPH5 metadata file
    if cache_dir:
        cache_dir_path = Path(cache_dir) / ".updlc_cache"
    else:
        cache_dir_path = Path(".updlc_cache")

    # Look for any UPH5 metadata file in the cache directory
    uph5_metadata_path = None
    if cache_dir_path.exists():
        for file_path in cache_dir_path.glob("uph5_metadata_*_quantize_params.json"):
            uph5_metadata_path = file_path
            break

    # Try to load parameters from UPH5 metadata
    uph5_params = load_quantization_params_from_uph5_metadata(uph5_metadata_path) if uph5_metadata_path else None

    if console and hasattr(console, "print"):
        console.print("\n[bold yellow]Input Quantization Parameters[/]")
        console.print("Enter quantization parameters for fp32->int16->fp32 conversion:")
        if uph5_params:
            console.print(f"[dim]Found in UPH5 metadata: scale={uph5_params['scale']}, zero_point={uph5_params['zero_point']}[/]")
            console.print("[dim]Press Enter to use UPH5 values, or type new values:[/]")
    else:
        print("\nInput Quantization Parameters")
        print("Enter quantization parameters for fp32->int16->fp32 conversion:")
        if uph5_params:
            print(f"Found in UPH5 metadata: scale={uph5_params['scale']}, zero_point={uph5_params['zero_point']}")
            print("Press Enter to use UPH5 values, or type new values:")

    try:
        if console and hasattr(console, "input"):
            scale_input = console.input(f"Scale: ").strip()
            if not scale_input and uph5_params:
                scale = uph5_params['scale']
                zero_point = uph5_params['zero_point']
                if console and hasattr(console, "print"):
                    console.print(f"[green]Using UPH5 metadata values: scale={scale}, zero_point={zero_point}[/]")
                else:
                    print(f"Using UPH5 metadata values: scale={scale}, zero_point={zero_point}")
            else:
                zero_point_input = console.input("Zero point (e.g., 0): ").strip()
                scale = float(scale_input)
                zero_point = int(zero_point_input) if zero_point_input else 0
        else:
            scale_input = input(f"Scale: ").strip()
            if not scale_input and uph5_params:
                scale = uph5_params['scale']
                zero_point = uph5_params['zero_point']
                print(f"Using UPH5 metadata values: scale={scale}, zero_point={zero_point}")
            else:
                zero_point_input = input("Zero point (e.g., 0): ").strip()
                scale = float(scale_input)
                zero_point = int(zero_point_input) if zero_point_input else 0

        return {"scale": scale, "zero_point": zero_point}

    except (ValueError, KeyboardInterrupt) as e:
        error_msg = f"Invalid input - {e}"
        if console and hasattr(console, "print"):
            console.print(f"[red]Error:[/] {error_msg}")
        else:
            print(f"Error: {error_msg}")
        raise


def quantize_fp32_to_int16(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize fp32 data to int16."""
    # Quantize: q = round(x / scale) + zero_point
    quantized = np.round(data / scale) + zero_point
    # Clamp to int16 range
    quantized = np.clip(quantized, -32768, 32767)
    return quantized.astype(np.int16)


def dequantize_int16_to_fp32(quantized_data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize int16 data back to fp32."""
    # Dequantize: x = (q - zero_point) * scale
    dequantized = (quantized_data.astype(np.float32) - zero_point) * scale
    return dequantized


def apply_quantization_cycle(data: np.ndarray, scale: float, zero_point: int, console=None) -> np.ndarray:
    """Apply complete quantization cycle: fp32 -> int16 -> fp32."""
    if console and hasattr(console, "print"):
        console.print(f"  Original range: [{data.min():.6f}, {data.max():.6f}]", style="dim")
    else:
        print(f"  Original range: [{data.min():.6f}, {data.max():.6f}]")

    # Step 1: fp32 -> int16
    quantized = quantize_fp32_to_int16(data, scale, zero_point)
    if console and hasattr(console, "print"):
        console.print(f"  Quantized range: [{quantized.min()}, {quantized.max()}]", style="dim")
    else:
        print(f"  Quantized range: [{quantized.min()}, {quantized.max()}]")

    # Step 2: int16 -> fp32
    dequantized = dequantize_int16_to_fp32(quantized, scale, zero_point)
    if console and hasattr(console, "print"):
        console.print(f"  Dequantized range: [{dequantized.min():.6f}, {dequantized.max():.6f}]", style="dim")
    else:
        print(f"  Dequantized range: [{dequantized.min():.6f}, {dequantized.max():.6f}]")

    return dequantized
