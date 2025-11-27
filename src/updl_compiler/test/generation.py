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
