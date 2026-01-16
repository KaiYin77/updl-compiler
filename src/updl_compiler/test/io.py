#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Helpers for emitting generated test data to C sources."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .configs import GenerationConfig
from ..core.codegen import serialize_input_feature_to_c_array
from ..core.license import MLPERF_APACHE_LICENSE_HEADER


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


def format_float_array_to_c(
    samples_2d: np.ndarray,
    labels: Sequence[str],
    array_name: str,
    *,
    values_per_line: int = 8,
) -> str:
    """Serialize flattened float32 activations into a C array string."""

    sample_count, flat_size = samples_2d.shape
    lines: list[str] = []
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
    """Persist per-layer activations to paired C/ header files."""
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
