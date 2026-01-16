#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Shared interactive prompts for test generation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .quantization import load_quantization_params_from_uph5_metadata


def prompt_layout_selection(console=None) -> str:
    """Prompt user to choose between TensorFlow and UPDL layout formats."""
    layout_options = {
        "tf": "TensorFlow native layout (HWIO weights, NHWC activations)",
        "updl": "UPDL C-optimized layout (OIHW weights, NCHW activations)",
    }

    printer = getattr(console, "print", print)
    printer("\nAvailable Layout Formats:")
    for key, description in layout_options.items():
        printer(f"  {key.upper()}: {description}")

    while True:
        reader = getattr(console, "input", input)
        choice = reader("\nSelect layout format (tf/updl): ").strip().lower()

        if choice in layout_options:
            return choice
        if choice == "":
            printer("Defaulting to TensorFlow layout")
            return "tf"
        printer(f"Invalid choice '{choice}'. Please enter 'tf' or 'updl'")


def prompt_layer_indices(
    layer_names: Sequence[str],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
    show_layer_list: bool = True,
    prompt_text: str = "Selection: ",
) -> list[int]:
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
            start = int(start_str)
            end = int(end_str)
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))

    resolved = sorted(idx for idx in indices if 0 <= idx < len(layer_names))
    if not resolved:
        raise ValueError("No valid layer indices selected.")
    return resolved


def prompt_fakequant_choice(console=None) -> bool:
    """Prompt user to choose whether to apply FakeQuant (fp32→int16→fp32)."""
    printer = getattr(console, "print", print)
    printer("\nApply FakeQuant Node")
    printer("Do you want to simulate TensorFlow Lite FakeQuant (fp32→int16→fp32)?")
    printer("[dim]This mimics the fixed-point effects that occur during inference.[/]")

    reader = getattr(console, "input", input)
    while True:
        choice = reader("Apply FakeQuant node? (y/n): ").strip().lower()
        if choice in {"y", "yes", ""}:
            return True
        if choice in {"n", "no"}:
            return False
        printer("Please enter 'y' for yes or 'n' for no")


def prompt_quantization_params(
    console=None, cache_dir: Path | str | None = None
) -> dict[str, float]:
    """Prompt user for input quantization parameters, prefilling from metadata."""

    cache_dir_path = Path(cache_dir) if cache_dir else Path(".updlc_cache")
    uph5_metadata_path = next(
        cache_dir_path.glob("uph5_metadata_*_quantize_params.json"), None
    )
    uph5_params = (
        load_quantization_params_from_uph5_metadata(uph5_metadata_path)
        if uph5_metadata_path
        else None
    )

    printer = getattr(console, "print", print)
    printer("\nInput Quantization Parameters")
    if uph5_params:
        printer(
            f"Found in UPH5 metadata: scale={uph5_params['scale']}, "
            f"zero_point={uph5_params['zero_point']}"
        )
        printer("Press Enter to use UPH5 values, or type new values:")

    reader = getattr(console, "input", input)
    try:
        scale_input = reader("Scale: ").strip()
        if not scale_input and uph5_params:
            return dict(scale=uph5_params["scale"], zero_point=uph5_params["zero_point"])

        zero_point_input = reader("Zero point (e.g., 0): ").strip()
        scale = float(scale_input)
        zero_point = int(zero_point_input) if zero_point_input else 0
        return {"scale": scale, "zero_point": zero_point}
    except (ValueError, KeyboardInterrupt) as exc:
        printer(f"[red]Error:[/] Invalid input - {exc}")
        raise
