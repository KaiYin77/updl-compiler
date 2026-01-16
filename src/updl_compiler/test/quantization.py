#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Quantization utilities shared across test workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def quantize_fp32_to_int16(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize fp32 data to int16."""
    quantized = np.round(data / scale) + zero_point
    quantized = np.clip(quantized, -32768, 32767)
    return quantized.astype(np.int16)


def dequantize_int16_to_fp32(quantized_data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize int16 data back to fp32."""
    return (quantized_data.astype(np.float32) - zero_point) * scale


def apply_fakequant_cycle(data: np.ndarray, scale: float, zero_point: int, console=None) -> np.ndarray:
    """Simulate a FakeQuant node: fp32 → int16 → fp32."""
    printer = getattr(console, "print", print)

    printer("\n[dim]FakeQuant: quantizing data to int16...[/]")
    quantized = quantize_fp32_to_int16(data, scale, zero_point)
    printer("[green]✓ Quantization complete[/]")

    printer("[dim]FakeQuant: dequantizing back to float32...[/]")
    dequantized = dequantize_int16_to_fp32(quantized, scale, zero_point)
    printer("[green]✓ Dequantization complete[/]")

    mse = np.mean((data - dequantized) ** 2)
    max_error = np.max(np.abs(data - dequantized))
    printer(f"[dim]Quantization cycle error metrics: MSE={mse:.6e}, Max Abs Error={max_error:.6e}[/]")

    return dequantized.astype(np.float32)


def load_quantization_params_from_uph5_metadata(metadata_file: Path | None) -> Dict[str, float] | None:
    """Load quantization parameters from UPH5 metadata JSON."""
    if not metadata_file or not metadata_file.exists():
        return None

    try:
        data = json.loads(metadata_file.read_text())
        input_quant = data.get("input")
        if input_quant and "scale" in input_quant and "zero_point" in input_quant:
            return {
                "scale": float(input_quant["scale"]),
                "zero_point": int(input_quant["zero_point"]),
            }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None
