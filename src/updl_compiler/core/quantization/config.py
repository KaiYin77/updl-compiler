#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Quantization configuration management.
"""

import os
import json
from ..logger import log_debug


class QuantizationConfig:
    """Manages quantization parameters and configuration state."""

    def __init__(self, udl_mode=True):
        """
        Initialize quantization configuration.

        Args:
            udl_mode: Enable UDL shift-only quantization mode
        """
        self.udl_shift_only_mode = udl_mode
        self.params = None

    def load_params_from_json(self, json_file=None):
        """
        Load quantization parameters from JSON file.

        Args:
            json_file: Path to JSON file with quantization parameters.
                      If None, searches for 'quantization_params_int16.json' in current directory.

        Returns:
            bool: True if parameters loaded successfully, False otherwise
        """
        if json_file is None:
            json_file = "quantization_params_int16.json"

        if not os.path.exists(json_file):
            log_debug(
                f"Quantization params file not found: {json_file}, using default parameters"
            )
            return False

        try:
            with open(json_file, "r") as f:
                self.params = json.load(f)
            log_debug(f"Loaded quantization parameters from {json_file}")
            return True
        except Exception as e:
            log_debug(f"Error loading quantization parameters: {e}")
            return False

    def get_layer_params(self, layer_name):
        """Get quantization parameters for a specific layer."""
        if not self.params:
            return None
        return self.params.get("layers", {}).get(layer_name, None)

    def get_input_params(self):
        """Get input quantization parameters."""
        if not self.params:
            return None
        return self.params.get("input", {})