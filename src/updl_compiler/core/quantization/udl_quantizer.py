#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UDL-specific quantization logic.
"""

from .config import QuantizationConfig
from .parameter_calculator import calculate_symmetric_quantization_params
from ..logger import log_debug, log_info


class UDLQuantizer:
    """UDL hardware-specific quantization operations."""

    def __init__(self, config: QuantizationConfig):
        """
        Initialize UDL quantizer with configuration.

        Args:
            config: QuantizationConfig instance
        """
        self.config = config

    def initialize_params(self, json_file, udl_mode=True):
        """
        Initialize quantization parameters from JSON file.

        Args:
            json_file: Path to quantization parameters JSON file
            udl_mode: Enable UDL shift-only mode

        Returns:
            bool: True if initialization successful
        """
        self.config.udl_shift_only_mode = udl_mode
        success = self.config.load_params_from_json(json_file)

        if success:
            log_info(f"UDL shift-only mode: {'ENABLED' if udl_mode else 'DISABLED'}")

        return success

    def get_layer_params(self, layer_name, layer_type=None, layer_idx=None):
        """
        Get quantization parameters for a specific layer.

        Args:
            layer_name: Name of the layer
            layer_type: Type of the layer (fallback lookup)
            layer_idx: Index of the layer (fallback lookup)

        Returns:
            tuple: (scale, zero_point) or None if not found
        """
        if not self.config.params:
            return None

        layers = self.config.params.get("layers", {})

        # Try to find layer by name first
        if layer_name in layers:
            layer_data = layers[layer_name]
            scale = layer_data.get("scale")
            zero_point = layer_data.get("zero_point", 0)
            log_debug(
                f"Found JSON params for {layer_name}: scale={scale:.8f}, zp={zero_point}"
            )
            return scale, zero_point

        # Try to find by layer type and index if name not found
        if layer_type and layer_idx is not None:
            for name, layer_data in layers.items():
                if (
                    layer_data.get("layer_type") == layer_type
                    and layer_data.get("layer_index") == layer_idx
                ):
                    scale = layer_data.get("scale")
                    zero_point = layer_data.get("zero_point", 0)
                    log_debug(
                        f"Found JSON params for {layer_type}[{layer_idx}]: scale={scale:.8f}, zp={zero_point}"
                    )
                    return scale, zero_point

        return None

    def get_input_params(self):
        """
        Get input quantization parameters.

        Returns:
            tuple: (scale, zero_point) or None if not found
        """
        if not self.config.params:
            return None

        input_data = self.config.params.get("input", {})
        if input_data:
            scale = input_data.get("scale")
            zero_point = input_data.get("zero_point", 0)
            log_debug(f"Found JSON input params: scale={scale:.8f}, zp={zero_point}")
            return scale, zero_point

        return None

    def calculate_weight_params(self, weights, layer_type, layer_idx=None, layer_name=None):
        """Calculate quantization parameters for layer weights."""
        import numpy as np
        from .parameter_calculator import calculate_symmetric_quantization_params

        if weights is not None:
            # Calculate optimal symmetric quantization based on actual weight distribution
            min_val = np.min(weights)
            max_val = np.max(weights)

            # Ensure minimum range to avoid division by zero
            if max_val - min_val < 0.01:
                center = (max_val + min_val) / 2
                min_val = center - 0.005
                max_val = center + 0.005

            # Calculate symmetric quantization parameters (UDL mode handled in calculate_symmetric_quantization_params)
            weight_scale, weight_zp = calculate_symmetric_quantization_params(
                min_val, max_val, udl_mode=self.config.udl_shift_only_mode
            )

            if self.config.udl_shift_only_mode:
                log_debug(f"Weight quantization (UDL mode): layer {layer_idx} ({layer_type}), "
                         f"range=[{min_val:.6f}, {max_val:.6f}], scale={weight_scale:.8f}")

            return weight_scale, weight_zp
        else:
            # Default for layers without weights (pooling, flatten, etc.)
            # Conservative symmetric range that accommodates most weight distributions
            weight_scale, weight_zp = calculate_symmetric_quantization_params(-4.0, 4.0, udl_mode=self.config.udl_shift_only_mode)
            return weight_scale, weight_zp

    def calculate_bias_params(self, bias, layer_name=None, layer_type=None, layer_idx=None):
        """Calculate quantization parameters for layer bias."""
        import numpy as np
        from .parameter_calculator import calculate_symmetric_quantization_params

        if bias is not None:
            # Calculate optimal symmetric quantization based on actual bias distribution
            min_val = np.min(bias)
            max_val = np.max(bias)

            # Ensure minimum range to avoid division by zero
            if max_val - min_val < 1e-8:
                center = (max_val + min_val) / 2
                min_val = center - 5e-9
                max_val = center + 5e-9

            # Calculate symmetric quantization parameters for bias (UDL mode handled in calculate_symmetric_quantization_params)
            bias_scale, bias_zp = calculate_symmetric_quantization_params(
                min_val, max_val, udl_mode=self.config.udl_shift_only_mode
            )

            if self.config.udl_shift_only_mode:
                log_debug(f"Bias quantization (UDL mode): layer {layer_idx} ({layer_type}), "
                         f"range=[{min_val:.6f}, {max_val:.6f}], scale={bias_scale:.8f}")

            return bias_scale, bias_zp

        else:
            # Default for layers without bias
            # Conservative range for bias values
            bias_scale, bias_zp = calculate_symmetric_quantization_params(-0.5, 0.5, udl_mode=self.config.udl_shift_only_mode)
            return bias_scale, bias_zp