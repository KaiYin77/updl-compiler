#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UPH5 (Upbeat Portable HDF5) Format Specification

This module centralizes all UPH5 binary format constants and data structures
to ensure consistency across serialization and deserialization modules.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import numpy as np


# === UPH5 Binary Format Constants ===
DESCRIPTION_LENGTH = 32
TAG_LENGTH = 16
STRING_LENGTH = 16
ALIGNMENT_4_BYTE = 4

# === Data Type Specifications ===
# CRITICAL: These lists MUST match the C enum definitions in updl_interpreter.h exactly
DTYPE_LIST = [
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "bool",
    "char",
    "dtype_t",
    "ltype_t",
    "ptype_t",
    "atype_t",
]

# CRITICAL: This order MUST match the C enum ltype_t in updl_interpreter.h exactly
LTYPE_LIST = [
    "Conv1D",  # 0: Ltype_conv_1d
    "Conv2D",  # 1: Ltype_conv_2d
    "DepthwiseConv2D",  # 2: Ltype_depthwise_conv_2d
    "MaxPooling2D",  # 3: Ltype_max_pooling_2d
    "AveragePooling2D",  # 4: Ltype_average_pooling_2d
    "Dense",  # 5: Ltype_dense
    "Flatten",  # 6: Ltype_flatten
    "Lambda",  # 7: Ltype_lambda
    "Add",  # 8: Ltype_add
    "Softmax",  # 9: Ltype_softmax
]

# CRITICAL: This order MUST match the C enum ptype_t in updl_interpreter.h exactly
# Ptype_valid=0, Ptype_same=1
PTYPE_LIST = ["valid", "same"]  # 0: Ptype_valid  # 1: Ptype_same

# CRITICAL: This order MUST match the C enum atype_t in updl_interpreter.h exactly
# Atype_none=0, Atype_linear=1, ..., Atype_tanh=6
ATYPE_LIST = [
    "none",  # 0: Atype_none
    "linear",  # 1: Atype_linear
    "relu",  # 2: Atype_relu
    "leakyrelu",  # 3: Atype_leakyrelu
    "softmax",  # 4: Atype_softmax
    "sigmoid",  # 5: Atype_sigmoid
    "tanh",  # 6: Atype_tanh
]


@dataclass
class UPH5QuantizationParams:
    """Quantization parameters for UPH5 format"""
    act_scale: float
    act_zp: int
    weight_scale: Optional[float] = None
    weight_zp: Optional[int] = None
    bias_scale: Optional[float] = None
    bias_zp: Optional[int] = None


@dataclass
class UPH5LayerMetadata:
    """Metadata for a single layer in UPH5 format"""
    name: str
    layer_type: str
    activation: str
    input_shape: List[int]
    output_shape: List[int]
    quantization: UPH5QuantizationParams

    # Layer-specific parameters
    filters: Optional[int] = None
    kernel_size: Optional[List[int]] = None
    strides: Optional[List[int]] = None
    padding: Optional[str] = None
    depth_multiplier: Optional[int] = None
    units: Optional[int] = None
    pool_size: Optional[List[int]] = None

    # Weight information
    weight_shape: Optional[List[int]] = None
    has_bias: bool = False


@dataclass
class UPH5FormatSpec:
    """Complete UPH5 format specification"""
    description: str
    model_name: str
    dtype: str = "int16_t"
    input_scale: float = 1.0
    input_shape: List[int] = None
    layers: List[UPH5LayerMetadata] = None

    def __post_init__(self):
        if self.input_shape is None:
            self.input_shape = [1, 6, 1, 1]
        if self.layers is None:
            self.layers = []

    @property
    def num_layers(self) -> int:
        """Number of layers in the model"""
        return len(self.layers)

    def validate_format(self) -> bool:
        """Validate UPH5 format constraints"""
        # Validate description length
        if len(self.description) > DESCRIPTION_LENGTH:
            return False

        # Validate model name length
        if len(self.model_name) > STRING_LENGTH:
            return False

        # Validate dtype
        if self.dtype not in DTYPE_LIST:
            return False

        # Validate input shape (should be 4D for compatibility)
        if len(self.input_shape) != 4:
            return False

        # Validate each layer
        for layer in self.layers:
            if not self._validate_layer(layer):
                return False

        return True

    def _validate_layer(self, layer: UPH5LayerMetadata) -> bool:
        """Validate individual layer metadata"""
        # Check layer type
        layer_type = layer.layer_type
        if layer_type == "BatchNormalization":
            layer_type = "BatchNorm"  # Map for storage

        if layer_type not in LTYPE_LIST:
            return False

        # Check activation
        if layer.activation not in ATYPE_LIST:
            return False

        # Check padding (if applicable)
        if layer.padding and layer.padding not in PTYPE_LIST:
            return False

        return True

    def get_c_array_filename(self) -> str:
        """Generate C array filename for this model"""
        return f"{self.model_name}_int16"

    def get_header_guard(self) -> str:
        """Generate header guard for C header file"""
        filename = self.get_c_array_filename()
        return f"{filename.upper()}_H"


# === Layout Optimization Specifications ===

class WeightLayoutSpec:
    """Specifications for weight layout optimizations"""

    # Convolutional layer layouts
    CONV2D_TF_FORMAT = "HWIO"  # TensorFlow: [H, W, I, O]
    CONV2D_OPTIMAL_FORMAT = "OIHW"  # C-optimal: [O, I, H, W]

    DEPTHWISE_TF_FORMAT = "HWID"  # TensorFlow: [H, W, I, D]
    DEPTHWISE_OPTIMAL_FORMAT = "I1HW"  # C-optimal: [I, 1, H, W]

    # Dense layer layouts
    DENSE_TF_FORMAT = "IF"  # TensorFlow: [input_features, output_features]
    DENSE_OPTIMAL_FORMAT = "FI"  # C-optimal: [output_features, input_features]

    @staticmethod
    def get_conv2d_transpose_axes():
        """Get transpose axes for Conv2D: HWIO -> OIHW"""
        return (3, 2, 0, 1)

    @staticmethod
    def get_depthwise_transpose_axes():
        """Get transpose axes for DepthwiseConv2D: HWID -> I1HW"""
        return (2, 3, 0, 1)

    @staticmethod
    def requires_transpose(layer_type: str, weight_name: str) -> bool:
        """Check if weight layout optimization is needed"""
        if weight_name == "bias":
            return False

        return layer_type in ["Conv2D", "DepthwiseConv2D", "Dense"]


# === UPH5 Compatibility Constants ===

class UPH5Compatibility:
    """UPH5 format compatibility specifications"""

    UDL_VERSION = "udl1.1"
    MEMORY_ALIGNMENT = 4  # 4-byte alignment for hardware compatibility
    DEFAULT_DTYPE = "int16_t"

    # File extensions
    BINARY_EXTENSION = ".uph5"
    HEADER_EXTENSION = ".h"
    SOURCE_EXTENSION = ".c"
    METADATA_EXTENSION = "_metadata.json"
    WEIGHTS_DEBUG_EXTENSION = "_weights.json"

    @staticmethod
    def get_alignment_padding(current_pos: int, alignment: int = MEMORY_ALIGNMENT) -> int:
        """Calculate padding bytes needed for alignment"""
        return (alignment - (current_pos % alignment)) % alignment

    @staticmethod
    def validate_alignment(file_pos: int, alignment: int = MEMORY_ALIGNMENT) -> bool:
        """Check if file position is properly aligned"""
        return (file_pos % alignment) == 0