#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
UPDL Compiler Core - Model-agnostic compilation pipeline.
"""

from .config import (
    DESCRIPTION_LENGTH,
    TAG_LENGTH,
    STRING_LENGTH,
    DTYPE_LIST,
    LTYPE_LIST,
    PTYPE_LIST,
    ATYPE_LIST,
    LAYER_CONFIG,
    is_layer_supported,
    is_layer_skipped,
    is_layer_fuseable,
    is_activation_layer,
    validate_layer_configuration,
)

from .loader import load_model

from .quantizer import (
    initialize_params,
    set_udl_shift_only_mode,
    calculate_weight_params,
    calculate_bias_params,
)

from .fuser import (
    fuse_layers_from_json,
    fuse_to_uph5_layer,
    combine_fused_data_step5,
)

from .serializer import (
    serialize_uph5_metadata_to_json,
    serialize_uph5_weight_to_json,
    serialize_uph5_to_c_array,
)

from .quantization_analyzer import QuantizationAnalyzer

from .quantization import (
    QuantizationConfig,
    UDLQuantizer,
    calculate_udl_power_of_2_scale,
    calculate_symmetric_quantization_params,
)


from .preprocessors import (
    DataPreprocessor,
)

from .logger import (
    LOG_LEVEL_OFF,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_TRACE,
    set_log_level,
    log_info,
    log_debug,
    log_warn,
    log_error,
)

__all__ = [
    # Config
    "DESCRIPTION_LENGTH", "TAG_LENGTH", "STRING_LENGTH",
    "DTYPE_LIST", "LTYPE_LIST", "PTYPE_LIST", "ATYPE_LIST", "LAYER_CONFIG",
    "is_layer_supported", "is_layer_skipped", "is_layer_fuseable", "is_activation_layer",
    "validate_layer_configuration",

    # Loader
    "load_model",

    # Quantizer
    "initialize_params", "set_udl_shift_only_mode",
    "calculate_weight_params", "calculate_bias_params",
    "calculate_udl_power_of_2_scale",

    # Fuser
    "fuse_layers_from_json", "fuse_to_uph5_layer", "combine_fused_data_step5",

    # Serializer
    "serialize_uph5_metadata_to_json", "serialize_uph5_weight_to_json",
    "serialize_uph5_to_c_array",

    # Quantization
    "QuantizationAnalyzer", "QuantizationConfig", "UDLQuantizer",
    "calculate_symmetric_quantization_params", "calculate_udl_power_of_2_scale",

    # Preprocessors
    "DataPreprocessor",

    # Logger
    "LOG_LEVEL_OFF", "LOG_LEVEL_ERROR", "LOG_LEVEL_WARN", "LOG_LEVEL_INFO",
    "LOG_LEVEL_DEBUG", "LOG_LEVEL_TRACE", "set_log_level",
    "log_info", "log_debug", "log_warn", "log_error",
]