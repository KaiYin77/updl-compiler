"""
KWS Common Module - Shared utilities for KWS quantization and inference scripts
This module contains common functions used across multiple KWS scripts to avoid duplication.
"""

from .keras_model import get_model
from .preprocessor import convert_wav_to_tensor, convert_wav_to_mfcc_features_tensor
from .quantizer import calculate_symmetric_quantization_params
from .parser import parse_command
from .model_utils import (
    prepare_model_settings,
    create_model_layer_output,
    extract_activation_function,
    split_dense_softmax_layers,
)

__all__ = [
    "convert_wav_to_tensor",
    "convert_wav_to_mfcc_features_tensor",
    "prepare_model_settings",
    "create_model_layer_output",
    "extract_activation_function",
    "split_dense_softmax_layers",
    "calculate_symmetric_quantization_params",
    "parse_command",
    "get_model",
]
