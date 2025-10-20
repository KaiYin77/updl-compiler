#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Quantization Analysis for UPDL Models

Analyzes models using representative datasets to determine optimal int16 quantization parameters.
Supports pluggable preprocessors for different model types (KWS, VAD, etc.).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .logger import log_info, log_debug, log_error
from .quantizer import calculate_udl_power_of_2_scale, calculate_params_symmetric


def calculate_symmetric_quantization_params(min_val, max_val, udl_mode=True):
    """Calculate symmetric quantization parameters for int16 range

    Args:
        min_val: Minimum expected float value
        max_val: Maximum expected float value
        udl_mode: If True, force scales to be power-of-2 for UDL hardware compatibility

    Returns:
        tuple: (scale, zero_point) for symmetric quantization
        Note: zero_point is always 0 for symmetric quantization
    """
    # Use the existing function from quantizer.py
    scale, zero_point = calculate_params_symmetric(min_val, max_val)

    # If UDL mode is enabled, convert to power-of-2 scale
    if udl_mode:
        max_abs = max(abs(min_val), abs(max_val))
        power_of_2_scale, shift, absolute_relative_error = calculate_udl_power_of_2_scale(scale, max_abs_value=max_abs)

        # Warn if approximation error is significant (>10% error)
        if absolute_relative_error > 0.1:
            log_debug(f"UDL Power-of-2 conversion: {scale:.8f} -> {power_of_2_scale:.8f} "
                     f"(shift={shift}, abs_rel_error={absolute_relative_error:.6f})")

        return power_of_2_scale, zero_point

    return scale, zero_point


def create_model_layer_output(model):
    """Create a model that outputs all intermediate layer activations"""
    # Get all layers that produce meaningful outputs
    layer_outputs = []
    layer_names = []
    layer_types = []

    for layer in model.layers:
        # Skip input layer and layers without meaningful outputs
        if hasattr(layer, "output") and layer.name != "input_1":
            try:
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
                layer_types.append(layer.__class__.__name__)
            except:
                continue

    # Create model that outputs all intermediate activations
    multi_output_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    return multi_output_model, layer_names, layer_types


def extract_activation_function(layer):
    """Extract activation function from layer config"""
    try:
        if hasattr(layer, "get_config"):
            config = layer.get_config()
            return config.get("activation", "linear")
    except:
        pass
    return "linear"


class QuantizationAnalyzer:
    """Generic quantization analyzer with pluggable preprocessors"""

    def __init__(self, preprocessor=None, batch_size=10):
        """
        Args:
            preprocessor: Data preprocessor implementing preprocess_sample() and load_calibration_data()
            batch_size: Number of samples to process at once
        """
        self.preprocessor = preprocessor
        self.batch_size = batch_size

    def analyze_model_quantization(self, model, calibration_data, output_path, udl_mode=True):
        """
        Analyze model using calibration data to determine optimal quantization parameters

        Args:
            model: TensorFlow/Keras model
            calibration_data: List of calibration samples
            output_path: Path to save quantization parameters JSON
            udl_mode: Enable UDL power-of-2 quantization

        Returns:
            dict: Quantization parameters
        """
        log_info(f"Starting quantization analysis with {len(calibration_data)} calibration samples")

        # Create multi-output model for layer logging
        multi_output_model, layer_names, layer_types = create_model_layer_output(model)

        log_info(f"Analyzing {len(layer_names)} layers for quantization parameters...")

        # Initialize accumulator for layer statistics
        layer_stats = {}
        input_mins, input_maxs = [], []

        # Process samples in batches to manage memory
        num_batches = (len(calibration_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing calibration batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(calibration_data))
            batch_samples = calibration_data[start_idx:end_idx]

            # Process batch samples
            batch_features = []
            for sample in batch_samples:
                try:
                    if self.preprocessor:
                        features = self.preprocessor.preprocess_sample(sample)
                    else:
                        # Assume sample is already preprocessed
                        features = sample

                    batch_features.append(features)

                    # Collect input statistics
                    input_flat = features.numpy().flatten()
                    input_mins.append(np.min(input_flat))
                    input_maxs.append(np.max(input_flat))

                except Exception as e:
                    log_error(f"Error processing sample: {e}")
                    continue

            if not batch_features:
                continue

            # Stack batch features and run inference
            try:
                batch_tensor = tf.concat(batch_features, axis=0)
                batch_layer_outputs = multi_output_model.predict(batch_tensor, verbose=0)

                # Accumulate layer statistics for each sample in the batch
                for sample_idx in range(batch_tensor.shape[0]):  # For each sample in batch
                    for layer_idx, (layer_name, layer_output) in enumerate(zip(layer_names, batch_layer_outputs)):
                        if layer_name not in layer_stats:
                            layer_stats[layer_name] = {
                                'mins': [],
                                'maxs': [],
                                'layer_idx': layer_idx,
                                'layer_type': layer_types[layer_idx]
                            }

                        # Collect min/max for this individual sample
                        sample_output = layer_output[sample_idx]  # Extract single sample output
                        output_flat = sample_output.flatten()
                        layer_stats[layer_name]['mins'].append(np.min(output_flat))
                        layer_stats[layer_name]['maxs'].append(np.max(output_flat))

            except Exception as e:
                log_error(f"Error processing batch: {e}")
                continue

        # Calculate global input quantization parameters
        global_input_min = np.min(input_mins)
        global_input_max = np.max(input_maxs)
        input_scale, input_zp = calculate_symmetric_quantization_params(global_input_min, global_input_max, udl_mode)

        log_info(f"Global input range: [{global_input_min:.6f}, {global_input_max:.6f}]")
        log_info(f"Input scale: {input_scale:.8f}, zero_point: {input_zp}")

        # Initialize quantization parameters structure
        quantization_params = {
            "metadata": {
                "method": "representative_dataset",
                "calibration_samples": len(input_mins),
                "udl_mode": udl_mode
            },
            "input": {
                "scale": float(input_scale),
                "zero_point": int(input_zp),
                "min_val": float(global_input_min),
                "max_val": float(global_input_max)
            },
            "layers": {}
        }

        log_info("Calculating layer-by-layer quantization parameters...")

        # Calculate global layer quantization parameters
        for layer_name, stats in layer_stats.items():
            if not stats['mins'] or not stats['maxs']:
                continue

            # Get layer object to extract activation function
            layer = model.get_layer(layer_name)
            activation = extract_activation_function(layer)

            # Calculate global min/max across all samples
            global_min = np.min(stats['mins'])
            global_max = np.max(stats['maxs'])

            # Calculate symmetric quantization parameters
            scale, zero_point = calculate_symmetric_quantization_params(global_min, global_max, udl_mode)

            # Store parameters
            quantization_params["layers"][layer_name] = {
                "layer_index": stats['layer_idx'],
                "layer_type": stats['layer_type'],
                "activation": activation,
                "scale": float(scale),
                "zero_point": int(zero_point),
                "min_val": float(global_min),
                "max_val": float(global_max),
                "range": float(global_max - global_min),
            }

            log_debug(f"Layer {stats['layer_idx']}: {layer_name} ({stats['layer_type']}, {activation})")
            log_debug(f"  Global range: [{global_min:.6f}, {global_max:.6f}] (from {len(stats['mins'])} samples)")
            log_debug(f"  Scale: {scale:.8f}, Zero-point: {zero_point}")

        # Save to JSON file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(quantization_params, f, indent=2)

        log_info(f"Quantization parameters saved to: {output_path}")
        log_info(f"Total layers analyzed: {len(quantization_params['layers'])}")
        log_info(f"Calibration samples processed: {len(input_mins)}")

        return quantization_params