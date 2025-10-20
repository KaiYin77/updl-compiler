#!/usr/bin/env python
"""
Model utilities for KWS models

Contains functions for model configuration, layer processing, and model modifications.
"""

import tensorflow as tf


def prepare_model_settings(label_count, args):
    """Calculates common settings needed for all models.
    Args:
      label_count: How many classes are to be recognized.
      args: Arguments containing sample_rate, clip_duration_ms, window_size_ms,
            window_stride_ms, dct_coefficient_count.
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)

    # For MFCC features (only feature type supported)
    dct_coefficient_count = args.dct_coefficient_count
    window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
    window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples

    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = args.dct_coefficient_count * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": args.sample_rate,
        "background_frequency": 0.8,
        "background_volume_range_": 0.1,
    }


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


def split_dense_softmax_layers(model):
    """
    Split final Dense layer with softmax activation into separate Dense (linear) + Softmax layers

    Args:
        model: Original Keras model

    Returns:
        Modified model with separated Dense and Softmax layers
    """
    print("\n=== Splitting Dense+Softmax into separate layers ===")

    # Find the last Dense layer with softmax activation
    last_dense_idx = None
    last_dense_layer = None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            activation = extract_activation_function(layer)
            if activation == "softmax":
                last_dense_idx = i
                last_dense_layer = layer
                print(f"Found Dense+Softmax layer at index {i}: {layer.name}")
                break

    if last_dense_idx is None:
        print("No Dense layer with softmax activation found - returning original model")
        return model

    # Count existing Dense and Softmax layers to determine proper naming
    dense_count = 0
    softmax_count = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_count += 1
        elif isinstance(layer, tf.keras.layers.Softmax) or (
            hasattr(layer, "name") and "softmax" in layer.name.lower()
        ):
            softmax_count += 1

    # Get the input to the Dense layer (output from previous layer)
    if last_dense_idx > 0:
        dense_input = model.layers[last_dense_idx - 1].output
    else:
        dense_input = model.input

    # Create new Dense layer with linear activation (same weights and bias)
    dense_config = last_dense_layer.get_config()
    dense_config["activation"] = "linear"  # Change activation to linear

    # Generate proper Dense layer name following Keras convention
    if dense_count == 1:
        new_dense_name = "dense"
    else:
        new_dense_name = f"dense_{dense_count - 1}"

    dense_config["name"] = new_dense_name

    # Create new Dense layer
    new_dense = tf.keras.layers.Dense.from_config(dense_config)
    dense_output = new_dense(dense_input)

    # Copy weights from original layer
    new_dense.set_weights(last_dense_layer.get_weights())

    # Generate proper Softmax layer name following Keras convention
    if softmax_count == 0:
        softmax_name = "softmax"
    else:
        softmax_name = f"softmax_{softmax_count}"

    # Add Softmax activation as separate layer
    softmax_layer = tf.keras.layers.Softmax(name=softmax_name)
    final_output = softmax_layer(dense_output)

    # Create new model
    new_model = tf.keras.Model(inputs=model.input, outputs=final_output)

    print(f"Created new Dense layer: {new_dense.name} (activation=linear)")
    print(f"Created new Softmax layer: {softmax_layer.name}")
    print(f"Original model layers: {len(model.layers)}")
    print(f"New model layers: {len(new_model.layers)}")

    return new_model
