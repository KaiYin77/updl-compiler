#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Streaming Wake Word model architecture

Simplified CNN model based on the benchmark streaming_wakeword implementation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2


def conv_block(inputs,
               repeat,
               kernel_size,
               filters,
               dilation,
               stride,
               filter_separable,
               residual=False,
               padding='valid',  # benchmark uses 'valid' by default
               dropout=0.0,
               activation='relu',
               l2_reg=1e-3,
               scale=True):
    """Convolutional block matching benchmark implementation"""

    regularizer = l2(l2_reg) if l2_reg > 0 else None

    net = inputs

    # Benchmark uses padding='valid' and manual padding for causal
    if kernel_size > 0:
        if padding == 'causal':
            net = tf.pad(net, [[0, 0], [kernel_size-1, 0], [0, 0], [0, 0]], 'constant')
            dw_pad = 'valid'
        elif padding == 'valid':
            dw_pad = 'valid'
        elif padding == 'same':
            dw_pad = 'same'
        else:
            dw_pad = padding

        # DepthwiseConv2D
        net = DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=(stride, stride),
            padding=dw_pad,
            dilation_rate=(dilation, 1),
            kernel_regularizer=regularizer,
            use_bias=False
        )(net)

    # Conv2D 1x1 - pointwise
    net = Conv2D(
        filters=filters,
        kernel_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        padding='valid'
    )(net)

    net = BatchNormalization(scale=scale)(net)

    if residual:
        # Residual connection (not used in benchmark but included for completeness)
        net_res = Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            padding='valid'
        )(inputs)
        net_res = BatchNormalization(scale=scale)(net_res)
        net = tf.keras.layers.Add()([net, net_res])

    net = Activation(activation)(net)
    net = Dropout(rate=dropout)(net)

    return net


def create_sww_model(input_shape, num_classes=3, l2_reg=0.001):
    """
    Create a streaming wake word detection model matching benchmark architecture

    Args:
        input_shape: Input shape [time_steps, 1, features]
        num_classes: Number of output classes (default: 3 for marvin/silence/unknown)
        l2_reg: L2 regularization coefficient

    Returns:
        Keras model (uncompiled)
    """

    # Benchmark architecture parameters
    ds_filters = [128, 128, 128, 32]
    ds_repeat = [1, 1, 1, 1]
    ds_residual = [0, 0, 0, 0]
    ds_kernel_size = [3, 5, 10, 15]
    ds_stride = [1, 1, 1, 1]
    ds_dilation = [1, 1, 1, 1]
    ds_padding = ['valid', 'valid', 'valid', 'valid']
    ds_filter_separable = [1, 1, 1, 1]
    ds_scale = 1
    dropout = 0.2
    activation = "relu"

    inputs = Input(shape=input_shape)
    net = inputs

    # Apply conv blocks exactly as in benchmark
    for count in range(len(ds_filters)):
        net = conv_block(
            net,
            repeat=ds_repeat[count],
            kernel_size=ds_kernel_size[count],
            filters=ds_filters[count],
            dilation=ds_dilation[count],
            stride=ds_stride[count],
            filter_separable=ds_filter_separable[count],
            residual=ds_residual[count],
            padding=ds_padding[count],
            scale=ds_scale,
            dropout=dropout,
            l2_reg=l2_reg,
            activation=activation
        )

    # Flatten instead of global average pooling (benchmark uses flatten)
    net = Flatten()(net)

    # Final classification layer with softmax
    outputs = Dense(num_classes, activation='softmax')(net)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_simple_sww_model(input_shape, num_classes=3):
    """
    Create a simpler streaming wake word detection model for quick testing

    Args:
        input_shape: Input shape [time_steps, 1, features]
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """

    inputs = Input(shape=input_shape)

    # Simple CNN architecture
    x = Conv2D(32, (3, 1), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Final classification layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    # Test model creation
    input_shape = [49, 1, 10]  # Example shape: 49 time steps, 1 channel, 10 mel coefficients
    num_classes = 3

    print("Creating streaming wake word model...")
    model = create_sww_model(input_shape, num_classes)
    model.summary()

    print(f"\nModel input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    print("\nCreating simple model for comparison...")
    simple_model = create_simple_sww_model(input_shape, num_classes)
    simple_model.summary()

    print(f"\nSimple model total parameters: {simple_model.count_params():,}")