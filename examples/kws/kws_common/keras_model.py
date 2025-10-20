#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Flatten,
    BatchNormalization,
    Dropout,
    Reshape,
)
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.regularizers import l2
from .model_utils import prepare_model_settings


def get_model(args):
    model_name = args.model_architecture

    label_count = 12
    model_settings = prepare_model_settings(label_count, args)

    if model_name == "fc4":
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(
                    input_shape=(
                        model_settings["spectrogram_length"],
                        model_settings["dct_coefficient_count"],
                    )
                ),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    model_settings["label_count"], activation="softmax"
                ),
            ]
        )

    elif model_name == "ds_cnn":
        print("DS CNN model invoked")
        input_shape = [
            model_settings["spectrogram_length"],
            model_settings["dct_coefficient_count"],
            1,
        ]
        filters = 64
        weight_decay = 1e-4
        regularizer = l2(weight_decay)
        final_pool_size = (int(input_shape[0] / 2), int(input_shape[1] / 2))

        # Model layers
        # Input pure conv2d
        inputs = Input(shape=input_shape)
        x = Conv2D(
            filters,
            (10, 4),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizer,
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.2)(x)

        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Second layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Third layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Fourth layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Reduce size and apply final softmax
        x = Dropout(rate=0.4)(x)

        x = AveragePooling2D(pool_size=final_pool_size)(x)
        x = Flatten()(x)
        outputs = Dense(model_settings["label_count"], activation="softmax")(x)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

    elif model_name == "td_cnn":
        print("TD CNN model invoked")
        input_shape = [
            model_settings["spectrogram_length"],
            model_settings["dct_coefficient_count"],
            1,
        ]
        print(f"Input shape = {input_shape}")
        filters = 64
        weight_decay = 1e-4
        regularizer = l2(weight_decay)

        # Model layers
        # Input time-domain conv
        inputs = Input(shape=input_shape)
        x = Conv2D(
            filters,
            (512, 1),
            strides=(384, 1),
            padding="valid",
            kernel_regularizer=regularizer,
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.2)(x)
        x = Reshape((41, 64, 1))(x)

        # True conv
        x = Conv2D(
            filters,
            (10, 4),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.2)(x)

        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Second layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Third layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Fourth layer of separable depthwise conv2d
        x = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Reduce size and apply final softmax
        x = Dropout(rate=0.4)(x)

        # x = AveragePooling2D(pool_size=(25,5))(x)
        x = GlobalAveragePooling2D()(x)

        x = Flatten()(x)
        outputs = Dense(model_settings["label_count"], activation="softmax")(x)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError("Model name {:} not supported".format(model_name))

    model.compile(
        # optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model
