#!/usr/bin/env python3

import tensorflow as tf


def build_autoencoder(input_dim: int, bottleneck_dim: int = 8) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(128)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for _ in range(3):
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(bottleneck_dim)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for _ in range(4):
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    outputs = tf.keras.layers.Dense(input_dim)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
