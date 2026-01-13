#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""DANet (Dynamic Adaptive Network) model builders.

DANet is designed for adaptive sensor fusion in IMU applications,
dynamically adjusting the Madgwick filter's beta parameter based on
sensor conditions and motion patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@dataclass(frozen=True)
class DANetModelConfig:
    """Configuration for building a DANet model."""

    input_dim: int = 9  # [acc(3), gyro(3), diff(1), beta(1), dt(1)]
    hidden_dim: int = 128
    output_dim: int = 1  # Beta parameter
    learning_rate: float = 2e-3
    beta_min: float = 0.001
    beta_max: float = 0.301
    dropout: float = 0.1


def build_danet_model(config: DANetModelConfig | None = None) -> tf.keras.Model:
    """Return a compiled DANet model for adaptive sensor fusion."""
    config = config or DANetModelConfig()

    # Simple Sequential model with only supported layers (Dense)
    # Note: Output will be raw sigmoid [0,1], scaling must be done post-processing
    model = models.Sequential([
        # First Dense layer - takes input shape automatically
        layers.Dense(
            config.hidden_dim,
            activation='relu',
            input_shape=(config.input_dim,),
            name='fc1'
        ),

        # Second Dense layer
        layers.Dense(
            config.hidden_dim,
            activation='relu',
            name='fc2'
        ),

        # Output layer with sigmoid activation
        # Raw output [0,1] - will need post-processing: output * 0.3 + 0.001
        layers.Dense(1, activation='sigmoid', name='beta_output'),
    ], name="danet")

    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
    )
    return model




@tf.function
def madgwick_step_tf(q_prev, beta, acc, gyro, dt):
    """
    TensorFlow implementation of Madgwick filter step for efficient sensor fusion.

    Args:
        q_prev: Previous quaternion [batch_size, 4]
        beta: Adaptive beta parameter [batch_size, 1]
        acc: Accelerometer readings [batch_size, 3]
        gyro: Gyroscope readings [batch_size, 3]
        dt: Time step [batch_size, 1]

    Returns:
        Updated quaternion [batch_size, 4]
    """
    q0, q1, q2, q3 = q_prev[:, 0], q_prev[:, 1], q_prev[:, 2], q_prev[:, 3]
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]

    # Quaternion rate of change from gyroscope
    q_dot_w = 0.5 * (-q1*gx - q2*gy - q3*gz)
    q_dot_x = 0.5 * ( q0*gx + q2*gz - q3*gy)
    q_dot_y = 0.5 * ( q0*gy - q1*gz + q3*gx)
    q_dot_z = 0.5 * ( q0*gz + q1*gy - q2*gx)

    # Normalize accelerometer measurement
    norm_a = tf.sqrt(ax*ax + ay*ay + az*az + 1e-8)
    ax, ay, az = ax/norm_a, ay/norm_a, az/norm_a

    # Auxiliary variables to avoid repeated arithmetic
    _2q0, _2q1, _2q2, _2q3 = 2*q0, 2*q1, 2*q2, 2*q3
    _4q0, _4q1, _4q2 = 4*q0, 4*q1, 4*q2

    # Gradient descent algorithm corrective step
    f_0 = 2*(q1*q3 - q0*q2) - ax
    f_1 = 2*(q0*q1 + q2*q3) - ay
    f_2 = 1 - 2*(q1*q1 + q2*q2) - az

    grad_w = -_2q2 * f_0 + _2q1 * f_1
    grad_x =  _2q3 * f_0 + _2q0 * f_1 - _4q1 * f_2
    grad_y = -_2q0 * f_0 + _2q3 * f_1 - _4q2 * f_2
    grad_z =  _2q1 * f_0 + _2q2 * f_1

    # Normalize gradient
    norm_grad = tf.sqrt(grad_w*grad_w + grad_x*grad_x + grad_y*grad_y + grad_z*grad_z + 1e-8)
    grad_w, grad_x, grad_y, grad_z = grad_w/norm_grad, grad_x/norm_grad, grad_y/norm_grad, grad_z/norm_grad

    # Apply feedback step
    q_new_w = q0 + (q_dot_w - tf.squeeze(beta) * grad_w) * tf.squeeze(dt)
    q_new_x = q1 + (q_dot_x - tf.squeeze(beta) * grad_x) * tf.squeeze(dt)
    q_new_y = q2 + (q_dot_y - tf.squeeze(beta) * grad_y) * tf.squeeze(dt)
    q_new_z = q3 + (q_dot_z - tf.squeeze(beta) * grad_z) * tf.squeeze(dt)

    # Normalize quaternion
    norm_q = tf.sqrt(q_new_w*q_new_w + q_new_x*q_new_x + q_new_y*q_new_y + q_new_z*q_new_z + 1e-8)
    return tf.stack([q_new_w/norm_q, q_new_x/norm_q, q_new_y/norm_q, q_new_z/norm_q], axis=1)