#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Training script for the DANet (Dynamic Adaptive Network) example.

This script demonstrates training a neural network to predict adaptive beta
parameters for the Madgwick filter used in IMU sensor fusion applications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from danet_model import DANetModelConfig, build_danet_model, madgwick_step_tf
from danet_preprocessor import DANetPreprocessor

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/kaiyin-upbeat/data")
OUTPUT_DIR = EXAMPLE_DIR / "ref_model"
EPOCHS = 150
BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
SEQ_LEN = 100
PENALTY_THRESHOLD = 0.1
CACHE_DATASET = False
AUTOTUNE = tf.data.AUTOTUNE

# Training parameters for progressive difficulty
ACC_FACTOR = 9.81
GYRO_FACTOR = 4.0
DIFF_FACTOR = 5.0
LPF_ALPHA_RISE = 0.5
LPF_ALPHA_FALL = 0.01


def softsign_normalize(x, factor):
    """Apply softsign normalization"""
    scaled = x / factor
    return scaled / (1 + np.abs(scaled))


def build_tensorflow_datasets(
    data_path: Path,
    preprocessor: DANetPreprocessor,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build training and validation datasets from CSV file"""

    # Load the IMU data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # Prepare the data arrays
    phys_acc = df[['ax', 'ay', 'az']].values.astype(np.float32)
    phys_gyro = df[['gx', 'gy', 'gz']].values.astype(np.float32)
    phys_dt = np.full((len(df), 1), 0.005, dtype=np.float32)  # 200Hz sampling
    gt_quats = df[['gt_q0', 'gt_q1', 'gt_q2', 'gt_q3']].values.astype(np.float32)

    # Normalize inputs using softsign
    nn_acc_base = softsign_normalize(phys_acc, ACC_FACTOR)
    nn_gyro_base = softsign_normalize(phys_gyro, GYRO_FACTOR)

    # Split into train/val (80/20 split)
    split_idx = int(0.8 * len(df))

    train_data = {
        'phys_acc': phys_acc[:split_idx],
        'phys_gyro': phys_gyro[:split_idx],
        'nn_acc': nn_acc_base[:split_idx],
        'nn_gyro': nn_gyro_base[:split_idx],
        'dt': phys_dt[:split_idx],
        'gt_quats': gt_quats[:split_idx]
    }

    val_data = {
        'phys_acc': phys_acc[split_idx:],
        'phys_gyro': phys_gyro[split_idx:],
        'nn_acc': nn_acc_base[split_idx:],
        'nn_gyro': nn_gyro_base[split_idx:],
        'dt': phys_dt[split_idx:],
        'gt_quats': gt_quats[split_idx:]
    }

    def data_generator(data_dict):
        """Generator for creating individual samples from the data"""
        num_samples = len(data_dict['phys_acc']) - 1
        indices = np.random.permutation(num_samples)

        for idx in indices:
            # Calculate acceleration difference
            acc_norm = np.linalg.norm(data_dict['phys_acc'][idx])
            acc_diff = abs(acc_norm - 9.81)
            nn_diff = softsign_normalize(np.array([acc_diff]), DIFF_FACTOR)[0]

            # Create input features: [acc(3), gyro(3), diff(1), beta_prev(1), dt(1)]
            beta_prev = 0.1  # Default beta value

            feature_vec = np.concatenate([
                data_dict['nn_acc'][idx],
                data_dict['nn_gyro'][idx],
                [nn_diff],
                [beta_prev],
                data_dict['dt'][idx]
            ])

            # Target is optimal beta (simplified - use constant for demonstration)
            target_beta = 0.1  # In real training, this would be computed from quaternion error

            yield feature_vec.astype(np.float32), np.array([target_beta], dtype=np.float32)

    # Create TensorFlow datasets
    def make_tf_dataset(data_dict):
        return tf.data.Dataset.from_generator(
            lambda: data_generator(data_dict),
            output_signature=(
                tf.TensorSpec(shape=(9,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32)
            )
        ).batch(batch_size).prefetch(AUTOTUNE)

    train_dataset = make_tf_dataset(train_data)
    val_dataset = make_tf_dataset(val_data)

    return train_dataset, val_dataset


def main() -> None:
    """Main training function"""
    data_dir = DATA_DIR.expanduser()
    output_dir = OUTPUT_DIR.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the reference CSV file for training
    csv_path = Path(__file__).parent.parent.parent / "references" / "ready_for_training.csv"
    if not csv_path.exists():
        print(f"Training data not found at {csv_path}")
        print("Please ensure the ready_for_training.csv file is available")
        return

    model_config = DANetModelConfig(
        input_dim=9,
        hidden_dim=128,
        learning_rate=LEARNING_RATE,
    )

    preprocessor = DANetPreprocessor(
        acc_factor=ACC_FACTOR,
        gyro_factor=GYRO_FACTOR,
        diff_factor=DIFF_FACTOR,
        lpf_alpha_rise=LPF_ALPHA_RISE,
        lpf_alpha_fall=LPF_ALPHA_FALL,
    )

    print("Building datasets...")
    train_dataset, val_dataset = build_tensorflow_datasets(
        data_path=csv_path,
        preprocessor=preprocessor,
        batch_size=BATCH_SIZE,
    )

    print("Building model...")
    model = build_danet_model(model_config)
    model.summary()

    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "danet_best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the trained model
    model.save(str(output_dir))
    print(f"Saved trained model to {output_dir}")

    # Print training summary
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final training loss: {final_train_loss:.6f}")
    print(f"Final validation loss: {final_val_loss:.6f}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()