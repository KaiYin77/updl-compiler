#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""Training script for streaming wake word detection model.

This script provides a minimal viable solution for training a wake word detection
model based on the benchmark streaming_wakeword implementation, adapted to follow
the patterns used in visual_wake_words example.
"""

import os
import tensorflow as tf

# Import our custom modules
from sww_dataset import load_speech_commands_data
from sww_model import create_sww_model

# Configuration - matching benchmark/training/streaming_wakeword defaults
DATASET_DIR = "/home/kaiyin-upbeat/data/speech_commands/0.0.2"
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 64.0  # benchmark default
WINDOW_STRIDE_MS = 32.0  # benchmark default
DCT_COEFFICIENT_COUNT = 40  # benchmark default
NUM_CLASSES = 3  # marvin, silence, unknown
BATCH_SIZE = 100  # benchmark default
EPOCHS = 65  # benchmark default
PRETRAIN_EPOCHS = 50  # benchmark default
LEARNING_RATE = 0.001
USE_QAT = False  # benchmark default

def train_epochs(model, train_dataset, val_dataset, epoch_count, learning_rate):
    """Train the model for a given number of epochs with benchmark-matching compilation"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(class_id=0, thresholds=0.95, name='precision'),
            tf.keras.metrics.Recall(class_id=0, thresholds=0.95, name='recall'),
        ]
    )

    model.fit(
        train_dataset,
        epochs=epoch_count,
        validation_data=val_dataset,
        verbose=1
    )

    return model

def main():
    """Main training function"""
    print("Streaming Wake Word Training")
    print("=" * 40)
    print(f"Data directory: {DATASET_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Check if data directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"\nERROR: Data directory does not exist: {DATASET_DIR}")
        print("Please download the Speech Commands dataset first.")
        return

    # Calculate model settings
    desired_samples = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)
    window_size_samples = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
    window_stride_samples = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    print(f"\nModel configuration:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Clip duration: {CLIP_DURATION_MS} ms ({desired_samples} samples)")
    print(f"  Window size: {WINDOW_SIZE_MS} ms ({window_size_samples} samples)")
    print(f"  Window stride: {WINDOW_STRIDE_MS} ms ({window_stride_samples} samples)")
    print(f"  Spectrogram length: {spectrogram_length}")
    print(f"  Mel coefficients: {DCT_COEFFICIENT_COUNT}")

    # Load datasets
    print("\nLoading datasets...")
    try:
        train_ds, test_ds, val_ds = load_speech_commands_data(
            DATASET_DIR,
            batch_size=BATCH_SIZE,
            num_classes=NUM_CLASSES,
            sample_rate=SAMPLE_RATE,
            window_size_ms=WINDOW_SIZE_MS,
            window_stride_ms=WINDOW_STRIDE_MS,
            dct_coefficient_count=DCT_COEFFICIENT_COUNT
        )
        print("Datasets loaded successfully!")
    except Exception as e:
        print(f"ERROR loading datasets: {e}")
        return

    # Create model
    print("\nCreating model...")
    model = create_sww_model(
        input_shape=[spectrogram_length, 1, DCT_COEFFICIENT_COUNT],
        num_classes=NUM_CLASSES,
        l2_reg=0.001  # benchmark default
    )

    model.summary()

    # Train model following benchmark pattern
    print("\nStarting training...")

    # Stage 1: Pretraining without QAT
    if PRETRAIN_EPOCHS > 0:
        print(f"Stage 1: Pretraining {PRETRAIN_EPOCHS} epochs without QAT")
        model = train_epochs(model, train_ds, val_ds, PRETRAIN_EPOCHS, LEARNING_RATE)

    # Stage 2: QAT training (if enabled and epochs remaining)
    qat_epochs = EPOCHS - PRETRAIN_EPOCHS
    if USE_QAT and qat_epochs > 0:
        print(f"Stage 2: QAT training {qat_epochs} epochs")
        # Apply QAT to the model
        try:
            import tensorflow_model_optimization as tfmot
            from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_scheme import DefaultNBitQuantizeScheme

            annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
            with tfmot.quantization.keras.quantize_scope():
                qat_model = tfmot.quantization.keras.quantize_apply(
                    annotated_model,
                    scheme=DefaultNBitQuantizeScheme(
                        disable_per_axis=False,
                        num_bits_weight=8,
                        num_bits_activation=8,
                    ),
                )
            model = train_epochs(qat_model, train_ds, val_ds, qat_epochs, LEARNING_RATE)
            print("QAT training completed")
        except ImportError:
            print("Warning: TensorFlow Model Optimization not available, skipping QAT")
            if qat_epochs > 0:
                model = train_epochs(model, train_ds, val_ds, qat_epochs, LEARNING_RATE)
    elif qat_epochs > 0:
        print(f"Stage 2: Additional training {qat_epochs} epochs without QAT")
        model = train_epochs(model, train_ds, val_ds, qat_epochs, LEARNING_RATE)

    # Save model
    model_save_path = 'ref_model'
    print(f"\nSaving model to: {model_save_path}")
    model.save(model_save_path)
    print("Model saved successfully!")
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()