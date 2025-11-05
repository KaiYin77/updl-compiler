#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0
"""Training script for the keyword spotting example.

The flow mirrors `examples/image_classification/ic_train.py` with the model
topologies borrowed from the MLCommons Tiny reference implementation
(`up-tiny/benchmark/training/keyword_spotting`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from kws_model import KWSModelConfig, build_kws_model, prepare_model_settings
from kws_preprocessor import KWSPreprocessor, WORD_LABELS

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/kaiyin-upbeat/data")
OUTPUT_DIR = EXAMPLE_DIR / "ref_model"
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_ARCH = "ds_cnn"
CACHE_DATASET = False
RUN_TEST_SET = False
AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(
    data_dir: Path,
    preprocessor: KWSPreprocessor,
    batch_size: int,
    cache: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    splits = ["train", "validation", "test"]
    ds_splits, info = tfds.load(
        "speech_commands",
        split=splits,
        data_dir=str(data_dir),
        with_info=True,
        as_supervised=False,
    )
    ds_train_raw, ds_val_raw, ds_test_raw = ds_splits

    def map_fn(sample: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        features = preprocessor.preprocess_sample(sample)
        features = tf.squeeze(features, axis=0)
        label = tf.cast(sample["label"], tf.int32)
        return features, label

    def prepare(
        dataset: tf.data.Dataset,
        shuffle_size: int,
        training: bool = False,
    ) -> tf.data.Dataset:
        ds = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_count = info.splits["train"].num_examples
    val_count = info.splits["validation"].num_examples
    test_count = info.splits["test"].num_examples

    ds_train = prepare(ds_train_raw, train_count, training=True)
    ds_val = prepare(ds_val_raw, val_count, training=False)
    ds_test = prepare(ds_test_raw, test_count, training=False)
    return ds_train, ds_val, ds_test, info


def main() -> None:
    data_dir = DATA_DIR.expanduser()
    output_dir = OUTPUT_DIR.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = KWSModelConfig(
        learning_rate=LEARNING_RATE,
        model_architecture=MODEL_ARCH,
        label_count=len(WORD_LABELS),
    )
    settings = prepare_model_settings(model_config)
    print("Model settings:", settings)

    preprocessor = KWSPreprocessor(
        sample_rate=model_config.sample_rate,
        clip_duration_ms=model_config.clip_duration_ms,
        window_size_ms=model_config.window_size_ms,
        window_stride_ms=model_config.window_stride_ms,
        dct_coefficient_count=model_config.dct_coefficient_count,
    )

    ds_train, ds_val, ds_test, info = build_datasets(
        data_dir=data_dir,
        preprocessor=preprocessor,
        batch_size=BATCH_SIZE,
        cache=CACHE_DATASET,
    )

    print("Dataset summary:")
    for split_name in ("train", "validation", "test"):
        count = info.splits[split_name].num_examples
        print(f"  {split_name}: {count} examples")

    model = build_kws_model(model_config)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_sparse_categorical_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "kws_best.weights.h5"),
            monitor="val_sparse_categorical_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Persist the trained model as a SavedModel directory.
    model.save(str(output_dir))
    print(f"Saved trained model to {output_dir}")

    if RUN_TEST_SET:
        test_loss, test_acc = model.evaluate(ds_test)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

    # Expose history metrics for downstream plotting.
    print("Training metrics keys:", list(history.history.keys()))


if __name__ == "__main__":
    main()
