#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Generator, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from ad_model import build_autoencoder
from ad_preprocessor import AnomalyDetectionPreprocessor


def default_data_root() -> Path:
    """Resolve the dataset root to match the baseline training script layout."""
    env_override = os.environ.get("AD_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / "data" / "anomaly_detection" / "dev_data"


DEFAULT_MACHINE_TYPE = "ToyCar"
DEFAULT_MODEL_DIR = Path("ref_model")
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 512
DEFAULT_VAL_SPLIT = 0.1
SHUFFLE_BUFFER_MIN = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train anomaly detection autoencoder without loading all features into memory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_data_root(),
        help="Path to dev_data directory (default: ~/data/anomaly_detection/dev_data or $AD_DATA_ROOT).",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default=DEFAULT_MACHINE_TYPE,
        help="Machine type to train on (e.g. ToyCar, ToyTrain).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Validation split ratio (0.0-0.5).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam optimizer learning rate.")
    return parser.parse_args()


def _validate_dataset_dir(dataset_root: Path, machine_type: str) -> Path:
    target_dir = dataset_root.expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {target_dir}")

    machine_dir = target_dir / machine_type
    train_dir = machine_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected training directory at {train_dir}")
    return train_dir


def _enumerate_file_windows(
    preprocessor: AnomalyDetectionPreprocessor,
    file_paths: Sequence[str],
) -> List[Tuple[str, int]]:
    counts: List[Tuple[str, int]] = []
    for path in file_paths:
        num_windows = preprocessor.count_windows_in_file(path)
        if num_windows == 0:
            continue
        counts.append((path, num_windows))
    if not counts:
        raise RuntimeError("No usable training windows found.")
    return counts


def _split_counts(num_windows: int, val_split: float) -> Tuple[int, int]:
    if val_split <= 0.0:
        return num_windows, 0
    val_windows = int(round(num_windows * val_split))
    val_windows = min(val_windows, num_windows)
    train_windows = num_windows - val_windows
    if train_windows == 0 and val_windows > 0:
        train_windows = num_windows - 1
        val_windows = 1
    return train_windows, val_windows


def _window_generator(
    preprocessor: AnomalyDetectionPreprocessor,
    split_plan: Sequence[Tuple[str, int, int]],
    take_validation: bool,
) -> Generator[np.ndarray, None, None]:
    for file_path, train_count, val_count in split_plan:
        target_count = val_count if take_validation else train_count
        if target_count <= 0:
            continue
        windows = preprocessor.extract_feature_windows(file_path)
        if windows.size == 0:
            continue

        if take_validation:
            start = train_count
            stop = train_count + val_count
        else:
            start = 0
            stop = train_count

        for window in windows[start:stop]:
            yield window.astype(np.float32, copy=False)


def build_datasets(
    preprocessor: AnomalyDetectionPreprocessor,
    dataset_dir: Path,
    batch_size: int,
    val_split: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset | None, int, int]:
    file_paths = preprocessor.list_audio_files(str(dataset_dir))
    if not file_paths:
        raise RuntimeError(f"No wav files found under {dataset_dir}")

    counts = _enumerate_file_windows(preprocessor, file_paths)
    split_plan = []
    train_total = 0
    val_total = 0
    for file_path, window_count in counts:
        train_count, val_count = _split_counts(window_count, val_split)
        split_plan.append((file_path, train_count, val_count))
        train_total += train_count
        val_total += val_count

    if train_total == 0:
        raise RuntimeError("No training windows available after split.")

    train_ds = tf.data.Dataset.from_generator(
        lambda: _window_generator(preprocessor, split_plan, take_validation=False),
        output_signature=tf.TensorSpec(shape=(preprocessor.feature_dim,), dtype=tf.float32),
    )

    shuffle_buffer = max(batch_size, min(train_total, max(SHUFFLE_BUFFER_MIN, batch_size * 8)))
    train_ds = (
        train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds: tf.data.Dataset | None = None
    if val_total > 0:
        val_ds = tf.data.Dataset.from_generator(
            lambda: _window_generator(preprocessor, split_plan, take_validation=True),
            output_signature=tf.TensorSpec(shape=(preprocessor.feature_dim,), dtype=tf.float32),
        )
        val_ds = (
            val_ds.batch(batch_size, drop_remainder=False)
            .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    return train_ds, val_ds, train_total, val_total


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.val_split < 0.5):
        raise ValueError("Validation split must be between 0.0 (inclusive) and 0.5 (exclusive).")

    train_dir = _validate_dataset_dir(args.dataset_root, args.machine_type)
    print(f"Using training data at {train_dir}")

    preprocessor = AnomalyDetectionPreprocessor()
    train_ds, val_ds, train_total, val_total = build_datasets(
        preprocessor=preprocessor,
        dataset_dir=train_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    print(f"Prepared {train_total} training windows and {val_total} validation windows.")

    model = build_autoencoder(preprocessor.feature_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="mse")

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        verbose=1,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir, overwrite=True)
    print(f"Saved trained model to {args.model_dir.resolve()}")

    if history.history:
        last_loss = history.history.get("loss", [None])[-1]
        last_val_loss = history.history.get("val_loss", [None])[-1]
        print(f"Final loss: {last_loss}, final val_loss: {last_val_loss}")


if __name__ == "__main__":
    main()
