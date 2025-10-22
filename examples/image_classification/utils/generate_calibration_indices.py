#!/usr/bin/env python3
"""
Generate balanced calibration indices for CIFAR-10 dataset.

This script creates a balanced set of calibration indices by selecting an equal number
of samples from each of the 10 CIFAR-10 classes, similar to the reference performance
sampling approach.
"""

import numpy as np
import os
import sys

# Add parent directory to path to import ic_train
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from ic_train import load_cifar_10_data
except ImportError:
    print("Error: Could not import ic_train. Make sure ic_train.py exists in the parent directory.")
    sys.exit(1)

# Configuration
CIFAR_DIR = '/home/kaiyin-upbeat/data/cifar-10-batches-py'
SAMPLES_PER_CLASS = 12
RANDOM_SEED = 8108
USE_TRAIN_DATA = True
OUTPUT_FILE = '../ic_quantize_calibrate_idxs.txt'


def generate_balanced_indices() -> np.ndarray:
    """
    Generate balanced calibration indices from CIFAR-10 dataset.

    Returns:
        np.ndarray: Array of selected indices
    """

    print(f"Loading CIFAR-10 dataset from: {CIFAR_DIR}")

    # Load CIFAR-10 data
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(CIFAR_DIR)

    # Choose which dataset to use for calibration
    if USE_TRAIN_DATA:
        data_labels = train_labels
        data_type = "training"
        print(f"Using training data ({len(train_data)} samples)")
    else:
        data_labels = test_labels
        data_type = "test"
        print(f"Using test data ({len(test_data)} samples)")

    # Convert one-hot labels to class indices
    class_indices = data_labels.argmax(axis=-1)
    all_idxs = np.arange(len(data_labels))

    print(f"CIFAR-10 classes: {[name.decode('utf-8') if isinstance(name, bytes) else name for name in label_names]}")
    print(f"Selecting {SAMPLES_PER_CLASS} samples per class...")

    # Generate balanced indices
    balanced_idxs = []
    rng = np.random.default_rng(RANDOM_SEED)

    for label in range(10):
        # Find all samples of this class
        mask = class_indices == label
        class_sample_idxs = all_idxs[mask]

        if len(class_sample_idxs) < SAMPLES_PER_CLASS:
            print(f"Warning: Class {label} has only {len(class_sample_idxs)} samples, "
                  f"requested {SAMPLES_PER_CLASS}")
            selected_idxs = class_sample_idxs
        else:
            # Randomly select SAMPLES_PER_CLASS from this class
            selected_idxs = rng.choice(class_sample_idxs, size=SAMPLES_PER_CLASS, replace=False)

        balanced_idxs.append(selected_idxs)
        class_name = label_names[label].decode('utf-8') if isinstance(label_names[label], bytes) else label_names[label]
        print(f"  Class {label} ({class_name}): selected {len(selected_idxs)} samples")

    # Concatenate and shuffle all selected indices
    final_idxs = np.concatenate(balanced_idxs)
    rng.shuffle(final_idxs)

    print(f"\nGenerated {len(final_idxs)} balanced calibration indices")
    print(f"Index range: {final_idxs.min()} - {final_idxs.max()}")

    # Verify class distribution
    selected_classes = class_indices[final_idxs]
    print("\nClass distribution verification:")
    for label in range(10):
        count = np.sum(selected_classes == label)
        class_name = label_names[label].decode('utf-8') if isinstance(label_names[label], bytes) else label_names[label]
        print(f"  Class {label} ({class_name}): {count} samples")

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)

    print(f"\nSaving indices to: {output_path}")
    with open(output_path, 'w') as f:
        for idx in final_idxs:
            f.write(f"{idx}\n")

    return final_idxs


def main():
    # Check if CIFAR-10 directory exists
    if not os.path.exists(CIFAR_DIR):
        print(f"Error: CIFAR-10 directory not found: {CIFAR_DIR}")
        print("Please make sure the CIFAR-10 dataset is downloaded and extracted.")
        sys.exit(1)

    # Generate indices
    try:
        indices = generate_balanced_indices()
        print(f"\n✅ Successfully generated {len(indices)} balanced calibration indices!")

    except Exception as e:
        print(f"❌ Error generating indices: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()