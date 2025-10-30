#!/usr/bin/env python3

import os

import numpy as np

from ad_preprocessor import AnomalyDetectionPreprocessor

DATASET_DIR = "/home/kaiyin-upbeat/data/anomaly_detection/dev_data/ToyCar/train"
OUTPUT_FILE = "ad_quantize_calibrate_idxs.txt"
CALIBRATION_SAMPLES = 120
SEED = 7


def main() -> None:
    preprocessor = AnomalyDetectionPreprocessor()
    audio_files = preprocessor.list_audio_files(DATASET_DIR)
    if not audio_files:
        raise RuntimeError(f"No wav files found under {DATASET_DIR}. Run download.sh first.")

    window_counts = [preprocessor.count_windows_in_file(path) for path in audio_files]
    total_windows = sum(window_counts)
    if total_windows == 0:
        raise RuntimeError("Dataset produced zero frame windows; cannot sample calibration set.")

    rng = np.random.default_rng(SEED)
    sample_total = min(CALIBRATION_SAMPLES, total_windows)
    replace = total_windows < CALIBRATION_SAMPLES
    selected = rng.choice(total_windows, size=sample_total, replace=replace)
    selected = np.unique(selected)

    np.savetxt(OUTPUT_FILE, selected.astype(int), fmt="%d")

    print(f"Found {total_windows} windows across {len(audio_files)} files.")
    print(f"Wrote {len(selected)} calibration indices to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
