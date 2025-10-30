#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from updl_compiler import compile_model

from ad_preprocessor import AnomalyDetectionPreprocessor


def main() -> int:
    model_path = "ref_model"
    dataset_dir = "/home/kaiyin-upbeat/data/anomaly_detection/dev_data/ToyCar/train"
    calibration_indices_file = "ad_quantize_calibrate_idxs.txt"

    if not os.path.exists(model_path):
        print(f"Model directory '{model_path}' not found. Train the model first.")
        return 1

    preprocessor = AnomalyDetectionPreprocessor()
    calibration_data = {
        "dataset_dir": dataset_dir,
        "indices_file": calibration_indices_file if os.path.exists(calibration_indices_file) else None,
    }

    compile_model(
        model=model_path,
        preprocessor=preprocessor,
        calibration_data=calibration_data,
        model_name="ad_ref_model",
        description="ToyCar anomaly detection autoencoder",
    )

    print("Compilation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
