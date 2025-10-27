import os
from collections import defaultdict

import numpy as np

from sww_dataset import SWW_LABELS, get_file_lists

# Configuration
DATASET_DIR = "/home/kaiyin-upbeat/data/speech_commands/0.0.2"
CAL_SAMPLES_PER_CLASS = 15
OUTPUT_FILE = "sww_quantize_calibrate_idxs.txt"
RNG = np.random.default_rng(seed=1)


def label_from_path(path: str) -> str:
    """Infer wake word label from file path."""
    parent = os.path.basename(os.path.dirname(path))
    if parent == "marvin":
        return "marvin"
    if parent == "silence":
        return "silence"
    return "unknown"


def safe_choice(indices, n, rng):
    """Select indices safely, allowing replacement if class is small."""
    if not indices:
        print("⚠️  Warning: class has no samples, skipping.")
        return np.array([], dtype=int)
    take = min(len(indices), n)
    replace = len(indices) < n
    return rng.choice(indices, size=take, replace=replace)


def main():
    train_files, test_files, val_files = get_file_lists(DATASET_DIR)

    ordered_files = []
    for file_list in (train_files, val_files, test_files):
        ordered_files.extend(sorted(file_list))

    if not ordered_files:
        raise RuntimeError(f"No audio files found under {DATASET_DIR}")

    class_to_indices = defaultdict(list)
    for idx, file_path in enumerate(ordered_files):
        label = label_from_path(file_path)
        try:
            label_idx = SWW_LABELS.index(label)
        except ValueError:
            label_idx = SWW_LABELS.index("unknown")
        class_to_indices[label_idx].append(idx)

    for class_idx in range(len(SWW_LABELS)):
        count = len(class_to_indices.get(class_idx, []))
        print(f"Class {class_idx} ({SWW_LABELS[class_idx]}): {count} samples")

    selections = []
    for class_idx in range(len(SWW_LABELS)):
        indices = class_to_indices.get(class_idx, [])
        chosen = safe_choice(indices, CAL_SAMPLES_PER_CLASS, RNG)
        selections.append(chosen)

    if not selections:
        raise RuntimeError("No calibration indices selected; aborting.")

    cal_idxs = np.concatenate([sel for sel in selections if sel.size > 0])
    cal_idxs.sort()

    with open(OUTPUT_FILE, "w") as fpo:
        for idx in cal_idxs:
            fpo.write(f"{int(idx)}\n")

    print(f"Wrote {len(cal_idxs)} calibration indices to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
