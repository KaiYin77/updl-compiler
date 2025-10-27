import tensorflow as tf
from sww_dataset import get_file_lists, create_label_mapping, get_label
import os

DATASET_DIR = "/home/kaiyin-upbeat/data/speech_commands/0.0.2"

train_files, test_files, val_files = get_file_lists(DATASET_DIR)
label_map = create_label_mapping(3)

def debug_labels(file_list):
    labels = [get_label(f).numpy().decode("utf-8") for f in file_list]
    mapped = [label_map.lookup(tf.constant(l)).numpy() for l in labels]
    summary = {}
    for l, m in zip(labels, mapped):
        summary.setdefault(l, 0)
        summary[l] += 1
    return summary

summary = debug_labels(val_files)
print(summary)
