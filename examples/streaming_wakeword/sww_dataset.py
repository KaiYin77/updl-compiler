#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Streaming Wake Word (SWW) dataset utilities for audio data
"""

import os
import tensorflow as tf
import numpy as np
import glob
import functools
from typing import List, Tuple, Dict, Any

# Default wake word labels for SWW
SWW_LABELS = ["marvin", "silence", "unknown"]

def decode_audio(audio_binary):
    """Decode WAV-encoded audio files to float32 tensors"""
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Ensure we get a 1D tensor by squeezing all extra dimensions
    audio = tf.squeeze(audio)
    return audio

def get_label(file_path):
    """Extract label from file path (parent directory name)"""
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    """Load waveform and extract label from file path"""
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return {'audio': waveform, 'label': label}

def preprocess_audio(audio_dict, target_length=16000):
    """Preprocess audio: pad/truncate to target length"""
    audio = audio_dict['audio']
    label = audio_dict['label']

    # Ensure audio is 1D
    audio = tf.squeeze(audio)

    # Pad or truncate to target length
    audio_length = tf.shape(audio)[0]
    if audio_length < target_length:
        padding = target_length - audio_length
        audio = tf.pad(audio, [[0, padding]])
    else:
        audio = audio[:target_length]

    # Keep as float32 for feature extraction
    audio = tf.cast(audio, tf.float32)

    return {'audio': audio, 'label': label}

def extract_log_mel_features(audio_dict,
                           sample_rate=16000,
                           window_size_ms=64.0,
                           window_stride_ms=32.0,
                           dct_coefficient_count=40):
    """Extract log Mel-frequency filterbank energies from audio (matching benchmark)"""
    audio = audio_dict['audio']
    label = audio_dict['label']

    # Ensure audio is 1D
    audio = tf.squeeze(audio)

    # Calculate window parameters
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # Apply preemphasis
    preemphasis_coef = 1 - 2 ** -5
    power_offset = 52

    # Pad for preemphasis - ensure we have the right shape
    paddings = [[1, 0]]  # For 1D tensor
    audio = tf.pad(tensor=audio, paddings=paddings, mode='CONSTANT')
    audio = audio[1:] - preemphasis_coef * audio[:-1]

    # Compute STFT
    stfts = tf.signal.stft(
        audio,
        frame_length=window_size_samples,
        frame_step=window_stride_samples,
        fft_length=None,
        window_fn=functools.partial(tf.signal.hamming_window, periodic=False),
        pad_end=False
    )

    # Compute magnitude spectrum
    magspec = tf.abs(stfts)
    num_spectrogram_bins = magspec.shape[-1]

    # Compute power spectrum
    powspec = (1.0 / window_size_samples) * tf.square(magspec)
    powspec_max = tf.reduce_max(input_tensor=powspec)
    powspec = tf.clip_by_value(powspec, 1e-30, powspec_max)

    # Create Mel filter bank
    lower_edge_hertz = 0.0
    upper_edge_hertz = sample_rate / 2.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=dct_coefficient_count,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )

    # Apply Mel filter bank
    mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Convert to log scale
    log_mel_spec = 10 * tf.math.log(mel_spectrograms) / tf.math.log(10.0)

    # Add channel dimension and normalize
    log_mel_spec = tf.expand_dims(log_mel_spec, -2)
    log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
    log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

    return {'audio': log_mel_spec, 'label': label}

def create_label_mapping(num_classes=3):
    """Create label mapping for wake word classification"""
    if num_classes == 3:
        label_strings = ["marvin", "silence", "unknown"]
    elif num_classes == 2:
        label_strings = ["marvin", "other"]
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(label_strings),
            values=tf.constant(range(num_classes)),
        ),
        default_value=tf.constant(num_classes-1),
        name="sww_label_mapping"
    )

def get_file_lists(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Get train, test, and validation file lists from speech commands dataset"""
    # Get all WAV files
    filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav'))

    # Load validation list
    val_files_path = os.path.join(data_dir, 'validation_list.txt')
    if os.path.exists(val_files_path):
        with open(val_files_path) as f:
            val_files = f.read().splitlines()
        val_files = [os.path.join(data_dir, fn).rstrip() for fn in val_files]
    else:
        val_files = []

    # Load test list
    test_files_path = os.path.join(data_dir, 'testing_list.txt')
    if os.path.exists(test_files_path):
        with open(test_files_path) as f:
            test_files = f.read().splitlines()
        test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]
    else:
        test_files = []

    # Fix path separators for Windows
    if os.sep != '/':
        val_files = [fn.replace('/', os.sep) for fn in val_files]
        test_files = [fn.replace('/', os.sep) for fn in test_files]

    # Filter out background noise files and create training set
    train_files = [f for f in filenames if f.split(os.sep)[-2][0] != '_']
    train_files = list(set(train_files) - set(test_files) - set(val_files))

    return train_files, test_files, val_files

def create_dataset(file_list: List[str],
                  num_classes: int = 3,
                  batch_size: int = 100,  # benchmark default
                  shuffle: bool = True,
                  target_length: int = 16000,
                  sample_rate: int = 16000,
                  window_size_ms: float = 64.0,
                  window_stride_ms: float = 32.0,
                  dct_coefficient_count: int = 40) -> tf.data.Dataset:
    """Create a TensorFlow dataset from file list with feature extraction"""

    # Create dataset from file paths
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    # Load audio and labels
    ds = ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Create label mapping
    label_map = create_label_mapping(num_classes)

    # Apply label mapping
    ds = ds.map(
        lambda d: {"audio": d["audio"], "label": label_map.lookup(d['label'])},
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Preprocess audio (pad/truncate)
    ds = ds.map(
        lambda d: preprocess_audio(d, target_length),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Extract log Mel features
    ds = ds.map(
        lambda d: extract_log_mel_features(
            d,
            sample_rate=sample_rate,
            window_size_ms=window_size_ms,
            window_stride_ms=window_stride_ms,
            dct_coefficient_count=dct_coefficient_count
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert to (features, labels) format for Keras
    ds = ds.map(
        lambda d: (d['audio'], tf.one_hot(d['label'], depth=num_classes, axis=-1)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache for performance (after feature extraction)
    ds = ds.cache()

    # Shuffle if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # Batch
    ds = ds.batch(batch_size)

    return ds

def load_speech_commands_data(data_dir: str,
                            batch_size: int = 100,  # benchmark default
                            num_classes: int = 3,
                            sample_rate: int = 16000,
                            window_size_ms: float = 64.0,
                            window_stride_ms: float = 32.0,
                            dct_coefficient_count: int = 40) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, test, and validation datasets from speech commands"""

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Get file lists
    train_files, test_files, val_files = get_file_lists(data_dir)

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(test_files)} test files")
    print(f"Found {len(val_files)} validation files")

    # Create datasets with feature extraction parameters
    train_ds = create_dataset(
        train_files, num_classes, batch_size, shuffle=True,
        sample_rate=sample_rate, window_size_ms=window_size_ms,
        window_stride_ms=window_stride_ms, dct_coefficient_count=dct_coefficient_count
    )
    test_ds = create_dataset(
        test_files, num_classes, batch_size, shuffle=False,
        sample_rate=sample_rate, window_size_ms=window_size_ms,
        window_stride_ms=window_stride_ms, dct_coefficient_count=dct_coefficient_count
    )
    val_ds = create_dataset(
        val_files, num_classes, batch_size, shuffle=False,
        sample_rate=sample_rate, window_size_ms=window_size_ms,
        window_stride_ms=window_stride_ms, dct_coefficient_count=dct_coefficient_count
    )

    return train_ds, test_ds, val_ds
