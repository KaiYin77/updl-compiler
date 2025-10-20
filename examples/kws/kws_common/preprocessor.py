#!/usr/bin/env python
"""
Audio processing utilities for KWS models

Contains functions for loading audio files and extracting MFCC features.
"""

import tensorflow as tf
import numpy as np
import librosa
from .model_utils import prepare_model_settings

# Labels used in the model
WORD_LABELS = [
    "Down",
    "Go",
    "Left",
    "No",
    "Off",
    "On",
    "Right",
    "Stop",
    "Up",
    "Yes",
    "Silence",
    "Unknown",
]


def convert_wav_to_tensor(wav_file, flags):
    """Load audio using librosa to support both 16-bit and 32-bit WAV files"""
    try:
        # Calculate desired samples from model settings
        model_settings = prepare_model_settings(len(WORD_LABELS), flags)
        desired_samples = model_settings["desired_samples"]

        # Use librosa to load the audio
        audio, sr = librosa.load(
            wav_file,
            sr=flags.sample_rate,
            mono=True,
            duration=flags.clip_duration_ms / 1000,
        )

        # Ensure we have the exact length needed
        if len(audio) < desired_samples:
            # Pad if too short
            audio = np.pad(audio, (0, desired_samples - len(audio)), "constant")
        elif len(audio) > desired_samples:
            # Trim if too long
            audio = audio[:desired_samples]

        # Normalize to [-1.0, 1.0] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Convert numpy array to TensorFlow tensor
        wav = tf.convert_to_tensor(audio, dtype=tf.float32)

        print(f"Successfully loaded audio from {wav_file}")
        return wav

    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def convert_wav_to_mfcc_features_tensor(wav, flags, debug=False):
    """Extract MFCC features using TensorFlow to match the training pipeline exactly

    Args:
        wav: Audio waveform tensor
        flags: Configuration flags
        debug: Enable debug logging for inference scripts
    """
    # Prepare model settings
    model_settings = prepare_model_settings(len(WORD_LABELS), flags)

    # Normalize audio
    wav = tf.cast(wav, tf.float32)
    max_val = tf.reduce_max(tf.abs(wav))
    wav = wav / (max_val + 1e-6)

    if debug:
        print(f"[PREPROCESSING] Raw samples (first 10): {wav.numpy()[:10]}")

    # Apply time offset (matching the training pipeline)
    padded_wav = tf.pad(wav, [[2, 2]], mode="CONSTANT")
    shifted_wav = tf.slice(padded_wav, [2], [model_settings["desired_samples"]])

    # Compute STFT with Hann window
    stfts = tf.signal.stft(
        shifted_wav,
        frame_length=model_settings["window_size_samples"],
        frame_step=model_settings["window_stride_samples"],
        window_fn=tf.signal.hann_window,
    )

    spectrogram = tf.abs(stfts)
    if debug:
        print(
            f"[PREPROCESSING] Spectrogram shape: {spectrogram.shape}, max: {np.max(spectrogram.numpy())}"
        )

    # Compute Mel spectrogram
    num_spectrogram_bins = tf.shape(stfts)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=model_settings["sample_rate"],
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0,
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([40]))

    # Compute log-mel spectrogram and extract MFCCs
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    if debug:
        print(
            f"[PREPROCESSING] Log-mel spectrogram max: {np.max(log_mel_spectrogram.numpy())}"
        )

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., : model_settings["dct_coefficient_count"]]

    if debug:
        print(
            f"[PREPROCESSING] MFCCs shape: {mfccs.shape}, max: {np.max(mfccs.numpy())}"
        )

    # Reshape to [spectrogram_length, dct_coefficient_count, 1]
    processed_features = tf.reshape(
        mfccs,
        [
            model_settings["spectrogram_length"],
            model_settings["dct_coefficient_count"],
            1,
        ],
    )

    # Add batch dimension
    features = tf.expand_dims(processed_features, axis=0)

    if debug:
        print(f"[PREPROCESSING] Final features shape: {features.shape}")
    return features
