#!/usr/bin/env python
"""
Representative Dataset-Aware Quantization Analysis for KWS Models.

This script analyzes multiple representative audio samples from the speech_commands dataset
to determine optimal int16 quantization parameters. Uses 120 calibration samples specified
in quant_cal_idxs.txt for more robust quantization parameter estimation.

Based on kws_quantize_int16.py but enhanced with multi-sample dataset analysis.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add python directory to path to import updl modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import common KWS utilities
from kws_common import (
    convert_wav_to_mfcc_features_tensor,
    create_model_layer_output,
    extract_activation_function,
    split_dense_softmax_layers,
    calculate_symmetric_quantization_params,
    parse_command
)

np.set_printoptions(precision=6, suppress=True)


def load_calibration_indices(indices_file):
    """Load calibration sample indices from text file
    
    Args:
        indices_file: Path to quant_cal_idxs.txt containing sample indices
        
    Returns:
        list: List of integer indices for calibration samples
    """
    indices = []
    try:
        with open(indices_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    indices.append(int(line))
        print(f"Loaded {len(indices)} calibration indices from {indices_file}")
        return indices
    except Exception as e:
        print(f"Error loading calibration indices: {e}")
        return []


def load_speech_commands_dataset(data_dir):
    """Load speech_commands dataset from TensorFlow Records
    
    Args:
        data_dir: Path to directory containing TFRecord files
        
    Returns:
        tuple: (train_dataset, dataset_info)
    """
    try:
        print(f"Loading speech_commands dataset from: {data_dir}")
        
        # Load only training split for calibration
        ds_train, ds_info = tfds.load(
            'speech_commands', 
            split='train',
            data_dir=data_dir, 
            with_info=True
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {ds_info.splits['train'].num_examples}")
        print(f"Classes: {ds_info.features['label'].names}")
        
        return ds_train, ds_info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def process_dataset_sample(sample, flags):
    """Process a single dataset sample to extract MFCC features
    
    Args:
        sample: Dictionary containing 'audio' and 'label' from dataset
        flags: Configuration flags
        
    Returns:
        tf.Tensor: Processed MFCC features ready for model inference
    """
    # Extract audio from dataset sample
    audio = sample['audio']
    
    # Convert to proper format and extract features
    audio = tf.cast(audio, tf.float32)
    features = convert_wav_to_mfcc_features_tensor(audio, flags)
    
    return features


def analyze_representative_dataset_quantization(model_path, data_dir, calibration_indices_file, flags, output_json, save_split_model=None):
    """Analyze representative dataset samples to determine optimal int16 quantization parameters"""
    
    # Load calibration indices
    calibration_indices = load_calibration_indices(calibration_indices_file)
    if not calibration_indices:
        print("Failed to load calibration indices")
        return None
    
    print(f"Using {len(calibration_indices)} calibration samples for quantization analysis")
    
    # Load speech commands dataset
    ds_train, ds_info = load_speech_commands_dataset(data_dir)
    if ds_train is None:
        print("Failed to load dataset")
        return None
    
    # Load Keras SavedModel
    print(f"\nLoading SavedModel from: {model_path}")
    original_model = tf.keras.models.load_model(model_path)
    
    print("Original model loaded successfully!")
    print(f"Input shape: {original_model.input_shape}")
    print(f"Output shape: {original_model.output_shape}")
    
    # Split Dense+Softmax layers for better quantization handling
    model = split_dense_softmax_layers(original_model)
    
    print(f"Modified model input shape: {model.input_shape}")
    print(f"Modified model output shape: {model.output_shape}")
    
    # Save split model if requested
    if save_split_model:
        print(f"\nSaving split model to: {save_split_model}")
        model.save(save_split_model)
        print("Split model saved successfully!")
    
    # Create multi-output model for layer logging
    multi_output_model, layer_names, layer_types = create_model_layer_output(model)
    
    print(f"\nAnalyzing {len(layer_names)} layers for quantization parameters...")
    
    # Convert dataset to list and select calibration samples
    print("Converting dataset and selecting calibration samples...")
    ds_list = list(ds_train.take(max(calibration_indices) + 500))  # Take enough samples to cover all indices with buffer
    
    # Select only the calibration samples
    calibration_samples = []
    for idx in calibration_indices:
        if idx < len(ds_list):
            calibration_samples.append(ds_list[idx])
        else:
            print(f"Warning: Index {idx} is out of range, skipping")
    
    print(f"Selected {len(calibration_samples)} valid calibration samples")
    
    # Initialize accumulator for layer statistics
    layer_stats = {}
    input_mins, input_maxs = [], []
    
    print(f"\nProcessing {len(calibration_samples)} calibration samples...")
    
    # Process samples in batches to manage memory
    batch_size = 10  # Process 10 samples at a time
    num_batches = (len(calibration_samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(calibration_samples))
        batch_samples = calibration_samples[start_idx:end_idx]
        
        # Process batch samples
        batch_features = []
        for sample in batch_samples:
            try:
                features = process_dataset_sample(sample, flags)
                batch_features.append(features)
                
                # Collect input statistics
                input_flat = features.numpy().flatten()
                input_mins.append(np.min(input_flat))
                input_maxs.append(np.max(input_flat))
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if not batch_features:
            continue
            
        # Stack batch features and run inference
        try:
            batch_tensor = tf.concat(batch_features, axis=0)
            batch_layer_outputs = multi_output_model.predict(batch_tensor, verbose=0)
            
            # Accumulate layer statistics for each sample in the batch
            for sample_idx in range(batch_tensor.shape[0]):  # For each sample in batch
                for layer_idx, (layer_name, layer_output) in enumerate(zip(layer_names, batch_layer_outputs)):
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {
                            'mins': [],
                            'maxs': [],
                            'layer_idx': layer_idx,
                            'layer_type': layer_types[layer_idx]
                        }
                    
                    # Collect min/max for this individual sample
                    sample_output = layer_output[sample_idx]  # Extract single sample output
                    output_flat = sample_output.flatten()
                    layer_stats[layer_name]['mins'].append(np.min(output_flat))
                    layer_stats[layer_name]['maxs'].append(np.max(output_flat))
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    # Calculate global input quantization parameters
    global_input_min = np.min(input_mins)
    global_input_max = np.max(input_maxs)
    input_scale, input_zp = calculate_symmetric_quantization_params(global_input_min, global_input_max)
    
    print(f"\n=== Representative Dataset Input Quantization Analysis ===")
    print(f"Samples processed: {len(input_mins)}")
    print(f"Global input range: [{global_input_min:.6f}, {global_input_max:.6f}]")
    print(f"Input scale: {input_scale:.8f}, zero_point: {input_zp}")
    
    # Initialize quantization parameters structure
    quantization_params = {
        "metadata": {
            "method": "representative_dataset",
            "calibration_samples": len(input_mins),
            "calibration_indices_file": str(calibration_indices_file),
            "data_directory": str(data_dir)
        },
        "input": {
            "scale": float(input_scale),
            "zero_point": int(input_zp),
            "min_val": float(global_input_min),
            "max_val": float(global_input_max)
        },
        "layers": {}
    }
    
    print(f"\n=== Representative Dataset Layer-by-Layer Quantization Analysis ===")
    
    # Calculate global layer quantization parameters
    for layer_name, stats in layer_stats.items():
        if not stats['mins'] or not stats['maxs']:
            continue
            
        # Get layer object to extract activation function
        layer = model.get_layer(layer_name)
        activation = extract_activation_function(layer)
        
        # Calculate global min/max across all samples
        global_min = np.min(stats['mins'])
        global_max = np.max(stats['maxs'])
        
        # Calculate symmetric quantization parameters
        scale, zero_point = calculate_symmetric_quantization_params(global_min, global_max)
        
        # Store parameters
        quantization_params["layers"][layer_name] = {
            "layer_index": stats['layer_idx'],
            "layer_type": stats['layer_type'],
            "activation": activation,
            "scale": float(scale),
            "zero_point": int(zero_point),
            "min_val": float(global_min),
            "max_val": float(global_max),
            "range": float(global_max - global_min),
        }
        
        print(f"Layer {stats['layer_idx']}: {layer_name} ({stats['layer_type']}, {activation})")
        print(f"  Global range: [{global_min:.6f}, {global_max:.6f}] (from {len(stats['mins'])} samples)")
        print(f"  Scale: {scale:.8f}, Zero-point: {zero_point}")
    
    # Save to JSON file
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(quantization_params, f, indent=2)
    
    print(f"\n=== Representative Dataset Quantization Parameters Saved ===")
    print(f"Output file: {output_path}")
    print(f"Total layers analyzed: {len(quantization_params['layers'])}")
    print(f"Calibration samples processed: {len(input_mins)}")
    
    return quantization_params

def main():
    # Get arguments from kws_common
    flags, _ = parse_command()
    
    # Add our own arguments
    parser = argparse.ArgumentParser(description='Analyze representative dataset samples for optimal int16 quantization parameters')
    parser.add_argument('--data_dir', 
                       default='/home/kaiyin-upbeat/data',
                       help='Path to TensorFlow dataset directory')
    parser.add_argument('--calibration_indices',
                       default='./kws_quantize_calibrate_idxs.txt',
                       help='Path to calibration sample indices file')
    parser.add_argument('--model_path',
                       default='trained_models/kws_ref_model',
                       help='Path to Keras SavedModel directory')
    parser.add_argument('--output_json',
                       default='artifacts/kws_quantize_int16.json',
                       help='Output JSON file for quantization parameters')
    parser.add_argument('--save_split_model',
                       default=None,
                       help='Path to save the model with split Dense+Softmax layers')
    
    # Parse our arguments
    test_args = parser.parse_args()
    
    print(f"=== Representative Dataset Quantization Analysis ===")
    print(f"Dataset directory: {test_args.data_dir}")
    print(f"Calibration indices: {test_args.calibration_indices}")
    print(f"Model path: {test_args.model_path}")
    print(f"Output JSON: {test_args.output_json}")
    
    # Analyze quantization parameters using representative dataset
    params = analyze_representative_dataset_quantization(
        test_args.model_path,
        test_args.data_dir,
        test_args.calibration_indices,
        flags, 
        test_args.output_json,
        test_args.save_split_model
    )
    
    if params is None:
        print("Analysis failed")
        return 1
    
    print("\n=== Representative Dataset Analysis Complete ===")
    return 0

if __name__ == '__main__':
    sys.exit(main())