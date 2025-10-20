#!/usr/bin/env python
"""
Inference script for quantized KWS model using INT8 TFLite model.
Combines patterns from eval_quantized_model.py, quantize.py, and infer.py.
"""

import tensorflow as tf
import numpy as np
import argparse
# Import common KWS utilities
from kws_common import convert_wav_to_tensor, convert_wav_to_mfcc_features_tensor, parse_command
from kws_common.preprocessor import WORD_LABELS


def print_layer_outputs(interpreter):
    """Print layer outputs in correct execution sequence with dequantized float values"""

    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Filter to get actual layer outputs (skip internal/intermediate tensors)
    # We'll identify layer outputs by filtering out tensors that are clearly intermediate
    layer_outputs = []

    for tensor in tensor_details:
        name = tensor["name"]
        # Skip input tensors
        if any(tensor["index"] == inp["index"] for inp in input_details):
            continue

        # Include tensors that represent actual layer outputs
        # These typically have names ending with layer operations or are final outputs
        if (
            any(
                layer_op in name.lower()
                for layer_op in [
                    "conv",
                    "dense",
                    "pool",
                    "flatten",
                    "batch_norm",
                    "relu",
                    "sigmoid",
                    "tanh",
                    "softmax",
                ]
            )
            or any(tensor["index"] == out["index"] for out in output_details)
            or "sequential" in name.lower()
            or "model" in name.lower()
        ):
            layer_outputs.append(tensor)

    # Sort by tensor index to get execution order
    layer_outputs.sort(key=lambda x: x["index"])

    print(f"Found {len(layer_outputs)} layer outputs in execution sequence:")

    for layer_idx, tensor in enumerate(layer_outputs):
        try:
            # Get tensor data
            tensor_data = interpreter.get_tensor(tensor["index"])

            if tensor_data is not None:
                # Parse layer type from tensor name
                name = tensor["name"]
                layer_type = "Unknown"
                if "conv" in name.lower():
                    if "depthwise" in name.lower():
                        layer_type = "DepthwiseConv"
                    else:
                        layer_type = "Conv"
                elif "dense" in name.lower():
                    layer_type = "Dense"
                elif "pool" in name.lower():
                    if "max" in name.lower():
                        layer_type = "MaxPooling"
                    else:
                        layer_type = "AvgPooling"
                elif "flatten" in name.lower():
                    layer_type = "Flatten"
                elif "batch_norm" in name.lower():
                    layer_type = "BatchNorm"
                elif any(
                    act in name.lower()
                    for act in ["relu", "sigmoid", "tanh", "softmax"]
                ):
                    layer_type = "Activation"
                elif tensor["index"] in [out["index"] for out in output_details]:
                    layer_type = "Output"

                print(f"\n[LAYER {layer_idx:02d}] {layer_type}")

                # Get flat data for analysis
                flat_data = tensor_data.flatten()

                # Handle quantization and print dequantized values
                if "quantization" in tensor and len(tensor["quantization"]) >= 2:
                    scale, zero_point = tensor["quantization"]
                    if scale != 0.0:  # Valid quantization parameters

                        # Also show the raw int8 values for reference
                        first_raw = (
                            flat_data[:10] if len(flat_data) >= 10 else flat_data
                        )
                        print(f"  INT8: {first_raw}")
                        print(
                            f"  Quantization: scale={scale:.8f}, zero_point={zero_point}"
                        )

                        # Dequantize all values
                        dequant_data = scale * (
                            flat_data.astype(np.float32) - zero_point
                        )

                        # Print first 10 dequantized values for comparison
                        first_dequant = (
                            dequant_data[:10]
                            if len(dequant_data) >= 10
                            else dequant_data
                        )
                        print(
                            f"  FLOAT32: [{', '.join([f'{v:.6f}' for v in first_dequant])}]"
                        )

        except Exception as e:
            print(f"  [ERROR] Reading tensor {layer_idx} ({tensor['name']}): {e}")


def infer_quantized_model(tflite_model_path, wav_file, flags):
    """Run inference with quantized TFLite model"""

    # Load audio and extract features
    wav = convert_wav_to_tensor(wav_file, flags)
    if wav is None:
        print("Failed to load audio file")
        return None

    features = convert_wav_to_mfcc_features_tensor(wav, flags, debug=True)

    # Load TFLite model
    print(f"\nLoading quantized TFLite model from: {tflite_model_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded successfully!")
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]["quantization"]
    print(f"Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")

    # Prepare input data
    features_np = features.numpy().astype(np.float32)

    # Print input tensor info
    print("\n=== Input Tensor ===")
    flat_input = features_np.flatten()
    print(f"Input Shape: {features_np.shape}")
    print(f" FLOAT32: {flat_input[:10]}")

    # Handle quantization based on input type
    if input_details[0]["dtype"] == np.float32:
        interpreter.set_tensor(input_details[0]["index"], features_np)
    elif input_details[0]["dtype"] == np.int8:
        # Quantize the input (matching eval_quantized_model.py)
        dat_q = np.array(features_np / input_scale + input_zero_point, dtype=np.int8)
        print(f" INT8: {dat_q.flatten()[:10]}")
        interpreter.set_tensor(input_details[0]["index"], dat_q)
    else:
        raise ValueError(
            f"TFLite file has input dtype {input_details[0]['dtype']}. Only np.int8 and np.float32 are supported"
        )

    print("\n=== Running Inference ===")
    # Run inference
    interpreter.invoke()

    # Print layer-by-layer outputs
    print_layer_outputs(interpreter)

    # Get output
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Get prediction
    prediction = output_data[0]
    predicted_class = np.argmax(prediction)

    # For quantized outputs, we might need to dequantize
    if output_details[0]["dtype"] == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
        # Dequantize output
        prediction_dequant = output_scale * (
            prediction.astype(np.float32) - output_zero_point
        )
        confidence = prediction_dequant[predicted_class]
        all_scores = prediction_dequant
    else:
        confidence = prediction[predicted_class]
        all_scores = prediction

    return predicted_class, WORD_LABELS[predicted_class], confidence, all_scores


def main():
    # Get arguments from kws_common
    flags, _ = parse_command()

    # Add our own arguments
    parser = argparse.ArgumentParser(description="KWS Quantized Inference")
    parser.add_argument(
        "--wav_file", default="../wavs/go.wav", help="Path to WAV file to test"
    )
    parser.add_argument(
        "--tflite_model_path",
        default="./trained_models/kws_ref_model.tflite",
        help="Path to quantized TFLite model file",
    )

    # Parse our arguments
    test_args = parser.parse_args()

    print(f"Loading quantized TFLite model: {test_args.tflite_model_path}")
    print(f"Testing audio file: {test_args.wav_file}")

    # Run inference
    result = infer_quantized_model(
        test_args.tflite_model_path, test_args.wav_file, flags
    )

    if result is None:
        print("Inference failed")
        return

    # Display results
    class_id, label, confidence, all_scores = result

    print(f"Predicted Class ID: {class_id}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.6f}")


if __name__ == "__main__":
    main()
