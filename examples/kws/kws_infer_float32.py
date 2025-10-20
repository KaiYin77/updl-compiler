#!/usr/bin/env python
"""
Inference script for KWS model using float32 TFLite model with layer-by-layer output logging.
"""

import tensorflow as tf
import numpy as np
import argparse
# Import common KWS utilities
from kws_common import (
    convert_wav_to_tensor,
    create_model_layer_output,
    convert_wav_to_mfcc_features_tensor,
    parse_command
)
from kws_common.preprocessor import WORD_LABELS

np.set_printoptions(precision=3, suppress=True)


def infer_with_layer_logging(model_path, wav_file, flags):
    """Run inference with layer-by-layer output logging using Keras SavedModel"""

    # Load audio and extract features
    wav = convert_wav_to_tensor(wav_file, flags)
    if wav is None:
        print("Failed to load audio file")
        return None

    features = convert_wav_to_mfcc_features_tensor(wav, flags, debug=True)

    # Load Keras SavedModel
    print(f"\nLoading SavedModel from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Print first conv2d layer weights (10x4 first filter) as float32 and fuse with BatchNorm
    print("\n=== Conv2D + BatchNorm Fusion Analysis ===")

    # Store fused weights for later use in handcrafted convolution
    global_fused_kernel = None
    global_fused_bias = None

    # Store fused depthwise weights for later use in handcrafted depthwise convolution
    global_fused_dw_kernel = None
    global_fused_dw_bias = None

    # Store Dense layer weights for handcrafted implementation
    global_dense_weight_oi = None  # Weight in (O, I) format
    global_dense_bias = None
    global_dense_layer_found = False

    for i, layer in enumerate(model.layers):
        if "dense" in layer.name.lower():
            weights = layer.get_weights()
            if len(weights) > 0:
                weight = weights[0]  # Original (I, O) format
                bias = weights[1]
                oi_tensor = tf.transpose(
                    weight, perm=[1, 0]
                )  # (O, I) format for C implementation

                # Store for handcrafted implementation
                global_dense_weight_oi = oi_tensor.numpy()
                global_dense_bias = bias
                global_dense_layer_found = True

                print(f"Dense layer found: {layer.name}")
                print("Original weight.shape (I,O): ", weight.shape)
                print("Transposed weight.shape (O,I): ", oi_tensor.shape)
                print("bias.shape: ", bias.shape)
                print("First 2 output neurons weights: ")
                print(oi_tensor[:2, :8].numpy())  # First 2 outputs, first 8 inputs
                print("bias: ", bias)
                break

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "kernel") and "conv" in layer.name.lower():
            conv_weights = layer.get_weights()
            if len(conv_weights) > 0:
                conv_kernel = conv_weights[0]  # Get kernel weights
                conv_bias = (
                    conv_weights[1]
                    if len(conv_weights) > 1
                    else np.zeros(conv_kernel.shape[-1])
                )

                print(f"Layer {i}: {layer.name}")
                print(f"  Conv kernel shape: {conv_kernel.shape}")

                # Print original first filter (10x4) - HWIO format: [height, width, input_ch, output_ch]
                if len(conv_kernel.shape) == 4:
                    h, w, in_ch, out_ch = conv_kernel.shape
                    if h >= 10 and w >= 4:
                        first_filter_orig = conv_kernel[
                            :10, :4, 0, 0
                        ]  # First 10x4 of first filter
                        print(f"  ORIGINAL CONV2D FIRST FILTER WEIGHTS (10x4): [")
                        for ky in range(10):
                            print(
                                f"    [{', '.join(f'{first_filter_orig[ky, kx]:.6f}' for kx in range(4))}]{',' if ky < 9 else ''}"
                            )
                        print(f"  ]")

                # Look for following BatchNormalization layer
                bn_layer = None
                if i + 1 < len(model.layers):
                    next_layer = model.layers[i + 1]
                    if (
                        "batch" in next_layer.name.lower()
                        or next_layer.__class__.__name__ == "BatchNormalization"
                    ):
                        bn_layer = next_layer
                        print(f"  Found BatchNorm layer: {next_layer.name}")

                # Perform fusion if BatchNorm found
                if bn_layer:
                    bn_weights = bn_layer.get_weights()
                    gamma = bn_weights[0]  # Scale
                    beta = bn_weights[1]  # Bias
                    moving_mean = bn_weights[2]
                    moving_variance = bn_weights[3]
                    epsilon = bn_layer.epsilon

                    print(
                        f"  BatchNorm params - gamma shape: {gamma.shape}, epsilon: {epsilon}"
                    )

                    # Compute fusion: W_fused = W * gamma / sqrt(var + eps)
                    inv_std = 1.0 / np.sqrt(moving_variance + epsilon)
                    scale_factor = gamma * inv_std

                    # Fuse weights: multiply each output channel by its scale factor
                    # Conv2D weights shape: (kernel_h, kernel_w, input_channels, output_channels)
                    fused_kernel = conv_kernel * scale_factor.reshape(1, 1, 1, -1)

                    # Fuse bias: b_fused = gamma * (b_conv - mean) / sqrt(var + eps) + beta
                    fused_bias = gamma * (conv_bias - moving_mean) * inv_std + beta

                    # Store for later use
                    global_fused_kernel = fused_kernel
                    global_fused_bias = fused_bias

                    # Print fused first filter (10x4)
                    if len(fused_kernel.shape) == 4:
                        h, w, in_ch, out_ch = fused_kernel.shape
                        if h >= 10 and w >= 4:
                            first_filter_fused = fused_kernel[
                                :10, :4, 0, 0
                            ]  # First 10x4 of first filter
                            print(f"  FUSED CONV2D FIRST FILTER WEIGHTS (10x4): [")
                            for ky in range(10):
                                print(
                                    f"    [{', '.join(f'{first_filter_fused[ky, kx]:.6f}' for kx in range(4))}]{',' if ky < 9 else ''}"
                                )
                            print(f"  ]")

                            # Show scale factors for first few output channels
                            print(
                                f"  Scale factors (first 8 channels): [{', '.join(f'{scale_factor[ch]:.6f}' for ch in range(min(8, len(scale_factor))))}]"
                            )
                            print(
                                f"  Original bias (first 8): [{', '.join(f'{conv_bias[ch]:.6f}' for ch in range(min(8, len(conv_bias))))}]"
                            )
                            print(
                                f"  Fused bias (first 8): [{', '.join(f'{fused_bias[ch]:.6f}' for ch in range(min(8, len(fused_bias))))}]"
                            )

                break  # Only process first conv2d layer

    # === Depthwise2D + BatchNorm Fusion Analysis ===
    print("\n=== Depthwise2D + BatchNorm Fusion Analysis ===")

    for i, layer in enumerate(model.layers):
        # DepthwiseConv2D uses 'depthwise_kernel' instead of 'kernel'
        is_depthwise = (
            hasattr(layer, "kernel") or hasattr(layer, "depthwise_kernel")
        ) and (
            "depthwise" in layer.name.lower()
            or "DepthwiseConv2D" in str(type(layer))
            or "depthwise_conv2d" in str(type(layer))
            or layer.__class__.__name__ == "DepthwiseConv2D"
            or hasattr(layer, "depth_multiplier")
        )

        if is_depthwise:
            print(
                f">>> FOUND DEPTHWISE LAYER: {layer.name} - {layer.__class__.__name__}"
            )
            dw_weights = layer.get_weights()
            if len(dw_weights) > 0:
                dw_kernel = dw_weights[0]  # Get depthwise kernel weights
                # DepthwiseConv2D kernel shape: (kernel_h, kernel_w, input_channels, depth_multiplier)
                # For bias, we need number of output channels = input_channels * depth_multiplier
                num_output_channels = (
                    dw_kernel.shape[2] * dw_kernel.shape[3]
                    if len(dw_kernel.shape) == 4
                    else dw_kernel.shape[-1]
                )
                dw_bias = (
                    dw_weights[1]
                    if len(dw_weights) > 1
                    else np.zeros(num_output_channels)
                )

                print(f"Layer {i}: {layer.name}")
                print(f"  Depthwise kernel shape: {dw_kernel.shape}")

                # Print original first filter (kernel_h x kernel_w) - HWIO format: [height, width, input_ch, depth_multiplier]
                if len(dw_kernel.shape) == 4:
                    h, w, in_ch, depth_mult = dw_kernel.shape
                    # Adapt to actual kernel size
                    display_h = min(h, 5)  # Show up to 5x5
                    display_w = min(w, 5)
                    first_filter_orig = dw_kernel[
                        :display_h, :display_w, 0, 0
                    ]  # First filter of first channel
                    print(
                        f"  ORIGINAL DEPTHWISE2D FIRST FILTER WEIGHTS ({display_h}x{display_w}): ["
                    )
                    for ky in range(display_h):
                        print(
                            f"    [{', '.join(f'{first_filter_orig[ky, kx]:.6f}' for kx in range(display_w))}]{',' if ky < display_h-1 else ''}"
                        )
                    print(f"  ]")

                # Look for following BatchNormalization layer
                bn_layer = None
                if i + 1 < len(model.layers):
                    next_layer = model.layers[i + 1]
                    if (
                        "batch" in next_layer.name.lower()
                        or next_layer.__class__.__name__ == "BatchNormalization"
                    ):
                        bn_layer = next_layer
                        print(f"  Found BatchNorm layer: {next_layer.name}")

                # Perform fusion if BatchNorm found
                if bn_layer:
                    bn_weights = bn_layer.get_weights()
                    gamma = bn_weights[0]  # Scale
                    beta = bn_weights[1]  # Bias
                    moving_mean = bn_weights[2]
                    moving_variance = bn_weights[3]
                    epsilon = bn_layer.epsilon

                    print(
                        f"  BatchNorm params - gamma shape: {gamma.shape}, epsilon: {epsilon}"
                    )

                    # Compute fusion: W_fused = W * gamma / sqrt(var + eps)
                    inv_std = 1.0 / np.sqrt(moving_variance + epsilon)
                    scale_factor = gamma * inv_std

                    # Fuse weights: multiply each output channel by its scale factor
                    # DepthwiseConv2D weights shape: (kernel_h, kernel_w, input_channels, depth_multiplier)
                    # For depthwise, each input channel has its own set of filters
                    fused_dw_kernel = dw_kernel * scale_factor.reshape(1, 1, -1, 1)

                    # Fuse bias: b_fused = gamma * (b_dw - mean) / sqrt(var + eps) + beta
                    fused_dw_bias = gamma * (dw_bias - moving_mean) * inv_std + beta

                    # Store for later use
                    global_fused_dw_kernel = fused_dw_kernel
                    global_fused_dw_bias = fused_dw_bias

                    # Print fused first filter (matching original size)
                    if len(fused_dw_kernel.shape) == 4:
                        h, w, in_ch, depth_mult = fused_dw_kernel.shape
                        # Use same display size as original
                        display_h = min(h, 5)
                        display_w = min(w, 5)
                        first_filter_fused = fused_dw_kernel[
                            :display_h, :display_w, 0, 0
                        ]  # First filter of first channel
                        print(
                            f"  FUSED DEPTHWISE2D FIRST FILTER WEIGHTS ({display_h}x{display_w}): ["
                        )
                        for ky in range(display_h):
                            print(
                                f"    [{', '.join(f'{first_filter_fused[ky, kx]:.6f}' for kx in range(display_w))}]{',' if ky < display_h-1 else ''}"
                            )
                        print(f"  ]")

                        # Show scale factors for first few channels
                        print(
                            f"  Scale factors (first 8 channels): [{', '.join(f'{scale_factor[ch]:.6f}' for ch in range(min(8, len(scale_factor))))}]"
                        )
                        print(
                            f"  Original bias (first 8): [{', '.join(f'{dw_bias[ch]:.6f}' for ch in range(min(8, len(dw_bias))))}]"
                        )
                        print(
                            f"  Fused bias (first 8): [{', '.join(f'{fused_dw_bias[ch]:.6f}' for ch in range(min(8, len(fused_dw_bias))))}]"
                        )

                break  # Only process first depthwise2d layer
    # Create multi-output model for layer logging
    multi_output_model, layer_names, layer_types = create_model_layer_output(model)

    print(f"\nCreated multi-output model with {len(layer_names)} intermediate layers")

    # Run inference and get all layer outputs
    features_np = features.numpy().astype(np.float32)

    # Print input tensor first 5 values
    print("\n=== Input Tensor ===")
    print(f"Input Shape: {features_np.shape}")
    print(f" FLOAT32: {features_np[0, :, :, 0]}")
    print()

    print("=== Layer-by-Layer Outputs (Keras SavedModel) ===")

    # Get all layer outputs
    layer_outputs = multi_output_model.predict(features_np, verbose=0)

    # Log each layer's output
    for i, (layer_name, layer_output) in enumerate(zip(layer_names, layer_outputs)):
        if layer_name == "average_pooling2d":
            N, H, W, C = layer_output.shape

            # Take first batch and transpose to CHW
            hwc_tensor = layer_output[0]  # (H, W, C)
            chw_tensor = tf.transpose(hwc_tensor, perm=[2, 0, 1])  # (C, H, W)

            print(f"Layer {i}: {layer_name}")
            print(f"- shape: NHWC({N}, {H}, {W}, {C}) -> CHW({C}, {H}, {W})")

            print("- chw_tensor.shape: ", chw_tensor.shape)
            print("- chw_tensor: ", chw_tensor.numpy().flatten())

        elif len(layer_output.shape) == 4:
            N, H, W, C = layer_output.shape

            # Take first batch and transpose to CHW
            hwc_tensor = layer_output[0]  # (H, W, C)
            chw_tensor = tf.transpose(hwc_tensor, perm=[2, 0, 1])  # (C, H, W)

            print(f"Layer {i}: {layer_name}")
            print(f"- shape: NHWC({N}, {H}, {W}, {C}) -> CHW({C}, {H}, {W})")

            print("- chw_tensor[0].shape: ", chw_tensor[0].shape)
            print("- chw_tensor[0:2]: ", chw_tensor[0:2].numpy())

        else:
            # For non-4D outputs, just flatten and log
            flat_output = layer_output.flatten()
            print(f"Layer {i}: {layer_name}")
            print(f"- shape: {layer_output.shape}")
            print(f"- flat_output[:10]: {flat_output[:10]}")

            # Add handcrafted Dense layer implementation after Flatten layer
            # if 'flatten' in layer_name.lower() and global_dense_layer_found:
            #     print(f"\n=== HANDCRAFTED DENSE LAYER TEST ===")
            #     print(f"Using flatten output as Dense input")

            #     flatten_output = flat_output  # This is the input to Dense layer
            #     dense_weight = global_dense_weight_oi  # Shape: (output_features, input_features)
            #     dense_bias = global_dense_bias

            #     print(f"Dense input shape: {flatten_output.shape}")
            #     print(f"Dense weight shape (O,I): {dense_weight.shape}")
            #     print(f"Dense bias shape: {dense_bias.shape}")
            #     print(f"First 8 input values: {flatten_output[:8]}")

            #     # Manual dense layer computation with detailed logging
            #     output_features, input_features = dense_weight.shape
            #     dense_output = np.zeros(output_features)

            #     print(f"\n--- Computing Dense Layer Output ---")
            #     for out_idx in range(min(2, output_features)):  # Show first 3 outputs
            #         dot_product = 0.0
            #         weight_row = dense_weight[out_idx, :]  # Get weight row for this output

            #         print(f"\nOutput[{out_idx}]:")
            #         print(f"  Weight row shape: {weight_row.shape}")
            #         print(f"  weights: {weight_row[:64]}")

            #         # Show detailed dot product computation for first few elements
            #         for in_idx in range(min(64, input_features)):
            #             input_val = flatten_output[in_idx]
            #             weight_val = weight_row[in_idx]
            #             product = input_val * weight_val
            #             dot_product += product
            #             print(f"    input[{in_idx}]={input_val:.6f} * weight[{out_idx},{in_idx}]={weight_val:.6f} = {product:.6f}")

            #         # Complete the dot product for remaining elements (without detailed logging)
            #         for in_idx in range(64, input_features):
            #             dot_product += flatten_output[in_idx] * weight_row[in_idx]

            #         # Add bias
            #         bias_val = dense_bias[out_idx]
            #         final_output = dot_product + bias_val
            #         dense_output[out_idx] = final_output

            #         print(f"  Dot product sum: {dot_product:.6f}")
            #         print(f"  Bias[{out_idx}]: {bias_val:.6f}")
            #         print(f"  Final output[{out_idx}]: {final_output:.6f}")

            #     # Complete computation for remaining outputs
            #     for out_idx in range(1, output_features):
            #         dot_product = np.dot(flatten_output, dense_weight[out_idx, :])
            #         dense_output[out_idx] = dot_product + dense_bias[out_idx]

            #     print(f"\n--- Handcrafted Dense Results ---")
            #     print(f"All outputs: {dense_output}")
            #     print(f"=== END HANDCRAFTED DENSE LAYER ===\n")
            # exit()
        # if i == 3:
        #     print("\n=== HANDCRAFTED CONVOLUTION TEST ===")
        #     # Use the already computed fused weights from lines 232 and 235
        #     if global_fused_kernel is not None and global_fused_bias is not None:
        #         # Convert input from NHWC to CHW for our convolution
        #         input_nhwc = features_np[0]  # Remove batch dimension: (H, W, C)
        #         input_chw = np.transpose(input_nhwc, (2, 0, 1))  # (C, H, W)
        #         print(f"- Input shape: {input_chw.shape}")

        #         fused_kernel = global_fused_kernel
        #         fused_bias = global_fused_bias
        #         fused_kernel_oihw = np.transpose(fused_kernel, (3, 2, 0, 1))  # HWIO -> OIHW

        #         print(f"- Fused kernel shape (OIHW): {fused_kernel_oihw.shape}")
        #         print(f"- Fused bias shape: {fused_bias.shape}")

        #         # Manual convolution implementation with OIHW weights
        #         def manual_conv2d_same_padding(input_chw, kernel_oihw, bias, stride_h=1, stride_w=1):
        #             """Manual Conv2D with SAME padding following TensorFlow, using OIHW weight layout"""
        #             C_in, H_in, W_in = input_chw.shape
        #             C_out, C_in_k, K_h, K_w = kernel_oihw.shape

        #             # Calculate output dimensions
        #             H_out = (H_in + stride_h - 1) // stride_h
        #             W_out = (W_in + stride_w - 1) // stride_w

        #             # Calculate padding (TensorFlow SAME padding)
        #             def calc_padding(input_size, kernel_size, stride):
        #                 if input_size % stride == 0:
        #                     pad_total = max(kernel_size - stride, 0)
        #                 else:
        #                     pad_total = max(kernel_size - (input_size % stride), 0)
        #                 pad_before = pad_total // 2
        #                 pad_after = pad_total - pad_before
        #                 return pad_before, pad_after

        #             pad_top, pad_bottom = calc_padding(H_in, K_h, stride_h)
        #             pad_left, pad_right = calc_padding(W_in, K_w, stride_w)

        #             print(f"- Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")

        #             # Initialize output
        #             output_chw = np.zeros((C_out, H_out, W_out), dtype=np.float32)

        #             # Convolution
        #             debug_counter = 0
        #             debug_counter2 = 0
        #             for out_y in range(H_out):
        #                 for out_x in range(W_out):
        #                     for out_c in range(C_out):
        #                         sum_val = 0.0

        #                         for in_c in range(C_in):
        #                             for ky in range(K_h):
        #                                 for kx in range(K_w):
        #                                     # Calculate input position
        #                                     in_y = out_y * stride_h + ky - pad_top
        #                                     in_x = out_x * stride_w + kx - pad_left

        #                                     # Apply padding (zero padding)
        #                                     if 0 <= in_y < H_in and 0 <= in_x < W_in:
        #                                         input_val = input_chw[in_c, in_y, in_x]
        #                                     else:
        #                                         input_val = 0.0

        #                                     # Get weight (OIHW format)
        #                                     weight_val = kernel_oihw[out_c, in_c, ky, kx]
        #                                     # if debug_counter < 10:
        #                                     #     print(f"[{debug_counter}] input_val: {input_val}")
        #                                     #     print(f"[{debug_counter}] weight_val: {weight_val}")
        #                                     #     debug_counter += 1
        #                                     sum_val += input_val * weight_val

        #                         # Add bias
        #                         # if debug_counter2 < 1:
        #                         #     print(f"[{debug_counter2}] output_chw[{out_c}, {out_y}, {out_x}]")
        #                         #     print(f"[{debug_counter2}] sum_val: {sum_val}")
        #                         #     print(f"[{debug_counter2}] bias[{out_c}]: {bias[out_c]}")
        #                         #     debug_counter2 += 1
        #                         # output_chw[out_c, out_y, out_x] = sum_val
        #                         if debug_counter < 10:
        #                             print(f"[{debug_counter}] bias[{out_c}]: { bias[out_c]}")
        #                             debug_counter +=1
        #                         output_chw[out_c, out_y, out_x] = sum_val + bias[out_c]

        #             return output_chw

        #         # Run manual convolution with OIHW kernel and stride (2, 2)
        #         manual_output_chw = manual_conv2d_same_padding(input_chw, fused_kernel_oihw, fused_bias, stride_h=2, stride_w=2)
        #         print(f"- Manual output CHW shape: {manual_output_chw.shape}")
        #         # Compare first channel
        #         print(f"- Manual convolution output (first channel, first 5x5):")
        #         print(manual_output_chw[0, :, :])

        #     print("=== END HANDCRAFTED CONVOLUTION TEST ===\n")

    print("=== End Layer Outputs ===\n")

    # Get final prediction from original model
    final_output = model.predict(features_np, verbose=0)

    # Get prediction
    prediction = final_output[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Add manual softmax reference calculation
    print("\n=== Manual Softmax Reference ===")
    # Find the Dense layer output (before softmax)
    dense_output = None
    for i, (layer_name, layer_output) in enumerate(zip(layer_names, layer_outputs)):
        if "dense" in layer_name.lower() and "softmax" not in layer_name.lower():
            dense_output = layer_output.flatten()
            print(f"Dense layer output (before softmax): {dense_output}")
            break

    if dense_output is not None:
        # Manual softmax calculation
        def manual_softmax(x):
            x_shifted = x - np.max(x)  # Numerical stability
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x)

        manual_softmax_result = manual_softmax(dense_output)
        print(f"Manual softmax result: {manual_softmax_result}")

    return predicted_class, WORD_LABELS[predicted_class], confidence, prediction


def main():
    # Get arguments from kws_common
    flags, _ = parse_command()

    # Add our own arguments
    parser = argparse.ArgumentParser(
        description="KWS Float32 Inference with Layer Logging"
    )
    parser.add_argument(
        "--wav_file", default="bcm-wavs/go.wav", help="Path to WAV file to test"
    )
    parser.add_argument(
        "--model_path",
        default="trained_models/kws_ref_model_split",
        help="Path to Keras SavedModel directory",
    )

    # Parse our arguments
    test_args = parser.parse_args()

    print(f"Loading float32 model: {test_args.model_path}")
    print(f"Testing audio file: {test_args.wav_file}")

    # Run inference with layer logging
    result = infer_with_layer_logging(test_args.model_path, test_args.wav_file, flags)

    if result is None:
        print("Inference failed")
        return

    # Display results
    class_id, label, confidence, all_scores = result

    print("\n=== Final Prediction Results ===")
    print(f"Predicted Class ID: {class_id}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.6f}")

    print("\nAll Class Probabilities:")
    for i, score in enumerate(all_scores):
        print(f"  {WORD_LABELS[i]}: {score:.6f}")


if __name__ == "__main__":
    main()
