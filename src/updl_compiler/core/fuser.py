#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import numpy as np
from .logger import log_debug, log_info, log_warn
from .config import is_layer_fuseable
from .quantizer import calculate_weight_params, calculate_bias_params


def get_fusion_groups(fusion_json_file):
    """Load fusion groups from JSON file and return fusion mapping"""
    try:
        with open(fusion_json_file, "r") as f:
            fusion_data = json.load(f)

        fusion_groups = fusion_data.get("layers", {})
        log_info(f"Loaded {len(fusion_groups)} fusion groups from {fusion_json_file}")
        return fusion_groups

    except FileNotFoundError:
        log_debug(
            f"Fusion file {fusion_json_file} not found, proceeding without fusion"
        )
        return {}
    except Exception as e:
        log_warn(f"Layer Fuser: Cannot load fusion configuration from {fusion_json_file} - {e}. Proceeding without layer fusion optimization.")
        return {}


def apply_fusion_to_model(model, fusion_groups):
    """Apply fusion based on JSON groups to Keras model

    Returns:
        fused_weights: dict mapping layer_idx -> {weights, bias, activation}
        skip_layers: set of layer indices to skip during serialization
    """
    fused_weights = {}
    skip_layers = set()

    if not fusion_groups:
        log_debug("No fusion groups provided, returning empty fusion data")
        return fused_weights, skip_layers

    # Process each fusion group
    for group_name, group_layers in fusion_groups.items():
        log_debug(f"Processing fusion group: {group_name}")

        # Find the layers in this group
        primary_layer = None
        bn_layer = None
        activation_layer = None
        dropout_layer = None
        primary_idx = None

        # Match by layer type and position
        for layer_name, layer_info in group_layers.items():
            layer_type = layer_info.get("layer_type", "")
            layer_index = layer_info.get("layer_index", -1)

            # Find the model layer at this index
            if 0 <= layer_index < len(model.layers):
                model_layer = model.layers[layer_index]
                expected_type = model_layer.__class__.__name__

                if expected_type == layer_type:
                    if layer_type in ["Conv2D", "DepthwiseConv2D", "Dense"]:
                        primary_layer = model_layer
                        primary_idx = layer_index
                    elif layer_type == "BatchNormalization":
                        bn_layer = model_layer
                        skip_layers.add(layer_index)
                    elif layer_type == "Activation":
                        activation_layer = model_layer
                        skip_layers.add(layer_index)
                    elif layer_type == "Dropout":
                        dropout_layer = model_layer
                        skip_layers.add(layer_index)
                else:
                    log_warn(
                        f"Layer Fuser: Model layer mismatch at index {layer_index} - fusion config expects {layer_type} but model has {expected_type}. Skipping this fusion group."
                    )

        # Perform fusion if we have the right layers
        if primary_layer and bn_layer and primary_idx is not None:
            log_info(
                f"Fusing BatchNorm into {primary_layer.__class__.__name__} layer '{primary_layer.name}' (index {primary_idx})"
            )
            fused_data = _fuse_layer_batchnorm(primary_layer, bn_layer, activation_layer)
            fused_weights[primary_idx] = fused_data
        elif primary_layer and activation_layer and primary_idx is not None:
            # Just activation fusion without BatchNorm
            log_info(
                f"Fusing activation into {primary_layer.__class__.__name__} layer '{primary_layer.name}' (index {primary_idx})"
            )
            activation_name = activation_layer.get_config()["activation"]
            fused_weights[primary_idx] = {"activation": activation_name}

    log_info(
        f"Fusion complete: {len(fused_weights)} layers with fused weights, {len(skip_layers)} layers to skip"
    )
    return fused_weights, skip_layers


def _fuse_layer_batchnorm(layer, bn_layer, activation_layer=None):
    """Fuse BatchNormalization into Conv/Dense layer weights"""
    # Get BatchNorm parameters
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]  # Scale
    beta = bn_weights[1]  # Bias
    moving_mean = bn_weights[2]
    moving_variance = bn_weights[3]
    epsilon = bn_layer.epsilon

    # Get layer weights
    layer_weights = layer.get_weights()[0]
    layer_bias = (
        layer.get_weights()[1]
        if len(layer.get_weights()) > 1
        else np.zeros(gamma.shape)
    )

    # Compute fusion: W_new = W * gamma / sqrt(var + eps)
    inv_std = 1.0 / np.sqrt(moving_variance + epsilon)
    scale_factor = gamma * inv_std

    # Fuse weights: multiply each output channel by its scale factor
    if layer.__class__.__name__ == "Conv2D":
        # Conv2D weights shape: (kernel_h, kernel_w, input_channels, output_channels)
        fused_weights = layer_weights * scale_factor.reshape(1, 1, 1, -1)
    elif layer.__class__.__name__ == "DepthwiseConv2D":
        # W: (kh, kw, in_ch, depth_multiplier)
        in_ch = layer_weights.shape[2]
        depth_mult = layer_weights.shape[3]
        assert (
            scale_factor.size == in_ch * depth_mult
        ), "BN channels must match in_ch*depth_multiplier"
        fused_weights = layer_weights * scale_factor.reshape(1, 1, in_ch, depth_mult)
    elif layer.__class__.__name__ == "Dense":
        # Dense weights shape: (input_units, output_units)
        fused_weights = layer_weights * scale_factor.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")

    # Fuse bias: The correct equation is derived from BatchNorm math:
    # Original: y = gamma * (layer_out - mean) / sqrt(var + eps) + beta
    # Where layer_out = W_layer * x + b_layer
    # Expanding: y = gamma * (W_layer * x + b_layer - mean) / sqrt(var + eps) + beta
    # Rearranging: y = (gamma / sqrt(var + eps)) * W_layer * x + gamma * (b_layer - mean) / sqrt(var + eps) + beta
    # So: b_fused = gamma * (b_layer - mean) / sqrt(var + eps) + beta
    fused_bias = gamma * (layer_bias - moving_mean) * inv_std + beta

    # Get activation
    activation_name = "linear"
    if activation_layer:
        activation_name = activation_layer.get_config()["activation"]
        log_info(f"  → Including {activation_name} activation in fusion")
    return {"weights": fused_weights, "bias": fused_bias, "activation": activation_name}


def group_fuseable_layers(layers_data):
    """Group layers that can be fused together based on the JSON structure"""
    grouped_layers = {}
    layer_names = list(layers_data.keys())
    skip_layers = set()
    group_idx = 0

    for i, layer_name in enumerate(layer_names):
        if layer_name in skip_layers:
            continue

        layer_info = layers_data[layer_name]
        layer_type = layer_info.get("layer_type", "")

        # Start a new group with this layer
        group_key = f"fusable_layer_{group_idx}"
        group = {layer_name: layer_info}

        # Look for fuseable layers that follow
        j = i + 1
        while j < len(layer_names):
            next_layer_name = layer_names[j]
            next_layer_info = layers_data[next_layer_name]
            next_layer_type = next_layer_info.get("layer_type", "")

            # Check if this layer can be fused
            can_fuse = False

            # Conv2D/DepthwiseConv2D/Dense + BatchNormalization fusion
            if (
                layer_type in ["Conv2D", "DepthwiseConv2D", "Dense"]
                and next_layer_type == "BatchNormalization"
            ):
                can_fuse = True

            # BatchNormalization + Activation fusion
            elif layer_type == "BatchNormalization" and next_layer_type == "Activation":
                can_fuse = True

            # Direct Activation fusion with fuseable layers
            elif is_layer_fuseable(layer_type) and next_layer_type == "Activation":
                can_fuse = True

            # Dropout can be fused with previous layer
            elif next_layer_type == "Dropout":
                can_fuse = True

            if can_fuse:
                group[next_layer_name] = next_layer_info
                skip_layers.add(next_layer_name)
                # Update layer_type for next iteration
                layer_type = next_layer_type
                j += 1
            else:
                break

        grouped_layers[group_key] = group
        group_idx += 1

    return grouped_layers


def fuse_layers_from_json(input_file, output_file):
    """Read JSON file and create fused layer groups"""
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        log_debug(f"Loaded {len(data['layers'])} layers from {input_file}")

        # Group fuseable layers
        grouped_layers = group_fuseable_layers(data["layers"])

        # Create output structure
        output_data = {"input": data.get("input", {}), "layers": grouped_layers}

        # Write fused data to output file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        log_debug(
            f"Fused {len(data['layers'])} layers into {len(grouped_layers)} groups"
        )
        log_debug(f"Output written to {output_file}")

        return output_data

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing fusion: {e}")
        sys.exit(1)


def fuse_to_uph5_layer(model, fusion_groups):
    """Step 4: Focus on fusion

    Input: model, fusion_groups
    Processing:
        1. Each fusable_layer group will fuse into single layer in uph5 layer
        2. First layer in fusable_layer is primary layer
        3. Following batch normalization should fused with primary layer
            - First do the weight fusion then do the quantization
        4. Following "layer_type": "Activation", should fused activation into primary layer
        5. Using activation.min_val & activation.max_val → as primary layer min_val max_val → and use it calculated fused_layer scale and bias
    Output: fused_groups
    """

    fused_groups = {}
    model_idx_to_fused_idx = {}
    fused_order = []

    if not fusion_groups:
        log_debug("No fusion groups provided, returning empty fused groups")
        return fused_groups

    # Build lookup from Keras layer object id to model index for quick dependency mapping
    layer_obj_to_model_idx = {id(layer): idx for idx, layer in enumerate(model.layers)}

    for group_name, group_layers in fusion_groups.items():
        log_debug(f"Processing fusion group: {group_name}")

        # Find primary layer (first layer in group by index)
        primary_layer = None
        primary_layer_idx = None
        primary_layer_info = None
        min_idx = float("inf")

        group_model_indices = set()
        for layer_name, layer_info in group_layers.items():
            layer_idx = layer_info.get("layer_index", -1)
            if layer_idx >= 0:
                group_model_indices.add(layer_idx)
            if layer_idx < min_idx:
                min_idx = layer_idx
                primary_layer_idx = layer_idx
                primary_layer_info = layer_info

        if primary_layer_info is None or primary_layer_idx >= len(model.layers):
            log_warn(f"Layer Fuser: Invalid primary layer configuration in fusion group '{group_name}' - layer index {primary_layer_idx} not found in model. Skipping fusion group.")
            continue

        primary_layer = model.layers[primary_layer_idx]
        primary_layer_type = primary_layer_info.get("layer_type", "")

        # Get layer configuration from the primary layer
        layer_config = primary_layer.get_config()

        # Initialize fused layer structure with layer-specific parameters
        fused_layer = {
            "layer_type": primary_layer_type,
            "activation": "linear",  # Default
            "act_scale": 1.0,
            "act_zp": 0,
            "act_min_val": 0.0,
            "act_max_val": 1.0,
            "weight": None,
            "bias": None,
            "weight_scale": 1.0,
            "weight_zp": 0,
            "input_shape": (
                list(primary_layer.input_shape) if primary_layer.input_shape else []
            ),
            "output_shape": (
                list(primary_layer.output_shape) if primary_layer.output_shape else []
            ),
            "weight_shape": None,
        }

        # Add layer-specific configuration parameters
        if primary_layer_type in ["Conv2D", "Conv1D"]:
            fused_layer.update(
                {
                    "filters": layer_config.get("filters"),
                    "kernel_size": layer_config.get("kernel_size"),
                    "strides": layer_config.get("strides"),
                    "padding": layer_config.get("padding"),
                }
            )
        elif primary_layer_type == "DepthwiseConv2D":
            fused_layer.update(
                {
                    "depth_multiplier": layer_config.get("depth_multiplier", 1),
                    "kernel_size": layer_config.get("kernel_size"),
                    "strides": layer_config.get("strides"),
                    "padding": layer_config.get("padding"),
                }
            )
        elif primary_layer_type == "Dense":
            fused_layer.update(
                {
                    "units": layer_config.get("units"),
                }
            )
        elif primary_layer_type in ["MaxPooling2D", "AveragePooling2D"]:
            fused_layer.update(
                {
                    "pool_size": layer_config.get("pool_size"),
                    "strides": layer_config.get("strides"),
                    "padding": layer_config.get("padding"),
                }
            )

        # Get original weights from primary layer
        try:
            original_weights = primary_layer.get_weights()
            if original_weights:
                fused_layer["weight"] = original_weights[0]
                fused_layer["weight_shape"] = list(original_weights[0].shape)
                if len(original_weights) > 1:
                    fused_layer["bias"] = original_weights[1]
                else:
                    # Initialize bias if doesn't exist
                    if primary_layer_type in ["Conv2D", "DepthwiseConv2D"]:
                        bias_shape = original_weights[0].shape[-1]  # Output channels
                    else:
                        bias_shape = original_weights[0].shape[-1]  # Units for Dense
                    fused_layer["bias"] = np.zeros(bias_shape, dtype=np.float32)
        except Exception as e:
            log_warn(f"Layer Fuser: Cannot extract weights from primary layer '{primary_layer.name}' in group '{group_name}' - {e}. Fusion may be incomplete.")

        # Process other layers in the group for fusion
        batch_norm_layer = None
        activation_layer = None

        for layer_name, layer_info in group_layers.items():
            layer_idx = layer_info.get("layer_index", -1)
            layer_type = layer_info.get("layer_type", "")

            if layer_idx == primary_layer_idx:
                continue  # Skip primary layer

            if layer_type == "BatchNormalization" and layer_idx < len(model.layers):
                batch_norm_layer = model.layers[layer_idx]

            elif layer_type == "Activation" and layer_idx < len(model.layers):
                activation_layer = model.layers[layer_idx]

        # Apply BatchNorm fusion if present
        if batch_norm_layer and fused_layer["weight"] is not None:
            log_info(
                f"Fusing BatchNorm into {primary_layer_type} in group {group_name}"
            )
            fused_data = _fuse_layer_batchnorm(
                primary_layer, batch_norm_layer, activation_layer
            )
            fused_layer["weight"] = fused_data["weights"]
            fused_layer["bias"] = fused_data["bias"]
            fused_layer["activation"] = fused_data["activation"]
            fused_layer["weight_shape"] = list(fused_data["weights"].shape)

        # Apply activation fusion if present (and no BatchNorm fusion applied)
        elif activation_layer:
            activation_name = activation_layer.get_config()["activation"]
            fused_layer["activation"] = activation_name
            log_info(
                f"Fusing activation '{activation_name}' into {primary_layer_type} in group {group_name}"
            )

        # Extract quantization parameters from activation layer if present
        final_activation = fused_layer["activation"]

        # Look for activation layer in the fusion group to get its quantization params
        for layer_name, layer_info in group_layers.items():
            if layer_info.get("layer_type") == "Activation":
                # Extract activation layer's quantization parameters directly
                if "scale" in layer_info:
                    fused_layer["act_scale"] = layer_info["scale"]
                if "zero_point" in layer_info:
                    fused_layer["act_zp"] = layer_info["zero_point"]
                if "min_val" in layer_info:
                    fused_layer["act_min_val"] = layer_info["min_val"]
                if "max_val" in layer_info:
                    fused_layer["act_max_val"] = layer_info["max_val"]

                log_info(
                    f"Using activation layer quantization: min_val={layer_info.get('min_val')}, max_val={layer_info.get('max_val')}, scale={layer_info.get('scale')}"
                )
                break
        else:
            # No activation layer found, use primary layer params as fallback
            if "scale" in primary_layer_info:
                fused_layer["act_scale"] = primary_layer_info["scale"]
            if "zero_point" in primary_layer_info:
                fused_layer["act_zp"] = primary_layer_info["zero_point"]
            if "min_val" in primary_layer_info:
                fused_layer["act_min_val"] = primary_layer_info["min_val"]
            if "max_val" in primary_layer_info:
                fused_layer["act_max_val"] = primary_layer_info["max_val"]

            log_debug(
                f"Using primary layer quantization: min_val={primary_layer_info.get('min_val')}, max_val={primary_layer_info.get('max_val')}"
            )

        # Calculate weight quantization parameters
        if fused_layer["weight"] is not None:
            weight_scale, weight_zp = calculate_weight_params(
                fused_layer["weight"],
                primary_layer.name,
                primary_layer_type,
                primary_layer_idx,
            )
            fused_layer["weight_scale"] = weight_scale
            fused_layer["weight_zp"] = weight_zp

        # Calculate bias quantization parameters (independent of weight scale)
        if fused_layer.get("bias") is not None:
            bias_scale, bias_zp = calculate_bias_params(
                fused_layer["bias"],
                primary_layer.name,
                primary_layer_type,
                primary_layer_idx,
            )
            fused_layer["bias_scale"] = bias_scale
            fused_layer["bias_zp"] = bias_zp

        fused_layer["_model_layer_index"] = primary_layer_idx
        fused_layer["input_layer_indices"] = []

        fused_groups[group_name] = fused_layer
        fused_order.append(group_name)

        fused_idx = len(fused_order) - 1
        for model_idx in group_model_indices:
            model_idx_to_fused_idx[model_idx] = fused_idx
        log_info(
            f"Created fused layer for group {group_name}: {primary_layer_type} with {fused_layer['activation']} activation"
        )

    _attach_graph_dependencies(model, fused_groups, fused_order, model_idx_to_fused_idx, layer_obj_to_model_idx)
    return fused_groups


def combine_fused_data_step5(fusable_data, fused_groups):
    """Step 5: Combine fusable_data['input'] + fused_groups → get the fused_data

    Input: fusable_data (with 'input' section), fused_groups
    Output: fused_data with 'input' and 'layers' sections
    """
    fused_data = {"input": fusable_data.get("input", {}), "layers": fused_groups}
    log_debug(f"Input data keys: {list(fused_data['input'].keys())}")
    log_debug(f"Fused layer names: {list(fused_data['layers'].keys())}")

    return fused_data


def _attach_graph_dependencies(model, fused_groups, fused_order, model_idx_to_fused_idx, layer_obj_to_model_idx):
    """Populate input_layer_indices for each fused layer using the original Keras graph."""
    for order_idx, group_name in enumerate(fused_order):
        fused_layer = fused_groups.get(group_name, {})
        model_idx = fused_layer.pop("_model_layer_index", None)
        if model_idx is None or model_idx >= len(model.layers):
            continue

        keras_layer = model.layers[model_idx]
        inbound_layers = _collect_inbound_layers(keras_layer)

        resolved_indices = []
        seen = set()
        for inbound_layer in inbound_layers:
            if inbound_layer is None:
                continue
            if inbound_layer.__class__.__name__ == "InputLayer":
                continue

            inbound_model_idx = layer_obj_to_model_idx.get(id(inbound_layer))
            if inbound_model_idx is None:
                continue

            fused_input_idx = model_idx_to_fused_idx.get(inbound_model_idx)
            if fused_input_idx is None:
                continue
            if fused_input_idx == order_idx:
                continue
            if fused_input_idx in seen:
                continue

            seen.add(fused_input_idx)
            resolved_indices.append(fused_input_idx)

        fused_layer["input_layer_indices"] = resolved_indices


def _collect_inbound_layers(layer):
    """Collect inbound Keras layers feeding into the given layer."""
    inbound = []
    for node in getattr(layer, "_inbound_nodes", []):
        node_layers = []

        inbound_attr = getattr(node, "inbound_layers", None)
        if inbound_attr is not None:
            if isinstance(inbound_attr, (list, tuple)):
                node_layers.extend(inbound_attr)
            else:
                node_layers.append(inbound_attr)
        else:
            input_tensors = getattr(node, "input_tensors", None)
            if input_tensors is None:
                continue
            for tensor in input_tensors:
                history = getattr(tensor, "_keras_history", None)
                if history:
                    node_layers.append(history[0])

        inbound.extend(node_layers)

    return inbound


def main():
    """Main function to read kws_quantize_int16.json and output fused version"""
    input_file = "artifacts/kws_quantize_int16.json"
    output_file = "artifacts/fused_kws_quantize_int16.json"

    print(f"UPDL Layer Fusion Tool")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    fused_data = fuse_layers_from_json(input_file, output_file)

    print(f"\nFusion completed successfully!")
    print(
        f"Original layers: {sum(len(group) for group in fused_data['layers'].values())} layer entries"
    )
    print(f"Fused groups: {len(fused_data['layers'])}")

    # Show some example groups
    for i, (group_name, group_layers) in enumerate(
        list(fused_data["layers"].items())[:3]
    ):
        layer_types = [
            info.get("layer_type", "Unknown") for info in group_layers.values()
        ]
        print(f"  {group_name}: {', '.join(layer_types)}")
        if i == 2 and len(fused_data["layers"]) > 3:
            print(f"  ... and {len(fused_data['layers']) - 3} more groups")


if __name__ == "__main__":
    main()
