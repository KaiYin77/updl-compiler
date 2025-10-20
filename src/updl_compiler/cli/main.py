#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
from ..core.logger import (
    LOG_LEVEL_OFF,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_TRACE,
    set_log_level,
    log_info,
    log_error,
)
from ..core.config import validate_layer_configuration
from ..core.loader import load_model
from ..core.fuser import (
    fuse_layers_from_json,
    fuse_to_uph5_layer,
    combine_fused_data_step5,
)
from ..core.quantizer import initialize_params
from ..core.serializer import serialize_uph5_metadata_to_json, serialize_uph5_weight_to_json, serialize_uph5_to_c_array


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 model to C array format for embedded deployment"
    )
    parser.add_argument("--model", required=True, help="Input HDF5 file path")
    parser.add_argument(
        "--description",
        default="no_description",
        help="Model description (max 32 chars)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (auto-extracted from filename if not provided)"
    )
    parser.add_argument(
        "--quant-config",
        required=True,
        help="JSON file with quantization parameters",
    )
    parser.add_argument(
        "--log-level",
        choices=["off", "error", "warn", "info", "debug", "trace"],
        default="info",
        help="Set log level (default: info)",
    )
    return parser.parse_args()


def main():
    """Main entry point - streamlined workflow"""
    args = parse_args()

    # Set log level
    log_level_map = {
        "off": LOG_LEVEL_OFF,
        "error": LOG_LEVEL_ERROR,
        "warn": LOG_LEVEL_WARN,
        "info": LOG_LEVEL_INFO,
        "debug": LOG_LEVEL_DEBUG,
        "trace": LOG_LEVEL_TRACE,
    }
    set_log_level(log_level_map[args.log_level])

    try:
        # === STREAMLINED WORKFLOW ===
        log_info("=== UPDL Model Conversion Pipeline ===")

        # 0. Validate configuration
        validate_layer_configuration()

        # 1. Load model using updl_loader
        log_info("Step 1: Loading keras model...")
        model = load_model(args.model)

        # Auto-extract model name if not provided
        if args.model_name is None:
            args.model_name = os.path.splitext(os.path.basename(args.model))[0]
            log_info(f"Auto-extracted model name: {args.model_name}")

        # 2. Load quantization parameters (UDL mode always enabled)
        log_info(
            f"Step 2: Loading pre-computed quantization parameters from {args.quant_config}"
        )
        log_info("UDL shift-only mode ENABLED - All scales will be converted to power-of-2 values")

        if not initialize_params(args.quant_config, udl_mode=True):
            raise RuntimeError(
                f"Failed to load quantization parameters from {args.quant_config}"
            )

        # 3. Create fusion groups JSON → fusion_data
        log_info("Step 3: Grouping fusable layers...")
        # Generate fusion filename based on quantization params filename
        base_name = os.path.splitext(os.path.basename(args.quant_config))[0]
        cache_dir = "./.updlc_cache"
        os.makedirs(cache_dir, exist_ok=True)
        fusion_json = os.path.join(cache_dir, f"fusable_{base_name}.json")
        fusable_data = fuse_layers_from_json(args.quant_config, fusion_json)
        fusable_layers = fusable_data["layers"]

        # 4. Focus on fusion to uph5 layer
        log_info("Step 4: Processing fusable_layers into uph5 layer...")
        fused_layers = fuse_to_uph5_layer(model, fusable_layers)
        log_info(
            f"Fusion processing complete: {len(fused_layers)} fused layers created"
        )

        # 5. Combine fusable_data['input'] + fused_layers → fused_data
        log_info("Step 5: Combining data structures...")
        fused_data = combine_fused_data_step5(fusable_data, fused_layers)
        log_info("Combined fused_data structure created")

        # 6. Focus on serialization (clean separation)
        log_info("Step 6a: Serializing metadata to json...")
        # Save cache files to .updlc_cache directory
        uph5_json = os.path.join(cache_dir, f"uph5_metadata_{base_name}.json")
        serialize_uph5_metadata_to_json(fused_data, uph5_json)

        log_info("Step 6b: Serializing CHW-optimized weights to json...")
        weights_debug_json = os.path.join(cache_dir, f"uph5_weights_{base_name}.json")
        serialize_uph5_weight_to_json(fused_data, weights_debug_json)

        log_info("Step 6c: Serializing to C array...")
        # Save C array files to uph5 directory
        output_dir = "uph5"
        os.makedirs(output_dir, exist_ok=True)
        file_size = serialize_uph5_to_c_array(
            fused_data, args.model_name, description=args.description, output_dir=output_dir
        )
        log_info(f"Generated C array files in {output_dir}/")

        log_info(f"File size: {file_size} bytes")
        log_info("=== Conversion completed successfully! ===")

    except Exception as e:
        log_error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
