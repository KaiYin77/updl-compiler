#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
FlatBuffers Configuration for UPDL
Optional feature - maintains existing UPH5 format by default
"""

import os
import subprocess
from pathlib import Path

SCHEMA_FILE = Path(__file__).parent / "src/updl_compiler/core/schema/updl_model.fbs"
OUTPUT_DIR = Path(__file__).parent / "src/updl_compiler/core/schema"

def generate_flatbuffers_bindings():
    """Generate FlatBuffers bindings if flatc compiler is available"""
    try:
        # Check if flatc is available
        result = subprocess.run(["flatc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("FlatBuffers compiler (flatc) not found - skipping generation")
            return False

        # Generate Python bindings
        cmd = [
            "flatc", "--python",
            "--gen-mutable",
            "--gen-object-api",
            "-o", str(OUTPUT_DIR),
            str(SCHEMA_FILE)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Generated FlatBuffers Python bindings in {OUTPUT_DIR}")
            return True
        else:
            print(f"FlatBuffers generation failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("FlatBuffers compiler not installed - using legacy UPH5 format")
        return False

if __name__ == "__main__":
    generate_flatbuffers_bindings()