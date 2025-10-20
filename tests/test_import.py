#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Test basic imports and package structure.
"""

import sys
import os

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_core_imports():
    """Test that core modules can be imported."""
    from updl_compiler.core import config
    from updl_compiler.core import logger
    from updl_compiler.core import loader
    from updl_compiler.core import quantizer
    from updl_compiler.core import fuser
    from updl_compiler.core import serializer
    # If we get here without ImportError, the test passes

def test_main_api():
    """Test that main API can be imported."""
    from updl_compiler import compile_model
    # If we get here without ImportError, the test passes

def test_cli():
    """Test that CLI can be imported."""
    from updl_compiler.cli import main
    # If we get here without ImportError, the test passes

if __name__ == "__main__":
    # For direct execution, run pytest
    import pytest
    pytest.main([__file__])