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
    try:
        from updl_compiler.core import config
        from updl_compiler.core import logger
        from updl_compiler.core import loader
        from updl_compiler.core import quantizer
        from updl_compiler.core import fuser
        from updl_compiler.core import serializer
        print("✓ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False

def test_main_api():
    """Test that main API can be imported."""
    try:
        from updl_compiler import compile_model
        print("✓ Main API imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import main API: {e}")
        return False

def test_cli():
    """Test that CLI can be imported."""
    try:
        from updl_compiler.cli import main
        print("✓ CLI imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False

def main():
    """Run all import tests."""
    print("Testing UPDL Compiler package structure...")
    print()

    tests_passed = 0
    total_tests = 3

    if test_core_imports():
        tests_passed += 1

    if test_main_api():
        tests_passed += 1

    if test_cli():
        tests_passed += 1

    print()
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✓ All import tests passed!")
        return 0
    else:
        print("✗ Some import tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())