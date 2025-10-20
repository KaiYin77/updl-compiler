#!/bin/bash
# Installation script for UPDL Compiler dependencies
# Run this when ready to install dependencies

echo "Installing UPDL Compiler dependencies with UV..."

# Install core dependencies (this may take several minutes due to TensorFlow)
uv add tensorflow==2.13.1 numpy h5py

# Install development dependencies
uv add --dev pytest black flake8 mypy

echo "Dependencies installed successfully!"
echo "To activate the environment: source .venv/bin/activate"
echo "To run tests: uv run pytest"
echo "To build package: uv build"