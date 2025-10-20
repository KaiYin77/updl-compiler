# UPDL Compiler

A Python package for converting TensorFlow models to UPH5 (Upbeat Portable HDF5) format optimized for UP301/201 Processors.

## Overview

UPDL Compiler transforms Keras/TensorFlow models into efficient, quantized representations suitable for deployment on embedded devices like UP301/UP201 processors. It provides:

## Installation

```bash
# Upcoming support for PyPI
```

## Quick Start

```bash
# CLI usage
updl-compile --model model.h5 --quant-config params.json

# Python API
from updl_compiler import compile_model

result = compile_model(
    model="model.h5",
    quant_config="params.json"
)
```

## Development

```bash
# Setup development environment
uv sync --dev

# Run tests
uv run pytest

# Build package
uv build
```

## License

Copyright 2025 Upbeat, Inc
SPDX-License-Identifier: Apache-2.0