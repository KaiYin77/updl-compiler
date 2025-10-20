# KWS (Keyword Spotting) Example for UPDL Compiler

Compile keyword spotting models to UPH5 format for embedded deployment. Uses models from [MLCommons Tiny Benchmark](https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting).

## Files

- `tf_model/` - TensorFlow model files
  - `kws_model_split.h5` - Main quantized model (197KB)
  - `kws_model.h5` - Alternative model (207KB)
  - `kws_ref_model/` - Reference SavedModel format
- `quant_config/` - Quantization configuration
  - `kws_quantize_int16_ref.json` - Pre-computed quantization parameters
- `uph5/` - Generated C files for embedding (50,550 bytes)
- `.updlc_cache/` - Compilation cache files

## Usage

```bash
# Command line (UDL mode enabled by default)
updl-compile --input tf_model/kws_model_split.h5 --quantization-params quant_config/kws_quantize_int16_ref.json

# Python API (simplified interface)
from updl_compiler import compile_model

compile_model(
    model="tf_model/kws_model_split.h5",
    quant_config="quant_config/kws_quantize_int16_ref.json"
)
```

## Output

Generated files in `uph5/`:
- `kws_model_split_int16.c` - C array with quantized model data
- `kws_model_split_int16.h` - Header file for embedding

Ready for integration into embedded C/C++ projects.

## License

Copyright 2025 Upbeat, Inc
SPDX-License-Identifier: Apache-2.0