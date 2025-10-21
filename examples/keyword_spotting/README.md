# KWS (Keyword Spotting) Example

Streamlined compilation of keyword spotting models to UPH5 format with automatic quantization analysis. Uses models from [MLCommons Tiny Benchmark](https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting).

## Files

- `tf_model/` - TensorFlow model files
  - `kws_ref_model/` - Original SavedModel format (207KB)
- `kws_preprocessor.py` - KWS-specific data preprocessor
- `kws_compiler.py` - Example script demonstrating streamlined workflow
- `uph5/` - Generated C files for embedding (50,550 bytes)
- `.updlc_cache/` - Compilation cache files (quantization params, metadata)

## Streamlined Workflow

**One command from original model to embedded C code:**

```bash
# Command line with automatic quantization
updl-compile --model tf_model/kws_ref_model/ --preprocessor kws --dataset /path/to/speech_commands

# Python API
from updl_compiler import compile_model
from kws_preprocessor import KWSPreprocessor

result = compile_model(
    model="tf_model/kws_ref_model/",
    preprocessor=KWSPreprocessor(),
    calibration_data="/path/to/speech_commands"
)
```

**What happens automatically:**
1. Loads original TensorFlow model
2. Loads calibration data from speech_commands dataset
3. Performs quantization analysis using 120 calibration samples
4. Generates quantization parameters → `.updlc_cache/`
5. Compiles to UPH5 format → `uph5/`

## Output

Generated files:
- `.updlc_cache/kws_uph5_model_quantize_params.json` - Auto-generated quantization parameters
- `uph5/kws_uph5_model_int16.c` - C array with quantized model data (50,550 bytes)
- `uph5/kws_uph5_model_int16.h` - Header file for embedding

Ready for integration into embedded C/C++ projects.

## Custom Preprocessors

For other model types, implement the `DataPreprocessor` interface:

```python
from updl_compiler.core.preprocessors import DataPreprocessor

class MyPreprocessor(DataPreprocessor):
    def preprocess_sample(self, sample):
        # Convert raw sample to model input format
        return processed_tensor

    def load_calibration_data(self, data_source, count=120):
        # Load calibration samples
        return sample_list
```

## License

Copyright 2025 Upbeat, Inc
SPDX-License-Identifier: Apache-2.0