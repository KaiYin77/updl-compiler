#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
High-level orchestration for the UPDL compilation pipeline.

CompilationSession encapsulates the canonical workflow:
    1. Load/normalize the source TensorFlow model.
    2. Run calibration to derive quantization parameters.
    3. Initialize the quantizer and build fusion groups.
    4. Serialize fused data into UPH5 artifacts.

This keeps the public compile_model() API thin while making the
sub-steps individually testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import validate_layer_configuration
from .fuser import (
    combine_fused_data_step5,
    fuse_layers_from_json,
    fuse_to_uph5_layer,
)
from .loader import load_model
from .logger import (
    LOG_LEVEL_INFO,
    log_info,
    set_log_level,
)
from .calibrator import Calibrator
from .quantizer import initialize_params
from .codegen import (
    serialize_uph5_metadata_to_json,
    serialize_uph5_to_c_array,
)
from ..core.preprocessors import DataPreprocessor


@dataclass
class CompilationResult:
    """Artifact summary returned to callers."""

    model_name: str
    file_size: int
    metadata_json: str
    uph5_dir: str
    output_dir: str
    quantization_params: str
    calibration_samples: int
    cache_dir: str


class CompilationSession:
    """Stateful helper that runs the full UPDL compilation workflow."""

    def __init__(
        self,
        *,
        model_path: str,
        preprocessor: DataPreprocessor,
        calibration_source: Any,
        calibration_count: int | None,
        model_name: str | None,
        description: str,
        output_dir: str | None,
        log_level: int = LOG_LEVEL_INFO,
    ) -> None:
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.calibration_source = calibration_source
        self.calibration_count = calibration_count
        self.description = description or "no_description"
        self.output_dir = output_dir or "."
        self.model_name = model_name or os.path.splitext(os.path.basename(model_path))[0]
        self.cache_dir = os.path.join(self.output_dir, ".updlc_cache")
        self._log_level = log_level

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> CompilationResult:
        """Execute all compilation stages and return artifact metadata."""
        self._prepare_environment()

        validate_layer_configuration()
        model = self._load_model()
        calibration_samples = self._load_calibration_samples()

        quant_params_path = self._run_calibration(model, calibration_samples)
        self._initialize_quant_parameters(quant_params_path)
        fused_data = self._fuse_layers(model, quant_params_path)
        metadata_json, uph5_dir, file_size = self._emit_outputs(fused_data, quant_params_path)

        return CompilationResult(
            model_name=self.model_name,
            file_size=file_size,
            metadata_json=metadata_json,
            uph5_dir=uph5_dir,
            output_dir=self.output_dir,
            quantization_params=quant_params_path,
            calibration_samples=len(calibration_samples),
            cache_dir=self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _prepare_environment(self) -> None:
        """Apply log configuration and ensure output directories exist."""
        set_log_level(self._log_level)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_model(self):
        """Frontend load/normalization stage."""
        log_info(f"Model Loader: Loading {self.model_path}")
        return load_model(self.model_path)

    def _load_calibration_samples(self) -> List[Any]:
        """Request representative data via the provided preprocessor."""
        if not self.calibration_source:
            raise ValueError("Calibrator: calibration_data is required for quantization analysis")

        log_info("Calibrator: Loading calibration samples")
        if self.calibration_count is None:
            samples = self.preprocessor.load_calibration_data(self.calibration_source)
        else:
            samples = self.preprocessor.load_calibration_data(
                self.calibration_source, count=self.calibration_count
            )

        if not samples:
            raise ValueError(
                "Calibrator: No calibration samples were loaded. "
                "Check dataset availability or preprocessing configuration."
            )

        return samples

    def _run_calibration(self, model, calibration_samples: List[Any]) -> str:
        """Run quantization analysis and persist parameters to disk."""
        analyzer = Calibrator(preprocessor=self.preprocessor)
        quant_params_path = os.path.join(
            self.cache_dir, f"{self.model_name}_quantize_params.json"
        )
        log_info(f"Calibrator: Starting analysis with {len(calibration_samples)} samples")
        analyzer.analyze_model_quantization(
            model,
            calibration_samples,
            quant_params_path,
            udl_mode=True,
        )
        return quant_params_path

    def _initialize_quant_parameters(self, quant_params_path: str) -> None:
        """Load analyzer output into the quantizer backend."""
        if not initialize_params(quant_params_path, udl_mode=True):
            raise RuntimeError(
                "Parameter Initializer: Failed to load quantization parameters "
                f"from {quant_params_path}"
            )

    def _fuse_layers(self, model, quant_params_path: str) -> Dict[str, Any]:
        """Run fusion passes using quantization metadata."""
        base_name = os.path.splitext(os.path.basename(quant_params_path))[0]
        fusion_json = os.path.join(self.cache_dir, f"fusable_{base_name}.json")
        fusable_data = fuse_layers_from_json(quant_params_path, fusion_json)
        fusable_layers = fusable_data["layers"]

        log_info(f"Layer Fuser: Processing {len(model.layers)} layers")
        fused_layers = fuse_to_uph5_layer(model, fusable_layers)
        fused_data = combine_fused_data_step5(fusable_data, fused_layers)
        return fused_data

    def _emit_outputs(
        self, fused_data: Dict[str, Any], quant_params_path: str
    ) -> tuple[str, str, int]:
        """Serialize fused representation into UPH5 artifacts."""
        base_name = os.path.splitext(os.path.basename(quant_params_path))[0]
        metadata_json = os.path.join(self.cache_dir, f"uph5_metadata_{base_name}.json")
        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
        serialize_uph5_metadata_to_json(fused_data, metadata_json)

        uph5_dir = os.path.join(self.output_dir, "uph5")
        os.makedirs(uph5_dir, exist_ok=True)
        file_size = serialize_uph5_to_c_array(
            fused_data,
            self.model_name,
            description=self.description,
            output_dir=uph5_dir,
            cache_dir=self.cache_dir,
        )
        return metadata_json, uph5_dir, file_size
