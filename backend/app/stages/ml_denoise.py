"""ML-based denoise — v2 STUB.

Intentionally unimplemented in v1. v1 uses classical BM3D (see
`bm3d_denoise.py`) so the project stays MIT-clean. When a permissively
licensed ONNX denoiser or a self-trained model is available, implement
`process` with the signature below so the pipeline wiring does not change.

Planned implementation sketch (do not enable in v1):
    - load ONNX model from backend/app/models/denoise_<name>.onnx
    - tile input into 256 or 512 px tiles with 32 px overlap
    - run onnxruntime per tile
    - blend seams with a cosine window
    - return float32 RGB in [0, 1]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# TODO(v2): accept `model_path` and load an ONNX session lazily.
def process(
    image: np.ndarray,
    model_path: Optional[Path] = None,
    tile_size: int = 512,
    overlap: int = 32,
    strength: float = 1.0,
) -> np.ndarray:
    """Run an ML denoiser over the image. Not implemented in v1."""
    del image, model_path, tile_size, overlap, strength
    raise NotImplementedError(
        "ML denoise is a v2 feature. See README v2 roadmap. "
        "v1 uses classical BM3D via bm3d_denoise.process()."
    )
