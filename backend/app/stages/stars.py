"""Star removal — v2 STUB.

Intentionally unimplemented in v1. Shipping a star-removal model requires
model weights under a license compatible with this repo (MIT). StarNet++ v2
weights are CC-BY-NC-SA and therefore cannot be bundled here; GraXpert's
weight licensing is ambiguous. See the v2 roadmap in README.md.

When permissively-licensed or self-trained ONNX weights become available,
implement `process` below with the signature kept stable so the pipeline
can slot it in without other changes.

Planned implementation sketch (do not enable in v1):
    - load an ONNX model from backend/app/models/starnet_<name>.onnx
    - tile input into 256 or 512 px tiles with 32 px overlap
    - run onnxruntime per tile
    - blend seams with a cosine window
    - return (stars_only, starless) both float32 RGB in [0,1]
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return (stars_only, starless). Not implemented in v1."""
    del image, model_path, tile_size, overlap
    raise NotImplementedError(
        "Star removal is a v2 feature. See README v2 roadmap. "
        "Use the classical pipeline (no star removal) for v1."
    )
