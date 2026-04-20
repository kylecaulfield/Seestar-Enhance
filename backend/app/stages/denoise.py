"""Placeholder: denoise. Pass-through until the ML model is wired in."""
from __future__ import annotations

import numpy as np


def process(image: np.ndarray, **_: object) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    return image.astype(np.float32, copy=False)
