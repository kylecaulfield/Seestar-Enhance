"""Mild S-curve contrast and saturation boost."""
from __future__ import annotations

import numpy as np


def _s_curve(x: np.ndarray, strength: float) -> np.ndarray:
    # Smooth S-curve pinned at (0,0) and (1,1) with a controllable midpoint slope.
    k = float(strength)
    if k <= 0:
        return x
    a = 1.0 + k
    centered = 2.0 * x - 1.0
    shaped = np.tanh(a * centered) / np.tanh(a)
    return (shaped + 1.0) * 0.5


def process(
    image: np.ndarray,
    contrast: float = 0.6,
    saturation: float = 1.2,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    img = _s_curve(np.clip(img, 0.0, 1.0), contrast)

    luma = img.mean(axis=-1, keepdims=True)
    img = luma + (img - luma) * float(saturation)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
