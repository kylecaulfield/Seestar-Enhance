"""Gentle unsharp mask."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def process(
    image: np.ndarray,
    radius: float = 1.5,
    amount: float = 0.4,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    blurred = np.empty_like(img)
    for c in range(3):
        blurred[..., c] = gaussian_filter(img[..., c], sigma=radius)

    high = img - blurred
    sharpened = img + amount * high
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)
