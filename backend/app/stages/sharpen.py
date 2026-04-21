"""Luma-only unsharp mask.

Sharpening per-channel on RGB introduces coloured fringing at edges when
the three channels disagree on exact edge position (which they do after
demosaic). Sharpening only the luma and adding the high-pass equally to
all three channels keeps chroma intact while boosting edge contrast on
brightness — standard astrophotography practice.
"""
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

    img = image.astype(np.float32, copy=True)
    luma = img.mean(axis=-1)
    blurred_luma = gaussian_filter(luma, sigma=float(radius))
    high = luma - blurred_luma  # luma-only high-pass
    # Broadcast the luma high-pass across the three colour channels so the
    # sharpening lifts edges without changing R:G:B ratios anywhere.
    img = img + amount * high[..., np.newaxis]
    return np.clip(img, 0.0, 1.0).astype(np.float32)
