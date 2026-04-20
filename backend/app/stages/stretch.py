"""Arcsinh stretch with auto black-point and white-point selection.

Black and white points are chosen as low/high histogram percentiles of the
luminance so the sky floor maps to zero and the brightest real structure
maps toward one. Seestar raw data typically fills only a small slice of
the [0, 1] range after debayer/background/color, so normalizing by the
actual white point (not by 1.0) is essential to get a visible image.
The arcsinh curve then compresses highlights while preserving faint
structure.
"""
from __future__ import annotations

import numpy as np


def process(
    image: np.ndarray,
    black_percentile: float = 0.1,
    white_percentile: float = 99.9,
    stretch: float = 25.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    luma = img.mean(axis=-1)
    black = float(np.percentile(luma, black_percentile))
    white = float(np.percentile(luma, white_percentile))

    denom = max(white - black, 1e-6)
    shifted = np.clip(img - black, 0.0, None)
    normalized = np.clip(shifted / denom, 0.0, 1.0)

    stretched = np.arcsinh(normalized * stretch) / np.arcsinh(stretch)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
