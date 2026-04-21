"""Arcsinh stretch with auto black-point and white-point selection.

Black and white points are chosen as low/high histogram percentiles of the
luminance so the sky floor maps to zero and the brightest real structure
maps toward one. Seestar raw data typically fills only a small slice of
the [0, 1] range after debayer/background/color, so normalizing by the
actual white point (not by 1.0) is essential to get a visible image.
The arcsinh curve then compresses highlights while preserving faint
structure.

Per-channel black subtraction: each channel gets its own percentile black
point subtracted before the shared luma-derived normalization. This zeroes
out the per-channel sky pedestal differences that WB and Bayer demosaic
leave behind, preventing those tiny offsets from becoming large chromatic
blotches after the extreme arcsinh amplification.
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

    img = image.astype(np.float32, copy=True)

    # Per-channel black subtraction equalises sky pedestals across R/G/B
    # before the nonlinear stretch amplifies inter-channel differences.
    for c in range(3):
        black_c = float(np.percentile(img[..., c], black_percentile))
        img[..., c] = np.clip(img[..., c] - black_c, 0.0, None)

    luma = img.mean(axis=-1)
    white = float(np.percentile(luma, white_percentile))
    denom = max(white, 1e-6)
    normalized = np.clip(img / denom, 0.0, 1.0)

    stretched = np.arcsinh(normalized * stretch) / np.arcsinh(stretch)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
