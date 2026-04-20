"""Arcsinh stretch with auto black-point selection.

Black point is chosen as a low histogram percentile of the luminance so that
the darkest sky pixels map to zero. The arcsinh curve compresses highlights
while preserving faint structure.
"""
from __future__ import annotations

import numpy as np


def process(
    image: np.ndarray,
    black_percentile: float = 0.1,
    stretch: float = 25.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    luma = img.mean(axis=-1)
    black = float(np.percentile(luma, black_percentile))

    shifted = np.clip(img - black, 0.0, None)
    denom = max(1.0 - black, 1e-6)
    normalized = shifted / denom

    stretched = np.arcsinh(normalized * stretch) / np.arcsinh(stretch)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
