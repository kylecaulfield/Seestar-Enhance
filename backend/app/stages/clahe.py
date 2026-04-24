"""Local contrast enhancement via CLAHE (luma-only).

Contrast-Limited Adaptive Histogram Equalization stretches local tile
histograms independently so dust-lane / filament detail at intermediate
scales (~50-200 px on the Seestar sensor) gets lifted without blowing
out bright regions. Classical, deterministic, no model weights.

We apply CLAHE to the luma only and reattach the original chroma, so
there's no coloured halo on edges. The contract matches the rest of
the pipeline: in/out are `(H, W, 3)` float32 in `[0, 1]`.
"""

from __future__ import annotations

import numpy as np
from skimage.exposure import equalize_adapthist


def process(
    image: np.ndarray,
    clip_limit: float = 0.01,
    kernel_size: int = 128,
    blend: float = 0.6,
) -> np.ndarray:
    """Apply luma-only CLAHE and reattach the original chroma.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3).
    clip_limit : float
        CLAHE clip limit (0..1). Higher = more aggressive local
        contrast. Skimage's default is 0.01.
    kernel_size : int
        CLAHE tile size in pixels. 128 suits Seestar's 1920x1080 for
        mid-scale structure (galaxy dust lanes, nebula filament edges).
    blend : float
        Blend factor between the CLAHE output and the original luma.
        1.0 = full CLAHE, 0.0 = no change. 0.6 gives visible punch
        without over-cooking.

    Returns
    -------
    np.ndarray
        Float32 RGB image in [0, 1] with CLAHE-enhanced luma.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")

    img = image.astype(np.float32, copy=False)
    luma = img.mean(axis=-1)
    chroma = img - luma[..., np.newaxis]

    # equalize_adapthist expects input in [0, 1] and returns float in [0, 1].
    enhanced = equalize_adapthist(
        np.clip(luma, 0.0, 1.0),
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    ).astype(np.float32)

    blended_luma = luma * (1.0 - blend) + enhanced * blend
    out = blended_luma[..., np.newaxis] + chroma
    return np.clip(out, 0.0, 1.0).astype(np.float32)
