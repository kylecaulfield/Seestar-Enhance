"""Cosmetic correction — hot pixels and cosmic rays.

Single-frame processing can't recover from sigma-clipped stacking, so
bright outliers (hot pixels, cosmic-ray hits, residual single-frame
noise peaks in a stack) survive into the pipeline. The arcsinh stretch
then amplifies them into vivid rainbow dots because their per-channel
values are uncorrelated.

This stage replaces pixels that are dramatically brighter than their
immediate neighbourhood with the neighbourhood median. It's a classical
sigma-clipped despike that runs pre-stretch in linear space, so the
outliers never reach arcsinh.

Contract: pure transform on `(H, W, 3)` float32 in `[0, 1]`; never
returns a pixel brighter than its input.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def process(
    image: np.ndarray,
    neighborhood: int = 3,
    sigma: float = 6.0,
) -> np.ndarray:
    """Replace bright outliers with local median.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3).
    neighborhood : int
        Median-filter window size. 3 catches single pixels; 5 catches
        2-pixel clusters (useful on heavily binned / Bayer-pattern data).
    sigma : float
        Rejection threshold in units of the robust (MAD) standard
        deviation of the residual. 6 is conservative — only obvious
        outliers get replaced, real stars (which are also local peaks
        but over several pixels) are preserved.

    Returns
    -------
    np.ndarray
        The input image with hot pixels / cosmic rays replaced by the
        per-channel median of their neighbourhood.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if neighborhood < 3:
        raise ValueError(f"neighborhood must be >= 3, got {neighborhood}")

    img = image.astype(np.float32, copy=True)
    for c in range(3):
        chan = img[..., c]
        med = median_filter(chan, size=neighborhood, mode="nearest")
        residual = chan - med
        mad = float(np.median(np.abs(residual - np.median(residual))))
        # 1.4826 converts MAD to Gaussian-equivalent standard deviation.
        threshold = sigma * mad * 1.4826
        if threshold <= 0:
            continue
        hot = residual > threshold
        chan[hot] = med[hot]
    return np.clip(img, 0.0, 1.0).astype(np.float32)
