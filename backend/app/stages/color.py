"""Background neutralization + simple white balance.

1. Neutralization: subtract per-channel median of dark pixels so the sky is
   grey rather than tinted.
2. White balance: scale channels so their mid-brightness medians match.
"""
from __future__ import annotations

import numpy as np


def _percentile_mask(luma: np.ndarray, low: float, high: float) -> np.ndarray:
    lo = float(np.percentile(luma, low))
    hi = float(np.percentile(luma, high))
    if hi <= lo:
        hi = lo + 1e-6
    return (luma >= lo) & (luma <= hi)


def process(
    image: np.ndarray,
    dark_percentile: float = 25.0,
    mid_low: float = 40.0,
    mid_high: float = 80.0,
    wb_strength: float = 1.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=True)
    luma = img.mean(axis=-1)

    dark_thresh = float(np.percentile(luma, dark_percentile))
    dark_mask = luma <= dark_thresh
    if dark_mask.sum() < 64:
        dark_mask = np.ones_like(luma, dtype=bool)

    for c in range(3):
        ch = img[..., c]
        pedestal = float(np.median(ch[dark_mask]))
        img[..., c] = ch - pedestal

    img = np.clip(img, 0.0, None)

    mid_mask = _percentile_mask(luma, mid_low, mid_high)
    if mid_mask.sum() < 64:
        mid_mask = np.ones_like(luma, dtype=bool)

    medians = np.array(
        [float(np.median(img[..., c][mid_mask])) for c in range(3)],
        dtype=np.float32,
    )
    target = float(medians.mean())
    if target <= 0:
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    gains = np.where(medians > 1e-6, target / medians, 1.0).astype(np.float32)
    gains = 1.0 + wb_strength * (gains - 1.0)

    for c in range(3):
        img[..., c] *= gains[c]

    return np.clip(img, 0.0, 1.0).astype(np.float32)
