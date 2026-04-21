"""Background neutralization + simple white balance + optional SCNR.

1. Neutralization: subtract per-channel median of dark pixels so the sky is
   grey rather than tinted.
2. White balance: scale channels so their mid-brightness medians match.
3. Optional SCNR (Subtractive Chromatic Noise Reduction) "average neutral":
   clip the green channel so it never exceeds the average of red and blue.
   Standard astrophotography fix for the inherent green bias of RGGB
   Bayer sensors — the Seestar in particular leaves a strong green cast
   in diffuse regions that WB alone can't remove once the nebulosity
   itself is channel-biased.
4. Optional pre-stretch chroma smoothing: Gaussian-blur the per-channel
   deviation from luma to suppress large-scale chromatic non-uniformity
   (Bayer demosaic residuals, LP gradient residuals) before the arcsinh
   stretch amplifies those tiny differences into visible colour blotches.
   Must be done in linear space (before stretch), not after.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


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
    green_clip: float = 0.0,
    pre_stretch_chroma_smooth: float = 0.0,
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
    if target > 0:
        gains = np.where(medians > 1e-6, target / medians, 1.0).astype(np.float32)
        gains = 1.0 + wb_strength * (gains - 1.0)
        for c in range(3):
            img[..., c] *= gains[c]

    if green_clip > 0.0:
        # SCNR average neutral: pull G down toward (R+B)/2 when it
        # exceeds that threshold. amount=1.0 is a hard clip; anything less
        # is a soft blend so real green signal (rare in deep-sky targets,
        # but real stars with green-ish tints exist) isn't fully erased.
        amount = float(np.clip(green_clip, 0.0, 1.0))
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        threshold = 0.5 * (r + b)
        clipped = np.minimum(g, threshold)
        img[..., 1] = g * (1.0 - amount) + clipped * amount

    if pre_stretch_chroma_smooth > 0.0:
        # High-pass chroma equalization: estimate the slowly-varying
        # per-channel color offset (LP gradient residuals, Bayer demosaic
        # structure) and subtract it, leaving luma and fine-scale chroma
        # (star colours, galaxy detail) intact. Must be done in linear
        # space before arcsinh amplifies the tiny offsets into visible blobs.
        luma3 = img.mean(axis=-1, keepdims=True)
        chroma = img - luma3  # per-channel deviation from luma
        sigma = float(pre_stretch_chroma_smooth)
        slow_chroma = np.stack(
            [gaussian_filter(chroma[..., c], sigma=sigma) for c in range(3)],
            axis=-1,
        )
        # Remove only the slow colour variation; keep fast chroma intact.
        img = np.clip(img - slow_chroma, 0.0, None).astype(np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)
