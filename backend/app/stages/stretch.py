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

Histogram-based auto-stretch: when `stretch` is set to "auto" (string),
the arcsinh strength is derived from the image itself — specifically
from the dynamic range of the sky-to-target luma histogram, so frames
with dim targets get more lift and frames with bright targets stay
conservative. `black_percentile` can also be "auto" in which case it
maps to the knee of the luma histogram (low-luma median + 1.5 MAD).
"""
from __future__ import annotations

from typing import Union

import numpy as np

Floatable = Union[float, str]


def _auto_black_percentile(luma: np.ndarray) -> float:
    """Find the knee of the low-luma histogram.

    Returns a percentile (e.g. 0.3) that lands just above the sky median
    + 1.5 MAD, which is the traditional "clip the sky, keep the signal"
    rule of thumb.
    """
    sky_floor = float(np.percentile(luma, 10.0))
    above_floor = luma[luma <= sky_floor + 1e-6]
    if above_floor.size < 16:
        return 0.1
    med = float(np.median(above_floor))
    mad = float(np.median(np.abs(above_floor - med))) * 1.4826
    knee_val = med + 1.5 * mad
    # Convert to percentile.
    return float((luma < knee_val).mean() * 100.0)


def _auto_stretch_factor(luma: np.ndarray, black_pc: float, white_pc: float) -> float:
    """Pick an arcsinh strength from the sky-to-target dynamic range.

    The range between black_pc and white_pc (in normalized units) tells
    us how bright the target is relative to the sky. Dim targets (narrow
    range) need more aggressive arcsinh so their signal actually shows;
    bright targets (wide range) need less so the core doesn't clip.
    """
    black = float(np.percentile(luma, black_pc))
    white = float(np.percentile(luma, white_pc))
    span = max(white - black, 1e-6)
    # Empirical fit: very dim targets (span ~0.01) want stretch ~25;
    # bright targets (span ~0.3+) want stretch ~10. Log-scale so the
    # curve is smooth across the dynamic range.
    return float(np.clip(-12.0 * np.log10(span + 1e-6) - 10.0, 6.0, 30.0))


def process(
    image: np.ndarray,
    black_percentile: Floatable = 0.1,
    white_percentile: float = 99.9,
    stretch: Floatable = 25.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=True)
    luma_preview = img.mean(axis=-1)

    # Resolve "auto" parameters up front so subsequent calls are deterministic.
    bp = (
        _auto_black_percentile(luma_preview)
        if isinstance(black_percentile, str) and black_percentile == "auto"
        else float(black_percentile)
    )
    s_factor = (
        _auto_stretch_factor(luma_preview, bp, white_percentile)
        if isinstance(stretch, str) and stretch == "auto"
        else float(stretch)
    )

    # Per-channel black subtraction equalises sky pedestals across R/G/B
    # before the nonlinear stretch amplifies inter-channel differences.
    for c in range(3):
        black_c = float(np.percentile(img[..., c], bp))
        img[..., c] = np.clip(img[..., c] - black_c, 0.0, None)

    luma = img.mean(axis=-1)
    white = float(np.percentile(luma, white_percentile))
    denom = max(white, 1e-6)
    normalized = np.clip(img / denom, 0.0, 1.0)

    stretched = np.arcsinh(normalized * s_factor) / np.arcsinh(s_factor)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
