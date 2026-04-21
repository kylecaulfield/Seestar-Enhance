"""Mild S-curve contrast and saturation boost.

Star preservation: a saturation > 1 naively applied to the whole image
will push already-bright, already-coloured star pixels into hue clipping
(red stars go pure red, blue go pure blue). `star_preserve_percentile`
defines a luma threshold above which the saturation gain is tapered
back toward 1.0 — colored stars keep their hue without blowing out.
Defaults are conservative so the extended-structure targets (galaxy
halos, nebulae) still get the full boost.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _s_curve(x: np.ndarray, strength: float) -> np.ndarray:
    # Smooth S-curve pinned at (0,0) and (1,1) with a controllable midpoint slope.
    k = float(strength)
    if k <= 0:
        return x
    a = 1.0 + k
    centered = 2.0 * x - 1.0
    shaped = np.tanh(a * centered) / np.tanh(a)
    return (shaped + 1.0) * 0.5


def process(
    image: np.ndarray,
    contrast: float = 0.6,
    saturation: float = 1.2,
    star_preserve_percentile: Optional[float] = 99.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    img = _s_curve(np.clip(img, 0.0, 1.0), contrast)

    luma = img.mean(axis=-1, keepdims=True)
    if star_preserve_percentile is not None and saturation != 1.0:
        # Taper the saturation multiplier back toward 1.0 for the
        # brightest ~1% of pixels so coloured stars don't clip in hue.
        luma_flat = luma[..., 0]
        thresh = float(np.percentile(luma_flat, star_preserve_percentile))
        if thresh > 0:
            # Linear ramp from full saturation (at threshold) to identity
            # (at 2x threshold and above).
            over = np.clip((luma_flat - thresh) / max(thresh, 1e-6), 0.0, 1.0)
            eff_sat = saturation - (saturation - 1.0) * over
            eff_sat = eff_sat[..., np.newaxis]
            img = luma + (img - luma) * eff_sat
        else:
            img = luma + (img - luma) * float(saturation)
    else:
        img = luma + (img - luma) * float(saturation)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
