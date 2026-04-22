"""Mild S-curve contrast and saturation boost.

Star preservation: a saturation > 1 naively applied to the whole image
will push already-bright, already-coloured star pixels into hue clipping
(red stars go pure red, blue go pure blue). `star_preserve_percentile`
defines a luma threshold above which the saturation gain is tapered
back toward 1.0 — colored stars keep their hue without blowing out.
Defaults are conservative so the extended-structure targets (galaxy
halos, nebulae) still get the full boost.

Optional `channel_gains` applies a per-channel multiplicative offset
before the S-curve. Useful for emission-nebula targets where the sensor
WB cannot recover the true Ha-dominant colour from a mostly-grey sky
stack. Defaults to (1, 1, 1) — no change.

`saturation_mode` picks between:
  - "linear" (default): scale chroma around the per-pixel luma mean.
    Fast and simple but can shift hue on saturated pixels.
  - "hsv": convert to HSV, scale S, convert back. Hue-preserving by
    construction — a pure-red pixel stays pure red, just more
    saturated. What PixInsight / Siril / Photoshop call "saturation".
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from skimage.color import hsv2rgb, rgb2hsv


def _s_curve(x: np.ndarray, strength: float) -> np.ndarray:
    # Smooth S-curve pinned at (0,0) and (1,1) with a controllable midpoint slope.
    k = float(strength)
    if k <= 0:
        return x
    a = 1.0 + k
    centered = 2.0 * x - 1.0
    shaped = np.tanh(a * centered) / np.tanh(a)
    return (shaped + 1.0) * 0.5


def _apply_saturation_linear(
    img: np.ndarray,
    sat: np.ndarray,  # (H, W, 1) or scalar
) -> np.ndarray:
    luma = img.mean(axis=-1, keepdims=True)
    return luma + (img - luma) * sat


def _apply_saturation_hsv(
    img: np.ndarray,
    sat: np.ndarray,
) -> np.ndarray:
    """Hue-preserving saturation scale via HSV round-trip."""
    clipped = np.clip(img, 0.0, 1.0)
    hsv = rgb2hsv(clipped).astype(np.float32)
    # sat may be scalar or (H, W, 1); align to the S channel shape.
    if np.ndim(sat) == 0:
        s_scale = float(sat)
        hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0.0, 1.0)
    else:
        hsv[..., 1] = np.clip(hsv[..., 1] * sat[..., 0], 0.0, 1.0)
    return hsv2rgb(hsv).astype(np.float32)


def process(
    image: np.ndarray,
    contrast: float = 0.6,
    saturation: float = 1.2,
    star_preserve_percentile: Optional[float] = 99.0,
    channel_gains: Optional[Sequence[float]] = None,
    saturation_mode: str = "linear",
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=True)
    img = _s_curve(np.clip(img, 0.0, 1.0), contrast)
    # Apply channel_gains AFTER the S-curve, weighted by luma so dark
    # pixels (sky) stay nearly unchanged while bright pixels (nebula,
    # stars) get the full colour shift. Without the luma-weighting,
    # multiplying a residual sky tint by 1.25R / 0.8B leaves a pink
    # cast across the entire frame.
    if channel_gains is not None:
        if len(channel_gains) != 3:
            raise ValueError(
                f"channel_gains must have 3 values, got {len(channel_gains)}"
            )
        luma = img.mean(axis=-1, keepdims=True)  # (H, W, 1) in [0,1]
        gains = np.asarray(channel_gains, dtype=np.float32)[None, None, :]
        # effective gain = 1 + (gain - 1) * luma
        effective = 1.0 + (gains - 1.0) * luma
        img = np.clip(img * effective, 0.0, 1.0).astype(np.float32)

    if saturation_mode not in ("linear", "hsv"):
        raise ValueError(
            f"saturation_mode must be 'linear' or 'hsv', got {saturation_mode!r}"
        )

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
        else:
            eff_sat = np.float32(saturation)
    else:
        eff_sat = np.float32(saturation)

    if saturation_mode == "hsv":
        img = _apply_saturation_hsv(img, eff_sat)
    else:
        img = _apply_saturation_linear(img, eff_sat)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
