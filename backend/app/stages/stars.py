"""Classical star/starless separation.

v2 of Seestar Enhance: separate an image into a starless component (the
extended structure — nebulae, galaxy disks, diffuse background) and a
stars component (compact bright features), stretch each appropriately,
then recombine. The win is that nebulae can be stretched HARD without
bloating star cores or inducing star-halo ringing.

Implementation is classical, MIT-clean, and requires no model weights:

  starless = per-channel median filter of the input
  stars    = max(0, input - starless)

Median filtering with a window 2x larger than the typical PSF removes
compact star signatures while preserving extended emission (nebula
filaments, galaxy disks), because:

  - median is a robust statistic: a bright outlier in a 15x15 window
    has almost no effect on the output.
  - smooth variation at scales larger than the window passes through
    unchanged: a nebula filament wider than ~7 px keeps its flux.

Tradeoffs vs ML separation:
  - fast and deterministic; no GPU, no onnxruntime.
  - imperfect: galaxy cores get slightly smoothed (acceptable when the
    galaxy is larger than the window, which is the common case).
  - does not recover wings of saturated stars.

For nebula targets this trade is a massive net win — community Seestar
stacks routinely use StarNet for this exact reason.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter


def process(image: np.ndarray, radius: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Split an RGB image into (stars_only, starless).

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image in [0, 1] with shape (H, W, 3).
    radius : int
        Radius in pixels of the median-filter window used to estimate the
        starless image. The window diameter must exceed the largest star
        PSF (including wings) you want to remove. 7 px (15 px window) is
        a sensible default for Seestar data at native resolution.

    Returns
    -------
    stars_only : np.ndarray
        Per-pixel flux attributed to compact sources, shape (H, W, 3),
        values in [0, 1].
    starless : np.ndarray
        Everything that is NOT a star, shape (H, W, 3), values in [0, 1].
        Sum of the two approximates the input (with stars = input - starless).
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    window = int(2 * radius + 1)
    img = image.astype(np.float32, copy=False)

    starless = np.empty_like(img)
    for c in range(3):
        starless[..., c] = median_filter(
            img[..., c],
            size=window,
            mode="reflect",
        )

    # Clip both: median on noisy data can overshoot very slightly.
    starless = np.clip(starless, 0.0, 1.0).astype(np.float32)
    stars_only = np.clip(img - starless, 0.0, 1.0).astype(np.float32)
    return stars_only, starless


def recombine(stars_only: np.ndarray, starless: np.ndarray) -> np.ndarray:
    """Combine a stretched/denoised starless back with the original stars.

    Uses the "screen" blend (1 - (1-a)(1-b)), which is the optical model
    of two emitters adding in a camera: never exceeds 1.0, matches how
    stars sit "on top of" diffuse background in reality. Straight
    addition would require a clip and would shift hues on bright stars.
    """
    if stars_only.shape != starless.shape:
        raise ValueError(
            f"shape mismatch: stars={stars_only.shape} starless={starless.shape}"
        )
    s = np.clip(stars_only.astype(np.float32, copy=False), 0.0, 1.0)
    b = np.clip(starless.astype(np.float32, copy=False), 0.0, 1.0)
    return (1.0 - (1.0 - s) * (1.0 - b)).astype(np.float32)
