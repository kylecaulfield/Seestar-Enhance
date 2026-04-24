"""Dark-frame subtraction.

Removes sensor-specific fixed-pattern noise when the caller supplies a
master dark (a stack of unexposed frames at the same temperature and
exposure as the lights). Common astrophotography calibration step.

The master dark is expected in the same FITS format as the light: a
2-D Bayer mosaic or a (H, W, 3) RGB array in `[0, 1]`. A 2-D Bayer
dark is demosaicked the same way as the light before subtraction.

Clamps the subtracted result to `[0, ∞)` so dark-current overshoot
doesn't introduce negative values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PathLike = str | Path


def process(
    image: np.ndarray,
    dark: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract a master-dark image from the light.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3) in [0, 1].
    dark : np.ndarray | None
        Master-dark RGB array of matching shape. If None, returns the
        input unchanged (no-op when the caller doesn't supply one).
    scale : float
        Scale factor applied to the dark before subtraction. 1.0 for
        equal exposure; scale down if the dark has longer exposure.

    Returns
    -------
    np.ndarray
        image - scale * dark, clamped to [0, 1].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if dark is None:
        return image
    if dark.shape != image.shape:
        raise ValueError(f"dark shape {dark.shape} does not match light shape {image.shape}")

    out = image.astype(np.float32, copy=False) - float(scale) * dark.astype(np.float32, copy=False)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
