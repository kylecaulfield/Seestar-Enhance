"""Classical BM3D denoise applied to the full RGB image.

BM3D (Block-Matching 3D filtering) is a well-studied, patent-expired image
denoiser. We use the `bm3d` pip package (GPL-free, academic-permissive
implementation). Noise standard deviation is either supplied or estimated
from the image using a robust MAD-based estimator on the luminance channel.
"""
from __future__ import annotations

from typing import Optional

import bm3d
import numpy as np
from scipy.ndimage import laplace


def _estimate_sigma(image: np.ndarray) -> float:
    """Robust MAD noise estimate in the stretched intensity domain.

    Uses the Immerkaer-style estimator on the luminance channel: a Laplacian
    of the image is noise-dominated, and the MAD of its response divided by
    a known constant gives a fair sigma estimate.
    """
    luma = image.mean(axis=-1)
    response = laplace(luma.astype(np.float32))
    mad = float(np.median(np.abs(response - np.median(response))))
    sigma = mad * 1.4826 / 6.0
    return float(np.clip(sigma, 1e-4, 0.2))


def process(
    image: np.ndarray,
    sigma: Optional[float] = None,
    strength: float = 1.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    s = _estimate_sigma(img) if sigma is None else float(sigma)
    s = max(1e-4, s * float(strength))

    denoised = bm3d.bm3d_rgb(img, sigma_psd=s)
    return np.clip(denoised.astype(np.float32, copy=False), 0.0, 1.0)
