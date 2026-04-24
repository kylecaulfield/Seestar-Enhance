"""Richardson-Lucy deconvolution for star-PSF tightening.

Astronomical imaging through atmosphere + optics smears star light into
a PSF of typical FWHM ~3 px on the Seestar S50. Deconvolving with a
synthetic Gaussian kernel of matching width narrows the stars back
toward point sources before the stretch amplifies their outer wings.

Deconvolution is applied to the LUMA only:
  1. Stars contribute to luma, their colour is dominated by the core
     pixel and is a spatially-constant feature we'd rather not sharpen
     (colour fringing looks worse than slight PSF bloom).
  2. Chroma channels on Seestar data have half the spatial resolution
     of luma (RGGB), so deconvolving them introduces artefacts.

Contract: `(H, W, 3)` float32 in `[0, 1]` in, same out.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import richardson_lucy


def _gaussian_psf(sigma: float, size: int) -> np.ndarray:
    """Normalised 2-D Gaussian PSF centred in a `size x size` array."""
    half = (size - 1) / 2.0
    y, x = np.mgrid[:size, :size].astype(np.float32) - half
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)


def process(
    image: np.ndarray,
    psf_sigma: float = 1.5,
    iterations: int = 8,
    psf_size: int = 11,
) -> np.ndarray:
    """Richardson-Lucy deconvolve the luma; keep the chroma untouched.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3).
    psf_sigma : float
        Gaussian PSF standard deviation in pixels. 1.5 approximates a
        typical Seestar star at native resolution (FWHM ≈ 3.5 px).
    iterations : int
        Richardson-Lucy iterations. 5-10 is the sweet spot — fewer
        and the PSF barely tightens; more and noise starts to grow.
    psf_size : int
        Support size of the PSF kernel. 11 covers ±5 px of a 1.5-sigma
        Gaussian, where the kernel is already <1 % of its peak.

    Returns
    -------
    np.ndarray
        Float32 RGB image in [0, 1] with tightened stars.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    img = image.astype(np.float32, copy=False)
    luma = img.mean(axis=-1)
    chroma = img - luma[..., np.newaxis]

    psf = _gaussian_psf(float(psf_sigma), int(psf_size))
    deconv_luma = richardson_lucy(
        np.clip(luma, 0.0, 1.0).astype(np.float32),
        psf,
        num_iter=int(iterations),
        clip=True,
    ).astype(np.float32)

    out = deconv_luma[..., np.newaxis] + chroma
    return np.clip(out, 0.0, 1.0).astype(np.float32)
