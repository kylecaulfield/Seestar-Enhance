"""Classical BM3D denoise applied to the full RGB image.

BM3D (Block-Matching 3D filtering) is a well-studied, patent-expired image
denoiser. We use the `bm3d` pip package (GPL-free, academic-permissive
implementation). Noise standard deviation is either supplied or estimated
from the image using a robust MAD-based estimator on the luminance channel.

Seestar frames are dominated by cross-channel "color noise" after an
aggressive stretch (the raw Bayer R/B planes are undersampled, so the
demosaic amplifies per-channel differences). BM3D alone reduces it
partially; an optional chroma-blur pass smooths the color-difference
channels further while leaving the luma intact, which is the usual astro
workflow for getting rid of chromatic speckle without softening stars.

Optional `downsample_factor` gives a large speed-up for full Seestar
frames: BM3D cost scales ~quadratically with pixel count, so running at
half resolution cuts wall time by ~4×. We upsample the denoised result
back with Lanczos, then take the detail (`image - denoised`) at full
resolution from the original to recover high-frequency structure — i.e.
denoise the low-frequency component only. On the Seestar sensor this is
nearly indistinguishable from full-res BM3D for the noise scales we care
about (chroma speckle is a low-frequency phenomenon).
"""
from __future__ import annotations

from typing import Optional

import bm3d
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace


def _estimate_sigma(image: np.ndarray) -> float:
    """Robust MAD noise estimate in the stretched intensity domain.

    The Laplacian kernel is noise-dominated compared to smooth image
    structure, so its robust spread is a fair sigma proxy. For scipy's
    standard 5-point Laplacian the response variance is 8*sigma^2, so
    sigma ≈ std(response) / sqrt(8). The classical /6 divisor applies to
    Immerkaer's 3x3 second-order kernel, which is not what scipy uses.
    """
    luma = image.mean(axis=-1)
    response = laplace(luma.astype(np.float32))
    mad = float(np.median(np.abs(response - np.median(response))))
    sigma = mad * 1.4826 / np.sqrt(8.0)
    return float(np.clip(sigma, 1e-4, 0.3))


def _chroma_smooth(image: np.ndarray, radius: float) -> np.ndarray:
    """Blur the per-channel difference from luma.

    Keeps luminance detail intact (stars, edges, filaments) while
    flattening chromatic speckle — the kind of noise BM3D can leave
    behind on a hard-stretched Seestar frame.
    """
    if radius <= 0:
        return image
    luma = image.mean(axis=-1, keepdims=True)
    diff = image - luma
    smoothed = np.empty_like(diff)
    for c in range(3):
        smoothed[..., c] = gaussian_filter(diff[..., c], sigma=radius)
    return np.clip(luma + smoothed, 0.0, 1.0).astype(np.float32)


def _resize(image: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Lanczos resize an (H, W, 3) float32 image to (new_h, new_w, 3)."""
    new_h, new_w = new_shape
    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    resized = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def process(
    image: np.ndarray,
    sigma: Optional[float] = None,
    strength: float = 1.0,
    chroma_blur: float = 0.0,
    downsample_factor: int = 1,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)
    s = _estimate_sigma(img) if sigma is None else float(sigma)
    s = max(1e-4, s * float(strength))

    if downsample_factor <= 1:
        denoised = bm3d.bm3d_rgb(img, sigma_psd=s).astype(np.float32, copy=False)
        denoised = np.clip(denoised, 0.0, 1.0)
    else:
        # Denoise a downscaled copy; recover high-frequency detail from
        # the original. Appropriate for suppressing chroma speckle which
        # is overwhelmingly a low-frequency phenomenon on Seestar data.
        h, w = img.shape[:2]
        dh = max(64, h // downsample_factor)
        dw = max(64, w // downsample_factor)
        small = _resize(img, (dh, dw))
        small_denoised = bm3d.bm3d_rgb(small, sigma_psd=s).astype(np.float32, copy=False)
        low_freq = _resize(np.clip(small_denoised, 0.0, 1.0), (h, w))
        # Original - upsampled-denoised-low-freq = high-frequency detail
        # that survived the noise floor. Add it back to the smoothed base.
        low_freq_orig = _resize(small, (h, w))
        high_freq = img - low_freq_orig
        denoised = np.clip(low_freq + high_freq, 0.0, 1.0).astype(np.float32)
    return _chroma_smooth(denoised, chroma_blur)
