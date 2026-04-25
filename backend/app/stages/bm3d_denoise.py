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


def _chroma_bilateral(image: np.ndarray, radius: float, luma_sigma: float) -> np.ndarray:
    """Edge-preserving chroma smooth: chroma gets smoothed strongly in
    flat-luma regions and weakly (or not at all) across luma edges.

    Naively computing local luma variance triggers the "edge" branch
    everywhere on noisy sky (the noise is what creates the variance),
    so we first smooth the luma at a large scale to get a stable
    estimate of the "real" structure. Then we compute the LOCAL
    gradient magnitude of that smoothed luma; only where this
    gradient is large do we stop smoothing chroma.

    Net effect: nebula-to-sky transitions (genuine large gradient)
    keep their chroma separation; noisy flat sky / noisy flat nebula
    (no large gradient at the smoothed scale) get the full chroma
    smoothing and become locally neutral.
    """
    if radius <= 0:
        return image
    luma = image.mean(axis=-1, keepdims=True)
    chroma = image - luma
    luma_2d = luma[..., 0]

    # 1. Smooth luma at a generous scale so noise doesn't masquerade
    #    as an "edge". Scale ≈ the chroma-smoothing radius.
    smoothed_l = gaussian_filter(luma_2d, sigma=float(radius))

    # 2. Gradient magnitude on the smoothed luma.
    gy, gx = np.gradient(smoothed_l)
    grad_mag = np.sqrt(gy * gy + gx * gx).astype(np.float32)

    # 3. Edge weight: 1 where the smoothed luma is flat, 0 at edges.
    #    luma_sigma is the gradient magnitude at which we switch off
    #    chroma smoothing; larger → more aggressive smoothing.
    edge_w = np.exp(-((grad_mag / max(luma_sigma, 1e-6)) ** 2)).astype(np.float32)
    edge_w = edge_w[..., np.newaxis]

    # 4. Gaussian-blur the chroma channels.
    smoothed_chroma = np.empty_like(chroma)
    for c in range(3):
        smoothed_chroma[..., c] = gaussian_filter(chroma[..., c], sigma=float(radius))

    # 5. Blend: flat-luma regions take the smoothed chroma, edges keep
    #    the original.
    out_chroma = smoothed_chroma * edge_w + chroma * (1.0 - edge_w)
    return np.clip(luma + out_chroma, 0.0, 1.0).astype(np.float32)


def _resize(image: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Lanczos resize an (H, W, 3) float32 image to (new_h, new_w, 3)."""
    new_h, new_w = new_shape
    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    resized = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def process(
    image: np.ndarray,
    sigma: float | None = None,
    strength: float = 1.0,
    chroma_blur: float = 0.0,
    downsample_factor: int = 1,
    chroma_edge_aware: bool = False,
    chroma_edge_luma_sigma: float = 0.05,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=False)

    # Guard: the upstream `bm3d` package divides by (block_max -
    # block_min) per 8x8 block and returns NaN whenever it hits a
    # constant block — full-frame constant input, gradients along
    # one axis, blank corrupt FITS, etc. Skip BM3D entirely on
    # zero-variance frames (no noise to denoise anyway) and apply
    # only the chroma blur, which handles constants fine.
    if float(img.max() - img.min()) < 1e-7:
        if chroma_edge_aware and chroma_blur > 0:
            return _chroma_bilateral(img, chroma_blur, chroma_edge_luma_sigma)
        return _chroma_smooth(img, chroma_blur)

    s = _estimate_sigma(img) if sigma is None else float(sigma)
    s = max(1e-4, s * float(strength))

    if downsample_factor <= 1:
        denoised = bm3d.bm3d_rgb(img, sigma_psd=s).astype(np.float32, copy=False)
        # Block-level NaN escape hatch: if BM3D hit a degenerate block
        # (gradient along one axis is the typical case in tests), fall
        # back to the original pixel rather than poisoning the output.
        if not np.all(np.isfinite(denoised)):
            denoised = np.where(np.isfinite(denoised), denoised, img)
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
        # Same NaN guard as the full-res branch — degenerate blocks
        # in the downsampled image NaN-out small_denoised, and the
        # downstream low_freq + high_freq math then poisons every
        # pixel. Fall back to the un-denoised downsampled input
        # before the upsample so the recovery is symmetric.
        if not np.all(np.isfinite(small_denoised)):
            small_denoised = np.where(np.isfinite(small_denoised), small_denoised, small)
        low_freq = _resize(np.clip(small_denoised, 0.0, 1.0), (h, w))
        # Original - upsampled-denoised-low-freq = high-frequency detail
        # that survived the noise floor. Add it back to the smoothed base.
        low_freq_orig = _resize(small, (h, w))
        high_freq = img - low_freq_orig
        denoised = np.clip(low_freq + high_freq, 0.0, 1.0).astype(np.float32)
    if chroma_edge_aware and chroma_blur > 0:
        return _chroma_bilateral(denoised, chroma_blur, chroma_edge_luma_sigma)
    return _chroma_smooth(denoised, chroma_blur)
