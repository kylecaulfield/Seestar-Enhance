"""Background neutralization + white balance + optional SCNR.

1. Neutralization: subtract per-channel median of dark pixels so the sky is
   grey rather than tinted. Optionally Mahalanobis-trimmed so coloured
   stars / nebula leakage in the sky mask don't bias the pedestal.
2. White balance: scale channels so their mid-brightness medians match.
   Optionally exempts the brightest ~1% of pixels from the gain so that
   red M-type and blue O-type stars keep their colour instead of being
   pulled to white.
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
5. Optional sensor CCM (color correction matrix): a 3x3 linear transform
   applied to each pixel first. Used to pre-compensate for known sensor
   quirks like the Seestar S50 IR leak into R.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


# Seestar S50 CCM. Empirically-tuned (not factory-calibrated) to
# compensate for:
#   - IR leak in R from the broad dual-band filter (pulls R down)
#   - Green dominance from the RGGB 2G:1R:1B sampling (pulls G down
#     slightly, relying mostly on SCNR to finish the job)
#   - Blue that's well-behaved; row preserves it with a tiny boost to
#     offset the softer blue-channel SNR
# The matrix is near-identity so it nudges rather than remaps the
# colour cube. Rows must sum to 1.0 to preserve neutral luma.
SEESTAR_S50_CCM: np.ndarray = np.array(
    [
        [0.92, -0.05, 0.13],  # R' = 0.92R - 0.05G + 0.13B  (subtract IR leak)
        [-0.05, 0.95, 0.10],  # G' gets a ~5% trim
        [-0.02, -0.03, 1.05],  # B' gets a modest lift
    ],
    dtype=np.float32,
)


def _percentile_mask(luma: np.ndarray, low: float, high: float) -> np.ndarray:
    lo = float(np.percentile(luma, low))
    hi = float(np.percentile(luma, high))
    if hi <= lo:
        hi = lo + 1e-6
    return (luma >= lo) & (luma <= hi)


def _mahalanobis_trimmed_median(
    samples: np.ndarray,
    sigma_cutoff: float = 2.5,
) -> np.ndarray:
    """Per-channel median over samples within `sigma_cutoff` Mahalanobis
    distance of the sample-cloud centre.

    Each row of `samples` is an RGB observation. Compared to a plain
    per-channel median, this rejects outliers in the *joint* RGB
    distribution — points with unusual colour, not just unusual
    brightness — so coloured stars, nebula leakage, and hot pixels
    don't pull the estimate.

    Falls back to the plain median if the covariance is singular.
    """
    if samples.ndim != 2 or samples.shape[-1] != 3 or samples.shape[0] < 16:
        return np.median(samples, axis=0).astype(np.float32)

    x = samples.astype(np.float32)
    mean = x.mean(axis=0)
    centred = x - mean
    cov = np.cov(centred, rowvar=False)
    try:
        inv = np.linalg.inv(cov + 1e-8 * np.eye(3, dtype=np.float32))
    except np.linalg.LinAlgError:
        return np.median(x, axis=0).astype(np.float32)

    d2 = np.einsum("ni,ij,nj->n", centred, inv, centred)
    kept = x[d2 < (sigma_cutoff * sigma_cutoff) * 3]
    if kept.shape[0] < 16:
        return np.median(x, axis=0).astype(np.float32)
    return np.median(kept, axis=0).astype(np.float32)


def process(
    image: np.ndarray,
    dark_percentile: float = 25.0,
    mid_low: float = 40.0,
    mid_high: float = 80.0,
    wb_strength: float = 1.0,
    green_clip: float = 0.0,
    pre_stretch_chroma_smooth: float = 0.0,
    pre_stretch_chroma_lowpass: float = 0.0,
    mahalanobis_wb: bool = True,
    star_protect_percentile: Optional[float] = 99.0,
    ccm: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    img = image.astype(np.float32, copy=True)

    if ccm is not None:
        # Apply the 3x3 colour correction matrix: new = old @ ccm.T.
        # Accepts a literal matrix or the string "seestar_s50" shortcut.
        if isinstance(ccm, str):
            if ccm == "seestar_s50":
                m = SEESTAR_S50_CCM
            else:
                raise ValueError(f"unknown ccm preset {ccm!r}")
        else:
            m = np.asarray(ccm, dtype=np.float32)
            if m.shape != (3, 3):
                raise ValueError(f"ccm must be 3x3, got shape {m.shape}")
        img = np.clip(img @ m.T, 0.0, None).astype(np.float32)
    luma = img.mean(axis=-1)

    dark_thresh = float(np.percentile(luma, dark_percentile))
    dark_mask = luma <= dark_thresh
    if dark_mask.sum() < 64:
        dark_mask = np.ones_like(luma, dtype=bool)

    # Per-channel dark pedestal. Mahalanobis-trimmed version rejects
    # coloured outliers inside the dark mask (e.g. a red star at low
    # luma) that would otherwise pull the median.
    dark_pixels = img[dark_mask]  # (N, 3)
    if mahalanobis_wb and dark_pixels.shape[0] >= 64:
        pedestal = _mahalanobis_trimmed_median(dark_pixels)
    else:
        pedestal = np.array(
            [float(np.median(img[..., c][dark_mask])) for c in range(3)],
            dtype=np.float32,
        )
    for c in range(3):
        img[..., c] = img[..., c] - float(pedestal[c])

    img = np.clip(img, 0.0, None)

    mid_mask = _percentile_mask(luma, mid_low, mid_high)
    if mid_mask.sum() < 64:
        mid_mask = np.ones_like(luma, dtype=bool)

    mid_pixels = img[mid_mask]
    if mahalanobis_wb and mid_pixels.shape[0] >= 64:
        medians = _mahalanobis_trimmed_median(mid_pixels)
    else:
        medians = np.array(
            [float(np.median(img[..., c][mid_mask])) for c in range(3)],
            dtype=np.float32,
        )
    target = float(medians.mean())
    if target > 0:
        gains = np.where(medians > 1e-6, target / medians, 1.0).astype(np.float32)
        gains = 1.0 + wb_strength * (gains - 1.0)
        # Star protection: blend the WB gain toward 1.0 (no change) for
        # the brightest pixels, so coloured stars keep their native
        # spectrum instead of being pulled to the sky-balanced white point.
        if star_protect_percentile is not None:
            star_thresh = float(np.percentile(luma, star_protect_percentile))
            if star_thresh > 0:
                # smooth ramp from full WB (below threshold) to identity
                # (well above); width = 2x threshold above = full protection.
                weight = np.clip((luma - star_thresh) / max(star_thresh, 1e-6), 0.0, 1.0)
                weight = weight[..., np.newaxis]  # (H, W, 1)
                for c in range(3):
                    img[..., c] = img[..., c] * (
                        gains[c] * (1.0 - weight[..., 0]) + 1.0 * weight[..., 0]
                    )
            else:
                for c in range(3):
                    img[..., c] *= gains[c]
        else:
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

    if pre_stretch_chroma_lowpass > 0.0:
        # Low-pass on chroma: Gaussian-blur the per-channel deviation
        # from luma, then reattach to the (unblurred) luma. Smooths the
        # pixel-scale chroma noise that otherwise gets amplified to
        # visible rainbow speckle by an aggressive arcsinh. Luma (stars,
        # nebula edges) is untouched; features larger than sigma in the
        # chroma (nebula color) are preserved.
        luma3 = img.mean(axis=-1, keepdims=True)
        chroma = img - luma3
        sigma = float(pre_stretch_chroma_lowpass)
        smoothed_chroma = np.stack(
            [gaussian_filter(chroma[..., c], sigma=sigma) for c in range(3)],
            axis=-1,
        )
        img = np.clip(luma3 + smoothed_chroma, 0.0, None).astype(np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)
