"""RBF-based gradient/background removal.

Samples the image on a coarse grid, sigma-clips the samples to reject stars
and bright nebulosity, fits a smooth radial basis function through the
remaining sky samples, and subtracts it from each channel independently.

An optional `sky_mask` parameter lets the caller supply a pre-computed
boolean map of "this pixel is sky". Grid samples that fall outside the
mask are rejected before sigma-clipping — much more principled than
hoping sigma-clipping can find the sky on its own for frames where the
subject dominates the histogram (dense star clusters, tight galaxy
framings).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


def _sigma_clip(values: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    mask = np.ones_like(values, dtype=bool)
    for _ in range(iters):
        sel = values[mask]
        if sel.size == 0:
            break
        med = float(np.median(sel))
        mad = float(np.median(np.abs(sel - med))) * 1.4826
        if mad == 0.0:
            break
        new_mask = (values >= med - sigma * mad) & (values <= med + sigma * mad)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def _fit_channel(
    channel: np.ndarray,
    grid: int,
    sigma: float,
    iters: int,
    smoothing: float,
    downscale: int,
    sky_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = channel.shape
    ys = np.linspace(0, h - 1, grid, dtype=np.float32)
    xs = np.linspace(0, w - 1, grid, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    sample_y = np.clip(yy.astype(np.int32), 0, h - 1)
    sample_x = np.clip(xx.astype(np.int32), 0, w - 1)
    samples = channel[sample_y, sample_x].ravel().astype(np.float32)

    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

    mask = _sigma_clip(samples, sigma=sigma, iters=iters)
    # Further restrict to caller-supplied sky mask if provided.
    if sky_mask is not None:
        mask_sampled = sky_mask[sample_y, sample_x].ravel().astype(bool)
        mask = mask & mask_sampled

    if mask.sum() < 16:
        return np.full_like(channel, float(np.median(samples)), dtype=np.float32)

    rbf = RBFInterpolator(
        coords[mask],
        samples[mask],
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    dh = max(1, h // downscale)
    dw = max(1, w // downscale)
    qy = np.linspace(0, h - 1, dh, dtype=np.float32)
    qx = np.linspace(0, w - 1, dw, dtype=np.float32)
    gy, gx = np.meshgrid(qy, qx, indexing="ij")
    query = np.stack([gy.ravel(), gx.ravel()], axis=1)
    low = rbf(query).reshape(dh, dw).astype(np.float32)

    from scipy.ndimage import zoom

    scale_y = h / dh
    scale_x = w / dw
    full = zoom(low, (scale_y, scale_x), order=3, mode="nearest")
    return full[:h, :w].astype(np.float32)


def process(
    image: np.ndarray,
    grid: int = 24,
    sigma: float = 2.5,
    iters: int = 5,
    smoothing: float = 1.0,
    downscale: int = 8,
    sky_mask: np.ndarray | None = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if sky_mask is not None:
        if sky_mask.shape != image.shape[:2] or sky_mask.dtype != bool:
            raise ValueError(
                f"sky_mask must be bool of shape {image.shape[:2]}, got "
                f"{sky_mask.shape} / {sky_mask.dtype}"
            )

    out = np.empty_like(image, dtype=np.float32)
    for c in range(3):
        bg = _fit_channel(
            image[..., c].astype(np.float32, copy=False),
            grid=grid,
            sigma=sigma,
            iters=iters,
            smoothing=smoothing,
            downscale=downscale,
            sky_mask=sky_mask,
        )
        pedestal = float(np.median(bg))
        out[..., c] = image[..., c] - bg + pedestal

    return np.clip(out, 0.0, 1.0).astype(np.float32)
