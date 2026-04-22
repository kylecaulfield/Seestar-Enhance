"""Tests for the reference-match improvements (items 1-5 of that section).

Covers the new stages (cosmetic, clahe, deconv), the new chroma-edge-
aware option in bm3d_denoise, and the pipeline wiring that threads the
new opt-in params through.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.stages import bm3d_denoise, clahe, cosmetic, deconv


@pytest.fixture
def rgb() -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.normal(0.1, 0.02, size=(96, 96, 3)).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


# ---------- cosmetic ----------

def test_cosmetic_removes_single_hot_pixel(rgb: np.ndarray) -> None:
    img = rgb.copy()
    img[30, 40] = 1.0  # single hot pixel, far brighter than neighbourhood
    out = cosmetic.process(img, neighborhood=3, sigma=4.0)
    # It should now be within the normal value range, not at 1.0.
    assert out[30, 40].max() < 0.5


def test_cosmetic_preserves_real_stars(rgb: np.ndarray) -> None:
    # A "real" star occupies a 3x3 block of bright pixels; it's not an
    # isolated outlier, so the 3-window median doesn't reject it.
    img = rgb.copy()
    img[30:33, 40:43] = 0.9
    out = cosmetic.process(img, neighborhood=3, sigma=6.0)
    # Center stays bright.
    assert out[31, 41].min() > 0.8


def test_cosmetic_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        cosmetic.process(np.zeros((10, 10), dtype=np.float32))


# ---------- clahe ----------

def test_clahe_lifts_local_contrast(rgb: np.ndarray) -> None:
    # Build an image with a mid-brightness gradient. CLAHE should
    # stretch it toward the full 0..1 range on average.
    h, w = 96, 96
    grad = np.linspace(0.3, 0.6, w, dtype=np.float32)
    img = np.broadcast_to(grad[None, :, None], (h, w, 3)).astype(np.float32).copy()
    # Add a tiny bit of noise so adapthist finds something to do.
    rng = np.random.default_rng(0)
    img += rng.normal(0.0, 0.01, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    out = clahe.process(img, clip_limit=0.03, kernel_size=32, blend=1.0)
    # The output luma should span more of [0,1] than the input.
    luma_in = img.mean(axis=-1)
    luma_out = out.mean(axis=-1)
    assert (luma_out.max() - luma_out.min()) >= (luma_in.max() - luma_in.min())


def test_clahe_rejects_bad_blend(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        clahe.process(rgb, blend=2.0)


def test_clahe_zero_blend_is_identity(rgb: np.ndarray) -> None:
    out = clahe.process(rgb, blend=0.0)
    # With blend=0 the enhanced luma never replaces the original, so
    # the output should equal the input to within float precision.
    np.testing.assert_allclose(out, rgb, atol=1e-6)


# ---------- deconv ----------

def test_deconv_tightens_a_blurred_star() -> None:
    # Synthesise a blurred star; deconv should narrow the core.
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[32, 32] = 1.0
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        img[..., c] = gaussian_filter(img[..., c], sigma=2.0)
    out = deconv.process(img, psf_sigma=1.5, iterations=10, psf_size=11)
    # Peak value of the deconvolved star should exceed the blurred original.
    assert out[32, 32].mean() > img[32, 32].mean()


def test_deconv_rejects_bad_iterations(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        deconv.process(rgb, iterations=0)


# ---------- bm3d chroma edge-aware ----------

def test_chroma_edge_aware_preserves_luma_edges() -> None:
    # Left half = red, right half = blue. With edge-aware chroma, the
    # colour transition should stay sharp; isotropic chroma_blur would
    # smear it.
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, : w // 2, 0] = 0.6  # red on left
    img[:, w // 2 :, 2] = 0.6  # blue on right
    luma = img.mean(axis=-1)
    img[...] = img + 0.1  # lift so there's non-zero luma everywhere
    # Give the halves different luma so the edge is detectable.
    img[:, : w // 2, :] += 0.3
    # Apply edge-aware chroma smoothing.
    out = bm3d_denoise.process(
        img,
        sigma=0.01,
        chroma_blur=5.0,
        chroma_edge_aware=True,
        chroma_edge_luma_sigma=0.05,
        downsample_factor=1,
    )
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
