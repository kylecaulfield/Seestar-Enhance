"""Tests for the v1.5 pipeline improvements:

  - stages/crop.py: rectangular crop
  - stages/bm3d_denoise.py: downsample_factor fast path
  - stages/color.py: Mahalanobis WB + star-color protection
  - stages/background.py: caller-supplied sky_mask
  - stages/curves.py: star_preserve_percentile
  - stages/stretch.py: "auto" black_percentile and stretch
"""
from __future__ import annotations

import numpy as np
import pytest

from app.stages import background, bm3d_denoise, color, crop, curves, stretch


@pytest.fixture
def rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.15, size=(96, 96, 3)).astype(np.float32)
    # Add a coloured "star" at (30, 30) and an extended "nebula" blob.
    base[30, 30] = np.array([1.0, 0.6, 0.4], dtype=np.float32)
    yy, xx = np.mgrid[:96, :96]
    blob = np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / 100.0) * 0.25
    base[..., 0] += blob.astype(np.float32)
    return np.clip(base, 0.0, 1.0)


def test_crop_basic(rgb: np.ndarray) -> None:
    out = crop.process(rgb, top=10, left=5, bottom=40, right=50)
    assert out.shape == (30, 45, 3)
    # Same dtype, same data values
    assert out.dtype == rgb.dtype
    np.testing.assert_array_equal(out, rgb[10:40, 5:50, :])


def test_crop_negative_indexing(rgb: np.ndarray) -> None:
    out = crop.process(rgb, top=0, left=0, bottom=-16, right=-8)
    assert out.shape == (80, 88, 3)


def test_crop_rejects_bad_box(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        crop.process(rgb, top=50, left=0, bottom=10, right=96)
    with pytest.raises(ValueError):
        crop.process(rgb, top=0, left=0, bottom=0, right=10)
    with pytest.raises(ValueError):
        crop.process(np.zeros((10, 10), dtype=np.float32))


def test_bm3d_downsample_matches_shape(rgb: np.ndarray) -> None:
    # We can't check exact values against full-res BM3D; verify the
    # downsample path produces an output of the expected shape in [0, 1].
    out = bm3d_denoise.process(rgb, sigma=0.01, downsample_factor=2)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_color_mahalanobis_preserves_star_hue(rgb: np.ndarray) -> None:
    out = color.process(rgb, mahalanobis_wb=True, star_protect_percentile=99.0)
    # The bright coloured "star" at (30, 30) should still be dominantly red.
    r, g, b = out[30, 30]
    assert r > g and r > b


def test_color_star_protect_keeps_bright_pixel(rgb: np.ndarray) -> None:
    # With star protect DISABLED the WB can over-correct the coloured star.
    off = color.process(rgb, star_protect_percentile=None)
    on = color.process(rgb, star_protect_percentile=99.0)
    r_off = float(off[30, 30, 0])
    r_on = float(on[30, 30, 0])
    # star_protect_percentile should keep the star *at least as red* as
    # the unprotected version.
    assert r_on >= r_off - 1e-6


def test_background_honors_sky_mask(rgb: np.ndarray) -> None:
    # Build a mask that excludes the nebula blob; background fit should
    # still produce something sensible without the blob biasing it.
    luma = rgb.mean(axis=-1)
    sky = luma < float(np.percentile(luma, 60.0))
    out = background.process(rgb, grid=8, sky_mask=sky)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_background_rejects_bad_mask(rgb: np.ndarray) -> None:
    bad = np.zeros(rgb.shape[:2], dtype=np.uint8)
    with pytest.raises(ValueError):
        background.process(rgb, sky_mask=bad)
    with pytest.raises(ValueError):
        background.process(rgb, sky_mask=np.zeros((10, 10), dtype=bool))


def test_curves_star_preserve_tapers_saturation(rgb: np.ndarray) -> None:
    # Compare saturation boost on the bright star pixel with and without
    # star preservation. With preservation ON, the star should show a
    # smaller saturation change.
    orig = rgb[30, 30]
    off = curves.process(rgb, saturation=2.0, star_preserve_percentile=None)[30, 30]
    on = curves.process(rgb, saturation=2.0, star_preserve_percentile=99.0)[30, 30]

    def sat(px: np.ndarray) -> float:
        m = float(px.max())
        return 0.0 if m <= 0 else (m - float(px.min())) / m

    # Saturation on the star should be MORE affected in off (unprotected).
    delta_off = abs(sat(off) - sat(orig))
    delta_on = abs(sat(on) - sat(orig))
    assert delta_on <= delta_off + 1e-6


def test_stretch_auto_black_and_factor(rgb: np.ndarray) -> None:
    # Auto mode should produce a valid output without needing explicit params.
    out = stretch.process(rgb, black_percentile="auto", stretch="auto")
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # The brightest point of the image should still be the brightest after stretch.
    y, x = np.unravel_index(rgb.mean(axis=-1).argmax(), rgb.shape[:2])
    out_max_y, out_max_x = np.unravel_index(out.mean(axis=-1).argmax(), out.shape[:2])
    # Allow a small tolerance — arcsinh can shift the peak within a pixel.
    assert abs(int(out_max_y) - int(y)) <= 2
    assert abs(int(out_max_x) - int(x)) <= 2
