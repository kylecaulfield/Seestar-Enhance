"""Tests for classical star/starless separation."""
from __future__ import annotations

import numpy as np
import pytest

from app.stages import stars


def _synthetic_scene(size: int = 128) -> np.ndarray:
    """A smooth nebula background + a few point-source 'stars'."""
    rng = np.random.default_rng(42)
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Smooth nebula: large Gaussian blob
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-((yy - size // 2) ** 2 + (xx - size // 2) ** 2) / (size * 4))
    img[..., 0] = blob * 0.2  # red nebula
    img[..., 2] = blob * 0.1
    # Point-source stars: random bright 1-2 pixel peaks
    for _ in range(20):
        y = rng.integers(5, size - 5)
        x = rng.integers(5, size - 5)
        img[y, x] = 1.0
    return img


def test_split_shapes_and_bounds():
    img = _synthetic_scene()
    stars_only, starless = stars.process(img, radius=5)
    assert stars_only.shape == img.shape
    assert starless.shape == img.shape
    assert stars_only.min() >= 0.0 and stars_only.max() <= 1.0
    assert starless.min() >= 0.0 and starless.max() <= 1.0
    assert stars_only.dtype == np.float32
    assert starless.dtype == np.float32


def test_stars_captures_point_sources():
    """Isolated bright pixels should end up in the stars channel."""
    img = _synthetic_scene()
    stars_only, starless = stars.process(img, radius=5)
    # The 20 planted stars were at value 1.0; their total flux should
    # mostly be in stars_only, not starless.
    total_flux = img.sum()
    star_flux = stars_only.sum()
    # At least 2% of the frame's total flux should be in stars (these
    # 20 pixels × 3 channels contribute meaningfully even in a nebula).
    assert star_flux / max(total_flux, 1e-6) > 0.02


def test_starless_preserves_smooth_structure():
    """The nebula blob should survive the median filter almost intact."""
    img = _synthetic_scene()
    _, starless = stars.process(img, radius=5)
    # Nebula signal is in the red channel centered on the middle.
    # Sample both the input and starless at the centre; they should agree
    # to within a few percent because the blob is much wider than 5 px.
    cy = cx = img.shape[0] // 2
    orig_red = img[cy - 8 : cy + 8, cx - 8 : cx + 8, 0].mean()
    star_red = starless[cy - 8 : cy + 8, cx - 8 : cx + 8, 0].mean()
    assert abs(orig_red - star_red) / max(orig_red, 1e-6) < 0.05


def test_recombine_reconstructs_input_approximately():
    img = _synthetic_scene()
    stars_only, starless = stars.process(img, radius=5)
    reconstructed = stars.recombine(stars_only, starless)
    # Screen blend is non-linear, so we allow a few percent of slack.
    mean_error = float(np.mean(np.abs(reconstructed - img)))
    assert mean_error < 0.02


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        stars.process(np.zeros((64, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        stars.process(np.zeros((64, 64, 4), dtype=np.float32))


def test_rejects_bad_radius():
    img = _synthetic_scene()
    with pytest.raises(ValueError):
        stars.process(img, radius=0)


def test_recombine_shape_check():
    a = np.zeros((32, 32, 3), dtype=np.float32)
    b = np.zeros((16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        stars.recombine(a, b)
