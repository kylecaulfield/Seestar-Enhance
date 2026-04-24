"""Tests for Phase 3 SPCC helpers.

All tests use synthetic inputs — hand-placed stars in a 200x200 blank
image and a tiny mock catalog — so no network, no bundled Parquet,
and no real Seestar FITS is required. These tests pin the contract
the Phase 4 fit will rely on.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.stages.spcc import (
    _catalog_in_fov,
    _cross_match,
    _detect_bright_stars,
    _measure_star_rgb,
)
from astropy.wcs import WCS


def _make_image_with_stars(
    star_yx: list[tuple[int, int]],
    star_rgb: list[tuple[float, float, float]] | None = None,
    size: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Blank-sky image with a Gaussian-ish "star" at each coordinate."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.02, 0.04, size=(size, size, 3)).astype(np.float32)
    if star_rgb is None:
        star_rgb = [(0.8, 0.8, 0.8)] * len(star_yx)
    # Stamp a 5x5 cross that peaks at the centre — enough for the
    # maximum-filter detector without needing scipy's gaussian.
    for (y, x), (r, g, b) in zip(star_yx, star_rgb, strict=False):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                w = max(0.0, 1.0 - (abs(dy) + abs(dx)) * 0.25)
                yy, xx = y + dy, x + dx
                if 0 <= yy < size and 0 <= xx < size:
                    img[yy, xx, 0] = max(img[yy, xx, 0], r * w)
                    img[yy, xx, 1] = max(img[yy, xx, 1], g * w)
                    img[yy, xx, 2] = max(img[yy, xx, 2], b * w)
    return np.clip(img, 0.0, 1.0)


def _make_simple_wcs(
    image_shape: tuple[int, int],
    centre_radec: tuple[float, float] = (180.0, 0.0),
    arcsec_per_pix: float = 3.6,  # roughly Seestar
) -> WCS:
    """Construct a tiny tangent-plane WCS centred on `centre_radec`."""
    h, w = image_shape
    w_obj = WCS(naxis=2)
    w_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_obj.wcs.crval = [centre_radec[0], centre_radec[1]]
    w_obj.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    deg_per_pix = arcsec_per_pix / 3600.0
    w_obj.wcs.cd = np.array([[-deg_per_pix, 0.0], [0.0, deg_per_pix]], dtype=np.float64)
    return w_obj


# ---------- _detect_bright_stars ----------


def test_detect_finds_planted_stars() -> None:
    planted = [(40, 50), (100, 120), (160, 30)]
    img = _make_image_with_stars(planted)
    found = _detect_bright_stars(img, n=10)
    assert found.shape[1] == 2
    # Each planted star should be detected within 2 px — the detector
    # may pick any of the 5 cross pixels as "the" local max.
    for py, px in planted:
        distances = np.hypot(found[:, 0] - py, found[:, 1] - px)
        assert distances.min() < 2.0, f"planted star at ({py},{px}) not found"


def test_detect_respects_n_limit() -> None:
    # Three real stars but ask for only two; returns brightest first.
    img = _make_image_with_stars(
        [(40, 50), (100, 120), (160, 30)],
        star_rgb=[(0.3, 0.3, 0.3), (0.9, 0.9, 0.9), (0.6, 0.6, 0.6)],
    )
    found = _detect_bright_stars(img, n=2)
    assert found.shape[0] == 2
    # Brightest planted star was at (100, 120) at 0.9 intensity.
    ys, xs = found[:, 0], found[:, 1]
    assert np.any((np.abs(ys - 100) <= 2) & (np.abs(xs - 120) <= 2))


def test_detect_rejects_bad_args() -> None:
    img = np.zeros((50, 50, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        _detect_bright_stars(img, n=0)
    with pytest.raises(ValueError):
        _detect_bright_stars(np.zeros((50, 50), dtype=np.float32))


# ---------- _catalog_in_fov ----------


def test_fov_keeps_in_frame_and_drops_far_catalog() -> None:
    wcs = _make_simple_wcs((200, 200), centre_radec=(180.0, 0.0))
    # Catalog: one star at the frame centre, one way off the sky,
    # one just inside the frame, one just outside.
    cat = {
        "ra": np.array([180.0, 10.0, 180.05, 181.0], dtype=np.float64),
        "dec": np.array([0.0, 45.0, 0.0, 0.0], dtype=np.float64),
        "phot_g_mean_mag": np.array([8.0, 10.0, 9.0, 7.0], dtype=np.float32),
    }
    kept = _catalog_in_fov(cat, wcs, (200, 200), margin_deg=0.01)
    # The first two RA values (180.0 and 10.0) — 180.0 is at centre
    # (always kept), 10.0 is way off (dropped). 181.0 is ~1 deg off
    # which exceeds Seestar-sized FOV.
    kept_ras = set(np.round(kept["ra"], 3).tolist())
    assert 180.0 in kept_ras
    assert 10.0 not in kept_ras
    assert 181.0 not in kept_ras


def test_fov_empty_catalog() -> None:
    wcs = _make_simple_wcs((200, 200))
    cat = {
        "ra": np.array([], dtype=np.float64),
        "dec": np.array([], dtype=np.float64),
    }
    kept = _catalog_in_fov(cat, wcs, (200, 200))
    assert kept["ra"].size == 0
    assert kept["dec"].size == 0


def test_fov_rejects_missing_columns() -> None:
    wcs = _make_simple_wcs((200, 200))
    with pytest.raises(ValueError):
        _catalog_in_fov({"dec": np.array([0.0])}, wcs, (200, 200))


# ---------- _cross_match ----------


def test_cross_match_pairs_nearby() -> None:
    img = np.array([[180.0, 0.0], [180.001, 0.001]], dtype=np.float64)
    cat = np.array([[180.000, 0.000], [45.0, -30.0], [180.002, 0.001]], dtype=np.float64)
    img_idx, cat_idx = _cross_match(img, cat, tol_arcsec=5.0)
    # Both image stars have a catalogue neighbour within 5 arcsec.
    assert img_idx.size == 2
    # Pairings: img[0] -> cat[0], img[1] -> cat[2] (or cat[0] if
    # tolerance catches both; we don't enforce which).
    assert 0 in img_idx and 1 in img_idx
    # The non-matching catalogue entry (45°, -30°) should never appear.
    assert 1 not in cat_idx


def test_cross_match_rejects_far_outliers() -> None:
    img = np.array([[180.0, 0.0]], dtype=np.float64)
    cat = np.array([[0.0, 30.0]], dtype=np.float64)  # half the sky away
    img_idx, cat_idx = _cross_match(img, cat, tol_arcsec=5.0)
    assert img_idx.size == 0
    assert cat_idx.size == 0


def test_cross_match_empty_inputs() -> None:
    empty = np.empty((0, 2), dtype=np.float64)
    a, b = _cross_match(empty, np.array([[1.0, 2.0]]), tol_arcsec=5.0)
    assert a.size == 0 and b.size == 0


# ---------- _measure_star_rgb ----------


def test_measure_star_rgb_recovers_hue() -> None:
    img = _make_image_with_stars(
        [(100, 100), (40, 40)],
        star_rgb=[(0.9, 0.2, 0.2), (0.1, 0.1, 0.9)],
    )
    stars = np.array([[100, 100], [40, 40]], dtype=np.int64)
    rgb = _measure_star_rgb(img, stars, aperture_r=3, annulus_inner=5, annulus_outer=8)
    assert rgb.shape == (2, 3)
    # First star was red-dominant; second was blue-dominant. The
    # background-subtracted fluxes should preserve that.
    assert rgb[0, 0] > rgb[0, 1] and rgb[0, 0] > rgb[0, 2]
    assert rgb[1, 2] > rgb[1, 0] and rgb[1, 2] > rgb[1, 1]


def test_measure_star_rgb_zeroes_edge_stars() -> None:
    img = _make_image_with_stars([(100, 100)])
    stars = np.array([[2, 2], [100, 100]], dtype=np.int64)
    rgb = _measure_star_rgb(img, stars, aperture_r=3, annulus_inner=5, annulus_outer=8)
    # The (2, 2) star can't fit an annulus; its row is zeroed. The
    # centred star gets a non-zero reading.
    assert np.all(rgb[0] == 0.0)
    assert np.any(rgb[1] > 0.0)


def test_measure_rejects_bad_radii() -> None:
    img = np.zeros((50, 50, 3), dtype=np.float32)
    stars = np.array([[25, 25]], dtype=np.int64)
    with pytest.raises(ValueError):
        _measure_star_rgb(img, stars, aperture_r=3, annulus_inner=3, annulus_outer=5)
    with pytest.raises(ValueError):
        _measure_star_rgb(img, stars, aperture_r=5, annulus_inner=3, annulus_outer=8)


def test_measure_empty_star_list() -> None:
    img = np.zeros((50, 50, 3), dtype=np.float32)
    out = _measure_star_rgb(img, np.empty((0, 2), dtype=np.int64))
    assert out.shape == (0, 3)
