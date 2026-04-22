"""Tests for the image content classifier."""
from __future__ import annotations

import numpy as np
import pytest

from app.stages.classify import _metrics, classify


def _rng_sky(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.01, 0.002, size=(h, w)).astype(np.float32)
    base = np.clip(base, 0.0, 1.0)
    return np.stack([base] * 3, axis=-1)


def _synthetic_nebula() -> np.ndarray:
    img = _rng_sky(seed=1)
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, img.shape[0], dtype=np.float32),
        np.linspace(-1, 1, img.shape[1], dtype=np.float32),
        indexing="ij",
    )
    # Broad diffuse emission that covers most of the frame.
    blob = 0.12 * np.exp(-(xx**2 + yy**2) / 1.5)
    for c in range(3):
        img[..., c] = np.clip(img[..., c] + blob, 0.0, 1.0)
    return img


def _synthetic_galaxy() -> np.ndarray:
    img = _rng_sky(seed=2)
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, img.shape[0], dtype=np.float32),
        np.linspace(-1, 1, img.shape[1], dtype=np.float32),
        indexing="ij",
    )
    # Tight bright core with a small halo (galaxy disk).
    core = 0.8 * np.exp(-(xx**2 + yy**2) / 0.02)
    for c in range(3):
        img[..., c] = np.clip(img[..., c] + core, 0.0, 1.0)
    # A sparse sprinkle of foreground stars — well under the cluster
    # density threshold.
    rng = np.random.default_rng(20)
    ys = rng.integers(0, img.shape[0], size=20)
    xs = rng.integers(0, img.shape[1], size=20)
    img[ys, xs, :] = 1.0
    return img


def _synthetic_cluster() -> np.ndarray:
    img = _rng_sky(seed=3)
    rng = np.random.default_rng(30)
    # Dense star field -> well above the cluster density threshold.
    ys = rng.integers(0, img.shape[0], size=2000)
    xs = rng.integers(0, img.shape[1], size=2000)
    img[ys, xs, :] = 1.0
    # Add a compact bright "core" region — the classifier requires a
    # visible dense region, not just point stars, to disambiguate a
    # real cluster from a faint-nebula filament in a dense starfield.
    h, w = img.shape[:2]
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing="ij",
    )
    core = 0.9 * np.exp(-(xx**2 + yy**2) / 0.01)
    for c in range(3):
        img[..., c] = np.clip(img[..., c] + core, 0.0, 1.0)
    return img


def test_metrics_keys() -> None:
    img = _rng_sky()
    m = _metrics(img)
    assert set(m.keys()) == {
        "non_star_median",
        "star_density_per_mpix",
        "largest_bright_fraction",
        "largest_bright_elongation",
    }


def test_classify_nebula_on_diffuse() -> None:
    # The synthetic nebula blob fills most of the frame, so it lands
    # in the "wide" sub-variant (largest_bright_fraction > 0.20).
    assert classify(_synthetic_nebula()) in ("nebula", "nebula_wide")


def test_classify_cluster_on_dense_stars() -> None:
    assert classify(_synthetic_cluster()) == "cluster"


def test_classify_galaxy_fallback() -> None:
    assert classify(_synthetic_galaxy()) == "galaxy"


def test_classify_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        _metrics(np.zeros((10, 10), dtype=np.float32))
