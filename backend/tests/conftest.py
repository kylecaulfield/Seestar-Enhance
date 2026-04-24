"""Shared pytest fixtures for stage tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    h, w = 96, 128

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing="ij",
    )
    gradient = 0.15 + 0.05 * xx + 0.03 * yy

    nebula = 0.25 * np.exp(-(xx**2 + yy**2) / 0.3)
    base = gradient + nebula
    noise = rng.normal(0.0, 0.01, size=(h, w)).astype(np.float32)

    r = np.clip(base + noise, 0.0, 1.0)
    g = np.clip(base * 0.9 + noise, 0.0, 1.0)
    b = np.clip(base * 1.1 + noise, 0.0, 1.0)

    img = np.stack([r, g, b], axis=-1).astype(np.float32)

    ys = rng.integers(0, h, size=25)
    xs = rng.integers(0, w, size=25)
    img[ys, xs, :] = 1.0
    return img
