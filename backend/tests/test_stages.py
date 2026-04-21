"""Tests for individual pipeline stages."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import png
import pytest

from app.stages import (
    background,
    bm3d_denoise,
    color,
    curves,
    export,
    ml_denoise,
    sharpen,
    stars,
    stretch,
)


def _assert_rgb01(arr: np.ndarray, shape: tuple[int, int, int]) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.shape == shape
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


def test_background_shape_range(sample_rgb: np.ndarray) -> None:
    out = background.process(sample_rgb)
    _assert_rgb01(out, sample_rgb.shape)


def test_background_reduces_gradient(sample_rgb: np.ndarray) -> None:
    out = background.process(sample_rgb)
    before = float(sample_rgb[..., 0].std())
    after = float(out[..., 0].std())
    assert after <= before + 1e-6


def test_color_shape_range(sample_rgb: np.ndarray) -> None:
    out = color.process(sample_rgb)
    _assert_rgb01(out, sample_rgb.shape)


def test_color_balances_mid_medians(sample_rgb: np.ndarray) -> None:
    out = color.process(sample_rgb)
    luma = out.mean(axis=-1)
    lo, hi = np.percentile(luma, [40, 80])
    mid = (luma >= lo) & (luma <= hi)
    meds = np.array([float(np.median(out[..., c][mid])) for c in range(3)])
    spread_before = (
        np.array(
            [float(np.median(sample_rgb[..., c][mid])) for c in range(3)]
        ).ptp()
    )
    assert meds.ptp() <= spread_before + 1e-3


def test_stretch_shape_range(sample_rgb: np.ndarray) -> None:
    out = stretch.process(sample_rgb)
    _assert_rgb01(out, sample_rgb.shape)


def test_stretch_brightens_midtones(sample_rgb: np.ndarray) -> None:
    out = stretch.process(sample_rgb)
    assert float(out.mean()) >= float(sample_rgb.mean())


def test_sharpen_shape_range(sample_rgb: np.ndarray) -> None:
    out = sharpen.process(sample_rgb)
    _assert_rgb01(out, sample_rgb.shape)


def test_curves_shape_range(sample_rgb: np.ndarray) -> None:
    out = curves.process(sample_rgb)
    _assert_rgb01(out, sample_rgb.shape)


def test_bm3d_denoise_shape_range(sample_rgb: np.ndarray) -> None:
    out = bm3d_denoise.process(sample_rgb, sigma=0.02)
    _assert_rgb01(out, sample_rgb.shape)


def test_bm3d_denoise_reduces_noise(sample_rgb: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    noisy = np.clip(
        sample_rgb + rng.normal(0.0, 0.04, size=sample_rgb.shape).astype(np.float32),
        0.0,
        1.0,
    )
    denoised = bm3d_denoise.process(noisy, sigma=0.04)
    noisy_res = float((noisy - sample_rgb).std())
    denoised_res = float((denoised - sample_rgb).std())
    assert denoised_res < noisy_res


def test_stars_splits_into_two_arrays(sample_rgb: np.ndarray) -> None:
    # v2: classical star/starless split. Detailed behaviour is in test_stars.py.
    stars_only, starless = stars.process(sample_rgb, radius=5)
    assert stars_only.shape == sample_rgb.shape
    assert starless.shape == sample_rgb.shape


def test_ml_denoise_stub_raises(sample_rgb: np.ndarray) -> None:
    with pytest.raises(NotImplementedError):
        ml_denoise.process(sample_rgb)


def test_export_writes_16bit_png(sample_rgb: np.ndarray, tmp_path: Path) -> None:
    out_path = tmp_path / "out.png"
    export.process(sample_rgb, out_path)
    assert out_path.is_file()

    reader = png.Reader(filename=str(out_path))
    width, height, rows, info = reader.read()
    assert width == sample_rgb.shape[1]
    assert height == sample_rgb.shape[0]
    assert info["bitdepth"] == 16
    assert info["planes"] == 3
    arr = np.array(list(rows)).reshape(height, width, 3)
    assert arr.dtype == np.uint16


def test_export_rejects_non_rgb(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export.process(np.zeros((4, 4), dtype=np.float32), tmp_path / "x.png")
