"""Tests for backend.app.stages.io_fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app.stages.io_fits import load_fits, save_preview_png

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"


def _sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".fit", ".fits", ".fts"})


SAMPLE_FILES = _sample_files()


@pytest.mark.skipif(not SAMPLE_FILES, reason="no sample FITS files in samples/")
@pytest.mark.parametrize("sample", SAMPLE_FILES, ids=lambda p: p.name)
def test_load_fits_returns_normalized_rgb(sample: Path) -> None:
    arr = load_fits(sample)

    assert isinstance(arr, np.ndarray), "load_fits must return a numpy array"
    assert arr.ndim == 3, f"expected 3D output, got shape {arr.shape}"
    assert arr.shape[-1] == 3, f"expected 3 channels, got shape {arr.shape}"
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"
    assert float(arr.min()) >= 0.0, "values must be >= 0"
    assert float(arr.max()) <= 1.0, "values must be <= 1"


@pytest.mark.skipif(not SAMPLE_FILES, reason="no sample FITS files in samples/")
def test_save_preview_png_writes_file(tmp_path: Path) -> None:
    arr = load_fits(SAMPLE_FILES[0])
    out = tmp_path / "preview.png"
    save_preview_png(arr, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_preview_png_rejects_non_rgb(tmp_path: Path) -> None:
    bad = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        save_preview_png(bad, tmp_path / "x.png")
