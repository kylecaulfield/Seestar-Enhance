"""End-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import png
import pytest
from astropy.io import fits

from app.pipeline import run


@pytest.fixture
def synthetic_fits(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    h, w = 128, 192
    mosaic = (rng.random((h, w)) * 30000 + 2000).astype(np.uint16)
    hdu = fits.PrimaryHDU(mosaic)
    hdu.header["BAYERPAT"] = "RGGB"
    path = tmp_path / "synth.fits"
    hdu.writeto(path)
    return path


def test_pipeline_end_to_end(synthetic_fits: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    result = run(synthetic_fits, out)

    assert out.is_file()
    assert result.ndim == 3 and result.shape[-1] == 3
    assert result.dtype == np.float32
    assert 0.0 <= float(result.min()) and float(result.max()) <= 1.0

    reader = png.Reader(filename=str(out))
    _, _, _, info = reader.read()
    assert info["bitdepth"] == 16
    assert info["planes"] == 3
