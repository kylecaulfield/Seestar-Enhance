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


# ---------- security: FITS-bomb defence ----------
#
# `_validate_fits_header` runs before astropy ever touches the file. The
# below tests build hand-crafted FITS bytes that astropy would happily
# parse but that should be rejected at the gate.


def _fits_block(cards: list[str]) -> bytes:
    """Pad ``cards`` to a 2880-byte FITS header block. Each card must be
    80 characters; we pad shorter ones with spaces and add an END card."""
    out = bytearray()
    for card in cards:
        if len(card) > 80:
            raise ValueError(f"card too long: {card!r}")
        out.extend(card.ljust(80).encode("ascii"))
    out.extend(b"END" + b" " * 77)
    while len(out) % 2880:
        out.extend(b" ")
    return bytes(out)


def test_validate_rejects_oversized_primary_naxis(tmp_path: Path) -> None:
    """Primary HDU declaring multi-GB image data is rejected."""
    from app.stages.io_fits import _validate_fits_header

    block = _fits_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                90000",
            "NAXIS2  =                90000",
        ]
    )
    p = tmp_path / "bomb.fits"
    p.write_bytes(block)
    with pytest.raises(ValueError, match="exceeds plausible|max allowed"):
        _validate_fits_header(p)


def test_validate_rejects_compressed_image_bomb(tmp_path: Path) -> None:
    """Tile-compressed extension declaring a multi-GB decompressed
    image is rejected, even when the on-disk BINTABLE wrapper is tiny.

    This is the security-critical case the new extension-HDU scan
    exists for — `astropy.io.fits` decompresses on `.data` access and
    will allocate the full ZNAXIS1*ZNAXIS2*|ZBITPIX| array, OOMing the
    worker. Pre-validator should reject before astropy sees it.
    """
    from app.stages.io_fits import _validate_fits_header

    primary = _fits_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ]
    )
    # BINTABLE extension carrying a tile-compressed image. The on-disk
    # data is small (4096-byte BINTABLE rows) but ZNAXIS1*ZNAXIS2 is
    # 4 GB decompressed, well past the 2 GB cap.
    ext = _fits_block(
        [
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                 4096",
            "NAXIS2  =                    8",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "TFIELDS =                    1",
            "ZIMAGE  =                    T",
            "ZBITPIX =                   16",
            "ZNAXIS  =                    2",
            "ZNAXIS1 =                65536",
            "ZNAXIS2 =                65536",
        ]
    )
    p = tmp_path / "ziplomb.fits"
    p.write_bytes(primary + ext)
    with pytest.raises(ValueError, match="exceeds plausible|max allowed"):
        _validate_fits_header(p)


def test_validate_accepts_normal_compressed_image(tmp_path: Path) -> None:
    """Sanity check: a plausibly-sized tile-compressed image is fine."""
    from app.stages.io_fits import _validate_fits_header

    primary = _fits_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ]
    )
    # 1080×1920 16-bit decompressed (~4 MB) is well inside the cap.
    ext = _fits_block(
        [
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                 4096",
            "NAXIS2  =                    1",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "TFIELDS =                    1",
            "ZIMAGE  =                    T",
            "ZBITPIX =                   16",
            "ZNAXIS  =                    2",
            "ZNAXIS1 =                 1920",
            "ZNAXIS2 =                 1080",
        ]
    )
    p = tmp_path / "ok.fits"
    p.write_bytes(primary + ext)
    # Should not raise.
    _validate_fits_header(p)
