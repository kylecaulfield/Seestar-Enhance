"""FITS I/O for Seestar images.

Loads raw Bayer-pattern FITS files produced by the ZWO Seestar telescope,
detects the Bayer pattern from the header, debayers to RGB, and returns a
float32 array normalized to [0, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from astropy.io import fits
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

PathLike = Union[str, Path]

_VALID_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}

# Defence against "FITS bomb" uploads: a tiny file whose header declares
# enormous image dimensions. astropy.io.fits eagerly allocates a numpy
# array sized by NAXIS1 * NAXIS2 * |BITPIX|/8 before reading any data, so
# we pre-parse the primary header and reject files declaring more than
# this many bytes of image data. A Seestar S50 raw frame is ~2.5 MP *
# 2 bytes = ~5 MB; 2 GiB leaves plenty of headroom for stacked output.
_MAX_DECLARED_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


def _validate_fits_header(path: PathLike) -> None:
    """Pre-parse the FITS primary header and reject obviously hostile files.

    Reads only the first header block(s) — never the data region — and
    walks the 80-char cards manually so we never hand a bomb to astropy.
    Raises ValueError on anything suspicious.
    """
    block_size = 2880
    card_size = 80
    values: dict[str, str] = {}

    with open(path, "rb") as f:
        # A FITS header can span multiple 2880-byte blocks; read up to 5
        # blocks (14400 bytes, room for ~180 cards) before giving up.
        for _ in range(5):
            block = f.read(block_size)
            if len(block) < block_size:
                raise ValueError("FITS header truncated")
            end_marker_found = False
            for i in range(0, block_size, card_size):
                card = block[i : i + card_size].decode("ascii", errors="replace")
                key = card[:8].strip()
                if key == "END":
                    end_marker_found = True
                    break
                if "=" in card[:10]:
                    value_part = card[10:].split("/", 1)[0].strip()
                    values[key] = value_part.strip("' ")
            if end_marker_found:
                break
        else:
            raise ValueError("FITS header did not terminate within 5 blocks")

    def _as_int(name: str, default: int = 0) -> int:
        try:
            return int(values.get(name, default))
        except (TypeError, ValueError):
            raise ValueError(f"FITS header {name} is not an integer") from None

    bitpix = _as_int("BITPIX")
    naxis = _as_int("NAXIS")
    if naxis < 0 or naxis > 4:
        raise ValueError(f"FITS NAXIS={naxis} outside supported range")
    if bitpix not in (8, 16, 32, 64, -32, -64):
        raise ValueError(f"FITS BITPIX={bitpix} not recognised")

    total = 1
    for axis in range(1, naxis + 1):
        dim = _as_int(f"NAXIS{axis}")
        if dim <= 0:
            return  # NAXIS=0 or missing size ⇒ no image data, safe.
        if dim > 100_000:
            raise ValueError(f"FITS NAXIS{axis}={dim} exceeds plausible image size")
        total *= dim
    declared_bytes = total * (abs(bitpix) // 8)
    if declared_bytes > _MAX_DECLARED_IMAGE_BYTES:
        raise ValueError(
            f"FITS declares {declared_bytes} bytes of image data; "
            f"max allowed is {_MAX_DECLARED_IMAGE_BYTES}"
        )


def _detect_bayer_pattern(header: fits.Header) -> str:
    """Read the Bayer pattern from a FITS header.

    Seestar files typically use the BAYERPAT keyword. We also accept
    COLORTYP / CFAPAT as fallbacks. Defaults to RGGB if unspecified, which
    matches the Seestar S50 native sensor.
    """
    for key in ("BAYERPAT", "BAYRPAT", "COLORTYP", "CFAPAT"):
        value = header.get(key)
        if value:
            pattern = str(value).strip().upper()
            if pattern in _VALID_BAYER_PATTERNS:
                return pattern
    return "RGGB"


def _to_float01(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Normalize raw integer data to float32 in [0, 1].

    Uses BITPIX from the header to determine the dynamic range. Falls back
    to the data's own dtype when BITPIX is missing or non-integer.
    """
    arr = np.asarray(data)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_value = float(info.max)
    elif np.issubdtype(arr.dtype, np.floating):
        finite_max = float(np.nanmax(arr)) if arr.size else 1.0
        max_value = finite_max if finite_max > 0 else 1.0
    else:
        max_value = 1.0

    bitpix = header.get("BITPIX")
    if isinstance(bitpix, (int, np.integer)) and bitpix > 0:
        max_value = float((1 << int(bitpix)) - 1)

    out = arr.astype(np.float32, copy=False) / np.float32(max_value)
    return np.clip(out, 0.0, 1.0)


def load_fits(path: PathLike) -> np.ndarray:
    """Load a Seestar FITS file and return a debayered RGB image.

    Parameters
    ----------
    path : str | Path
        Path to a Seestar-produced FITS file containing a 2D Bayer mosaic.

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 3) with values in [0, 1].
    """
    _validate_fits_header(path)
    with fits.open(str(path), memmap=False) as hdul:
        primary = hdul[0]
        data = primary.data
        header = primary.header

        if data is None:
            for hdu in hdul[1:]:
                if hdu.data is not None:
                    data = hdu.data
                    header = hdu.header
                    break

    if data is None:
        raise ValueError(f"FITS file contains no image data: {path}")

    if data.ndim == 3 and data.shape[0] in (3, 4):
        rgb = np.moveaxis(data[:3], 0, -1)
        return _to_float01(rgb, header)

    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return _to_float01(data[..., :3], header)

    if data.ndim != 2:
        raise ValueError(
            f"Unsupported FITS data shape {data.shape}; expected 2D Bayer mosaic"
        )

    pattern = _detect_bayer_pattern(header)
    mono = _to_float01(data, header)
    rgb = demosaicing_CFA_Bayer_Malvar2004(mono, pattern=pattern)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    return rgb


def save_preview_png(arr: np.ndarray, path: PathLike) -> None:
    """Save a quick 8-bit PNG preview using a simple log stretch.

    Intended for debugging only. Applies log1p to compress dynamic range,
    then linearly remaps to [0, 255].
    """
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got shape {arr.shape}")

    data = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
    stretched = np.log1p(data * 1000.0)
    peak = float(stretched.max()) if stretched.size else 1.0
    if peak > 0:
        stretched = stretched / peak

    out = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(str(path))
