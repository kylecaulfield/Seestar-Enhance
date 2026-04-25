"""FITS I/O for Seestar images.

Loads raw Bayer-pattern FITS files produced by the ZWO Seestar telescope,
detects the Bayer pattern from the header, debayers to RGB, and returns a
float32 array normalized to [0, 1].

Two entry points are exposed:

    load_fits(path)            -> np.ndarray
    load_fits_with_wcs(path)   -> (np.ndarray, astropy.wcs.WCS | None)

`load_fits` stays around for existing callers that just want the pixels.
`load_fits_with_wcs` returns the debayered image plus, when the FITS
header carries a valid astrometric solution (Seestar firmware embeds
CTYPE/CRVAL/CRPIX/CD plus SIP distortion coefficients in every file),
a parsed `astropy.wcs.WCS` object that maps pixel ↔ RA/Dec. This is
what v2 SPCC (phase 4) will use to project the Gaia catalog onto the
image plane. When the header has no usable WCS, the second element is
`None` and downstream code is expected to skip the SPCC branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from PIL import Image

PathLike = str | Path

_VALID_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}

# Defence against "FITS bomb" uploads: a tiny file whose header declares
# enormous image dimensions. astropy.io.fits eagerly allocates a numpy
# array sized by NAXIS1 * NAXIS2 * |BITPIX|/8 before reading any data, so
# we pre-parse the primary header and reject files declaring more than
# this many bytes of image data. A Seestar S50 raw frame is ~2.5 MP *
# 2 bytes = ~5 MB; 2 GiB leaves plenty of headroom for stacked output.
_MAX_DECLARED_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


def _read_fits_header_blocks(f, max_blocks: int = 5) -> dict[str, str]:
    """Walk one FITS header (2880-byte blocks) and return its KEYWORD=VALUE
    cards as a string dict. Stops at the END card. Caller positions ``f``
    at the start of the header before calling.

    Reads at most ``max_blocks`` blocks (default 14400 bytes / ~180 cards)
    before giving up — long enough for any plausible Seestar header but
    short enough to bound work on a hostile input.
    """
    block_size = 2880
    card_size = 80
    values: dict[str, str] = {}

    for _ in range(max_blocks):
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
            return values
    raise ValueError("FITS header did not terminate within block limit")


def _check_declared_image_bytes(values: dict[str, str], hdu_label: str) -> None:
    """Inspect a parsed header dict and raise if the declared image data
    (or compressed-image decompressed size) exceeds the cap.

    Handles two patterns:

    * Plain image HDU: ``NAXIS`` + ``NAXISn`` + ``BITPIX`` describe raw
      pixel storage. Product is the on-disk size.
    * Tile-compressed image extension (``ZIMAGE = T``): the on-disk
      ``BINTABLE`` is small, but ``ZNAXISn`` + ``ZBITPIX`` describe the
      decompressed array. A "fits-bomb" frame can declare a 100 kB
      BINTABLE that decompresses to gigabytes; astropy will happily
      allocate the full array on ``.data`` access. The cap is the same
      for both forms.
    """

    def _as_int(name: str, default: int = 0) -> int:
        try:
            return int(values.get(name, default))
        except (TypeError, ValueError):
            raise ValueError(f"FITS header {name} is not an integer") from None

    # Compressed image takes precedence — if the HDU is a tile-compressed
    # image, the ZNAXIS*/ZBITPIX keywords describe the real allocation
    # size, not NAXIS/BITPIX (which describe the BINTABLE wrapper).
    is_compressed = values.get("ZIMAGE", "").upper().startswith("T") or (
        values.get("XTENSION", "").upper() == "BINTABLE" and "ZNAXIS" in values
    )

    if is_compressed:
        bitpix_key, naxis_key, axis_key_fmt = "ZBITPIX", "ZNAXIS", "ZNAXIS{}"
    else:
        bitpix_key, naxis_key, axis_key_fmt = "BITPIX", "NAXIS", "NAXIS{}"

    bitpix = _as_int(bitpix_key)
    naxis = _as_int(naxis_key)
    if naxis < 0 or naxis > 4:
        raise ValueError(f"FITS {hdu_label} {naxis_key}={naxis} outside supported range")
    if bitpix not in (8, 16, 32, 64, -32, -64):
        raise ValueError(f"FITS {hdu_label} {bitpix_key}={bitpix} not recognised")

    total = 1
    for axis in range(1, naxis + 1):
        dim = _as_int(axis_key_fmt.format(axis))
        if dim <= 0:
            return  # naxis=0 or missing size ⇒ no image data, safe.
        if dim > 100_000:
            raise ValueError(
                f"FITS {hdu_label} {axis_key_fmt.format(axis)}={dim} exceeds plausible image size"
            )
        total *= dim
    declared_bytes = total * (abs(bitpix) // 8)
    if declared_bytes > _MAX_DECLARED_IMAGE_BYTES:
        raise ValueError(
            f"FITS {hdu_label} declares {declared_bytes} bytes of "
            f"image data; max allowed is {_MAX_DECLARED_IMAGE_BYTES}"
        )


def _validate_fits_header(path: PathLike) -> None:
    """Pre-parse FITS headers and reject obviously hostile files.

    Walks both the primary HDU and any extension HDUs (up to a fixed
    cap) checking each for plausible image-data sizes. The compressed-
    image case (``ZIMAGE``) is the security-critical one: a small
    BINTABLE can declare a multi-GB decompressed array, and astropy
    allocates that array on ``.data`` access in `_read_fits_hdu`.

    Raises ``ValueError`` on anything suspicious; never reads the data
    region of any HDU. Does not call into astropy.
    """
    block_size = 2880
    max_extensions = 16  # plenty for any real Seestar/stacked frame.

    with open(path, "rb") as f:
        # Primary HDU
        primary = _read_fits_header_blocks(f)
        _check_declared_image_bytes(primary, hdu_label="primary HDU")
        primary_bitpix = abs(int(primary.get("BITPIX", 0) or 0))
        primary_data_bytes = 0
        primary_naxis = int(primary.get("NAXIS", 0) or 0)
        if primary_naxis > 0:
            n = 1
            for axis in range(1, primary_naxis + 1):
                dim = int(primary.get(f"NAXIS{axis}", 0) or 0)
                if dim <= 0:
                    n = 0
                    break
                n *= dim
            primary_data_bytes = n * (primary_bitpix // 8)

        # Skip past primary data region (zero-padded to 2880-byte blocks)
        # so the next header read picks up the first extension's header.
        if primary_data_bytes > 0:
            padded = ((primary_data_bytes + block_size - 1) // block_size) * block_size
            try:
                f.seek(padded, 1)
            except OSError as e:
                raise ValueError(f"FITS file truncated at primary data: {e}") from None

        for ext_idx in range(max_extensions):
            # Peek at the first 80 bytes; if we can't read a full card,
            # we're at EOF (no more extensions).
            head = f.read(80)
            if len(head) < 80:
                return
            f.seek(-80, 1)
            xtension = head[:8].decode("ascii", errors="replace").strip()
            if xtension not in ("XTENSION", ""):
                # Not a recognised extension start; bail rather than
                # try to interpret arbitrary bytes.
                return
            try:
                ext_values = _read_fits_header_blocks(f)
            except ValueError:
                # Truncated extension header — let astropy report it
                # instead of pre-rejecting on partial input.
                return
            _check_declared_image_bytes(ext_values, hdu_label=f"extension HDU {ext_idx}")

            # Skip the extension's data region too. For BINTABLEs this
            # is NAXIS1*NAXIS2 bytes (NAXIS1 already includes the row
            # width incl. compressed payload), plus PCOUNT for the
            # heap. We bound by the size cap and bail on overflow.
            try:
                ext_bitpix = abs(int(ext_values.get("BITPIX", 0) or 0))
                ext_naxis = int(ext_values.get("NAXIS", 0) or 0)
                ext_pcount = int(ext_values.get("PCOUNT", 0) or 0)
                ext_gcount = int(ext_values.get("GCOUNT", 1) or 1)
            except (TypeError, ValueError):
                return
            if ext_naxis <= 0:
                continue
            n = 1
            for axis in range(1, ext_naxis + 1):
                dim = int(ext_values.get(f"NAXIS{axis}", 0) or 0)
                if dim <= 0:
                    n = 0
                    break
                n *= dim
            ext_data_bytes = (n * (ext_bitpix // 8) + ext_pcount) * ext_gcount
            if ext_data_bytes < 0 or ext_data_bytes > _MAX_DECLARED_IMAGE_BYTES:
                raise ValueError(
                    f"FITS extension HDU {ext_idx} declares "
                    f"{ext_data_bytes} bytes; max allowed is "
                    f"{_MAX_DECLARED_IMAGE_BYTES}"
                )
            if ext_data_bytes > 0:
                padded = ((ext_data_bytes + block_size - 1) // block_size) * block_size
                try:
                    f.seek(padded, 1)
                except OSError as e:
                    raise ValueError(
                        f"FITS file truncated at extension {ext_idx} data: {e}"
                    ) from None


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


def _header_has_wcs(header: fits.Header) -> bool:
    """True if the header carries a usable astrometric solution.

    We require at minimum CTYPE1 / CTYPE2 plus the reference-point
    keywords (CRVAL1/2, CRPIX1/2). Seestar firmware writes all of
    these on every frame; a missing CTYPE is the signal that the
    plate-solve-at-capture step failed.
    """
    required = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
    return all(k in header for k in required)


def _parse_wcs(header: fits.Header) -> WCS | None:
    """Construct an astropy WCS from the header if it looks valid.

    Swallows the astropy parser's warnings/exceptions because a bad
    WCS is not a reason to fail the whole load — v1 callers that
    don't need the WCS should continue to work. When the parse fails
    we return None and the caller is expected to fall back gracefully.
    """
    if not _header_has_wcs(header):
        return None
    # Seestar FITS carry NAXIS=3 (H, W, 3) when debayered, but the WCS
    # itself only describes the 2 sky axes and has SIP distortion
    # coefficients which WCSLIB refuses to mix with >2 dims. Pass
    # `naxis=2` so astropy picks the sky projection and ignores the
    # degenerate 3rd axis.
    try:
        wcs = WCS(header, naxis=2)
    except Exception:  # noqa: BLE001 — astropy raises various types here
        return None
    # Degenerate WCS check: astropy returns a valid object for many
    # malformed headers (all-zero CD / CDELT). Treat those as no-WCS
    # rather than handing downstream code a projection that silently
    # returns nonsense. Check CD first (astropy warns on cdelt access
    # when both are present).
    try:
        if wcs.wcs.has_cd():
            if not np.any(wcs.wcs.cd):
                return None
        else:
            if not np.any(wcs.wcs.cdelt):
                return None
    except Exception:  # noqa: BLE001
        return None
    return wcs


def _read_fits_hdu(path: PathLike) -> tuple[np.ndarray, fits.Header]:
    """Open the FITS file, pick the first HDU with image data, return
    (data_array, header). Shared by `load_fits` and `load_fits_with_wcs`
    so both entry points see identical pixels."""
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
    return data, header


def _data_to_rgb(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Debayer / reshape the raw data array to (H, W, 3) float32 [0,1]."""
    if data.ndim == 3 and data.shape[0] in (3, 4):
        rgb = np.moveaxis(data[:3], 0, -1)
        return _to_float01(rgb, header)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return _to_float01(data[..., :3], header)
    if data.ndim != 2:
        raise ValueError(f"Unsupported FITS data shape {data.shape}; expected 2D Bayer mosaic")
    pattern = _detect_bayer_pattern(header)
    mono = _to_float01(data, header)
    rgb = demosaicing_CFA_Bayer_Malvar2004(mono, pattern=pattern)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def load_fits(path: PathLike) -> np.ndarray:
    """Load a Seestar FITS file and return a debayered RGB image.

    Backwards-compat wrapper around `load_fits_with_wcs` that drops the
    WCS. New callers should prefer the `_with_wcs` variant so SPCC can
    calibrate colour against a star catalog.

    Returns a float32 array of shape (H, W, 3) with values in [0, 1].
    """
    img, _ = load_fits_with_wcs(path)
    return img


def load_fits_with_wcs(
    path: PathLike,
) -> tuple[np.ndarray, WCS | None]:
    """Load a Seestar FITS file and return `(image, wcs_or_none)`.

    The image contract is the same as `load_fits`: float32 `(H, W, 3)`
    in `[0, 1]`. The second element is an `astropy.wcs.WCS` instance
    when the header carries a valid astrometric solution (Seestar
    firmware writes one on every frame via its built-in plate solve),
    or `None` when the solve failed at capture time.

    Downstream code (SPCC) should treat a missing WCS as "skip
    photometric calibration, fall back to heuristic WB".
    """
    data, header = _read_fits_hdu(path)
    rgb = _data_to_rgb(data, header)
    wcs = _parse_wcs(header)
    return rgb, wcs


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
