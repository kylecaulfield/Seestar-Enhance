"""Export the final pipeline image to PNG, TIFF, or FITS.

The output format is dispatched by file suffix so callers don't need a
separate `format=` arg — `output.png` / `output.tif` / `output.fits`
each write the right bytes:

* **PNG** (`.png`): 16-bit RGB via `pypng`. Pillow's high-level API
  doesn't support 16-bit RGB output cleanly, so we go through pypng.
  Default for the web UI — universally supported, sized for screen.
* **TIFF** (`.tif`/`.tiff`): 16-bit RGB via `tifffile`. The format
  PixInsight, Siril, GIMP, and Photoshop all read losslessly. Same
  precision as the PNG path, different container.
* **FITS** (`.fits`/`.fit`/`.fts`): float32 RGB cube (axis order
  `(H, W, 3)` flipped to `(3, H, W)` per the FITS NAXIS convention)
  via `astropy.io.fits`. Preserves the full pipeline precision —
  values stay in [0, 1] floating-point with no quantisation.
  Useful for users who want to re-stretch or further process in
  PixInsight without losing dynamic range.

All three paths return the (clipped, float32) image array so callers
that want both an in-memory copy and a file on disk don't need a
second round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import png

PathLike = str | Path


_PNG_SUFFIXES = {".png"}
_TIFF_SUFFIXES = {".tif", ".tiff"}
_FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def _format_for(path: Path) -> str:
    """Return ``"png" | "tiff" | "fits"`` based on the path's suffix.

    Raises ``ValueError`` for an unrecognised extension so the caller
    fails loudly rather than silently writing the wrong format.
    """
    suffix = path.suffix.lower()
    if suffix in _PNG_SUFFIXES:
        return "png"
    if suffix in _TIFF_SUFFIXES:
        return "tiff"
    if suffix in _FITS_SUFFIXES:
        return "fits"
    raise ValueError(
        f"unsupported output suffix {suffix!r}; expected one of "
        f"{sorted(_PNG_SUFFIXES | _TIFF_SUFFIXES | _FITS_SUFFIXES)}"
    )


def _write_png(image_clipped: np.ndarray, path: Path) -> None:
    """Write a 16-bit RGB PNG via pypng."""
    as_u16 = np.round(image_clipped * 65535.0).astype(np.uint16)
    h, w, _ = as_u16.shape
    rows = as_u16.reshape(h, w * 3)
    writer = png.Writer(width=w, height=h, bitdepth=16, greyscale=False)
    with path.open("wb") as f:
        writer.write(f, rows.tolist())


def _write_tiff(image_clipped: np.ndarray, path: Path) -> None:
    """Write a 16-bit RGB TIFF via tifffile.

    Imported lazily so callers that only need PNG don't pay the
    import cost. tifffile is the de-facto standard for 16-bit
    scientific TIFF — Pillow can do it but quirks with photometric
    interpretation under PIL on Windows are well known, and PixInsight
    is happier reading tifffile output.
    """
    import tifffile

    as_u16 = np.round(image_clipped * 65535.0).astype(np.uint16)
    tifffile.imwrite(
        str(path),
        as_u16,
        photometric="rgb",
        # zlib (deflate) is the only lossless compressor tifffile
        # ships natively without `imagecodecs`. LZW would be a
        # hair smaller for natural images but pulls a 50 MB C-ext
        # dep that isn't worth it for our use case. Every TIFF
        # reader supports zlib.
        compression="zlib",
    )


def _write_fits(image_clipped: np.ndarray, path: Path) -> None:
    """Write a float32 (3, H, W) FITS cube via astropy.

    FITS NAXIS convention puts the slowest-varying axis last, so RGB
    becomes axis 3 (NAXIS3=3). We reorder (H, W, 3) -> (3, H, W) on
    the way out. Values stay in [0, 1] float32 — no quantisation,
    matches the pipeline's working precision.
    """
    from astropy.io import fits

    cube = np.moveaxis(image_clipped.astype(np.float32, copy=False), -1, 0)
    hdu = fits.PrimaryHDU(cube)
    hdu.header["BUNIT"] = "normalised"
    hdu.header["COMMENT"] = "Seestar Enhance pipeline output. RGB stored as NAXIS3=3 cube."
    # `overwrite=True` mirrors PNG behaviour; the caller expects a
    # write, not an error if a previous run is on disk.
    hdu.writeto(str(path), overwrite=True)


def process(image: np.ndarray, path: PathLike) -> np.ndarray:
    """Write the pipeline output to disk in the format implied by ``path``.

    Returns the clipped float32 image (so a caller doing both an
    in-memory and on-disk pass doesn't need to re-clip).
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = _format_for(out)

    if fmt == "png":
        _write_png(clipped, out)
    elif fmt == "tiff":
        _write_tiff(clipped, out)
    elif fmt == "fits":
        _write_fits(clipped, out)
    else:  # unreachable — _format_for already raised
        raise ValueError(f"unhandled format {fmt!r}")

    return clipped
