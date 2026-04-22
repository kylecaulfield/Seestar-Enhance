"""Tests for WCS loading (Phase 2 of SPCC).

`load_fits_with_wcs` should
  1. Return a valid `astropy.wcs.WCS` when the FITS header carries
     the Seestar-style CTYPE/CRVAL/CRPIX/CD card set.
  2. Round-trip a pixel coordinate through pix→world→pix with
     sub-pixel accuracy (proves the WCS was actually parsed, not just
     returned as a stub).
  3. Fall back to `None` when the header is missing WCS cards, so
     callers without astrometric solutions (older firmware, failed
     solves, non-Seestar data) stay on the heuristic-WB path.

Sample-using tests auto-skip when `samples/` is empty so the suite
stays green in environments that don't ship the real frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from app.stages.io_fits import _parse_wcs, load_fits_with_wcs


_SAMPLES = Path(__file__).resolve().parents[2] / "samples"
_SEESTAR_SAMPLE = _SAMPLES / "NGC 6888.fit"


@pytest.mark.skipif(
    not _SEESTAR_SAMPLE.is_file(),
    reason=f"sample FITS not present: {_SEESTAR_SAMPLE}",
)
def test_load_seestar_sample_has_wcs() -> None:
    img, wcs = load_fits_with_wcs(_SEESTAR_SAMPLE)
    assert img.shape == (1920, 1080, 3)
    assert wcs is not None, "Seestar firmware should always write WCS"


@pytest.mark.skipif(
    not _SEESTAR_SAMPLE.is_file(),
    reason=f"sample FITS not present: {_SEESTAR_SAMPLE}",
)
def test_wcs_round_trip_pixel_to_sky_to_pixel() -> None:
    """A pixel round-tripped through pix→world→pix should return to
    itself within a small fraction of a pixel. This is a sanity check
    on the WCS parse, not on astropy — if either step is silently
    wrong, the round-trip will drift by many pixels."""
    _, wcs = load_fits_with_wcs(_SEESTAR_SAMPLE)
    assert wcs is not None
    # A few test pixels including the WCS reference pixel.
    test_pixels = np.array(
        [[540, 960], [100, 100], [1000, 1800], [781.98, 1235.30]],
        dtype=np.float64,
    )
    world = wcs.all_pix2world(test_pixels, 0)
    back = wcs.all_world2pix(world, 0)
    # Allow up to 0.01 pixel round-trip drift; typical is ~1e-9.
    assert np.max(np.abs(back - test_pixels)) < 0.01


@pytest.mark.skipif(
    not _SEESTAR_SAMPLE.is_file(),
    reason=f"sample FITS not present: {_SEESTAR_SAMPLE}",
)
def test_wcs_matches_header_ra_dec() -> None:
    """The image-centre pixel should map to roughly the OBJCTRA/OBJCTDEC
    in the header (Seestar's slew target). Not exact — the image centre
    is not the plate-solve reference point — but within a degree."""
    _, wcs = load_fits_with_wcs(_SEESTAR_SAMPLE)
    assert wcs is not None
    with fits.open(str(_SEESTAR_SAMPLE)) as hdul:
        ra = float(hdul[0].header["RA"])
        dec = float(hdul[0].header["DEC"])
    # Image is (H=1920, W=1080); WCS first axis is sky X (RA), so we
    # pass (x=W/2, y=H/2) = (540, 960).
    centre_world = wcs.all_pix2world([[540, 960]], 0)[0]
    assert abs(centre_world[0] - ra) < 1.0
    assert abs(centre_world[1] - dec) < 1.0


def test_parse_wcs_returns_none_without_ctype() -> None:
    """Empty / no-WCS header should return None, not raise."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    assert _parse_wcs(hdr) is None


def test_parse_wcs_returns_none_on_zero_cd() -> None:
    """Degenerate WCS (zero CD matrix) should return None — otherwise
    downstream code would get nonsense coordinates from a silently-
    broken projection."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX1"] = 32.0
    hdr["CRPIX2"] = 32.0
    hdr["CD1_1"] = 0.0
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.0
    assert _parse_wcs(hdr) is None


def test_load_fits_wrapper_still_returns_ndarray() -> None:
    """Existing callers use `load_fits()`; that entry point must keep
    returning a single ndarray so we don't break every test file."""
    from app.stages.io_fits import load_fits

    if not _SEESTAR_SAMPLE.is_file():
        pytest.skip(f"sample FITS not present: {_SEESTAR_SAMPLE}")
    img = load_fits(_SEESTAR_SAMPLE)
    assert isinstance(img, np.ndarray)
    assert img.shape == (1920, 1080, 3)
