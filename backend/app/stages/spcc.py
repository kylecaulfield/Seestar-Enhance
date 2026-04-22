"""Photometric colour calibration (SPCC) — v2 placeholder.

Reads a bundled Gaia DR3 subset (`app/data/gaia_bright.parquet`),
projects catalogue stars onto the image plane using the WCS parsed
from the FITS header, cross-matches against detected stars, and fits
a 3×3 colour correction matrix that maps our sensor response to the
Gaia-predicted sRGB. The fitted CCM is applied in place.

This module lands in phases:

  Phase 2 (this commit): module exists; `process()` is a no-op that
    accepts the WCS + opt-in params so the pipeline can be wired
    end-to-end without committing to the final signature.
  Phase 3: private helpers — star detection, FOV-filter, cross-match,
    aperture photometry.
  Phase 4: the actual lstsq fit, plus the "too few matches → skip"
    fallback.

Signature is stable across the phases: callers that wire up now
(`pipeline.run`, profile entries) will not need to change when the
real implementation lands.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def process(
    image: np.ndarray,
    wcs: Optional[Any] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Apply photometric colour calibration.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3) in [0, 1].
    wcs : astropy.wcs.WCS | None
        Parsed WCS from the FITS header. SPCC cannot run without it;
        when `None`, the function returns the image unchanged.
    **kwargs
        Opt-in params carried via the profile dict (e.g. `min_matches`,
        `catalog_path`). Accepted and ignored in phase 2.

    Returns
    -------
    np.ndarray
        The input image, unchanged in phase 2. Phase 4 returns the
        colour-calibrated image.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    del wcs, kwargs  # consumed in phase 4
    return image.astype(np.float32, copy=False)
