"""Photometric colour calibration (SPCC) — v2.

Reads a bundled Gaia DR3 subset (`app/data/gaia_bright.parquet`),
projects catalogue stars onto the image plane using the WCS parsed
from the FITS header, cross-matches against detected stars, and fits
a 3×3 colour correction matrix that maps our sensor response to the
Gaia-predicted sRGB. The fitted CCM is applied in place.

This module lands in phases:

  Phase 2: module exists; `process()` is a no-op so the pipeline can
    be wired without committing to the final signature.
  Phase 3 (this commit): private helpers — star detection, FOV filter,
    cross-match, aperture photometry. Tested against synthetic data.
  Phase 4: the actual lstsq fit in `process()`.

The helpers are module-private (leading underscore) so their exact
signatures can still change before Phase 4 wires them up. Phase 4
will expose the subset that `process()` uses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


# ---------- helpers ---------------------------------------------------------


def _detect_bright_stars(
    image: np.ndarray,
    n: int = 100,
    suppression: int = 7,
    mad_threshold: float = 6.0,
) -> np.ndarray:
    """Return the (y, x) pixel coordinates of up to `n` detected stars.

    Stars are local maxima of the luma channel above a MAD-based
    threshold (`mad_threshold` sigma equivalent). A `suppression`-
    sized window enforces minimum separation so a bright star doesn't
    get counted multiple times by adjacent pixels near the peak.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3) in [0, 1].
    n : int
        Maximum stars to return. Sorted by luma, brightest first.
    suppression : int
        Size of the local-maximum window (odd, ≥ 3). 7 comfortably
        separates adjacent Seestar stars (FWHM ~3 px).
    mad_threshold : float
        Rejection threshold in MAD-scaled Gaussian sigmas. Lower =
        catches fainter stars at the cost of more noise peaks.

    Returns
    -------
    np.ndarray
        Shape `(N, 2)` int array of `(y, x)` pairs. N ≤ n.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    luma = image.mean(axis=-1).astype(np.float32)
    med = float(np.median(luma))
    mad = float(np.median(np.abs(luma - med))) * 1.4826
    if mad < 1e-6:
        mad = 1e-6
    threshold = med + mad_threshold * mad

    local_max = maximum_filter(luma, size=suppression, mode="reflect")
    is_peak = (luma >= local_max) & (luma > threshold)
    ys, xs = np.nonzero(is_peak)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Sort by luma descending, keep top n.
    luma_at_peaks = luma[ys, xs]
    order = np.argsort(-luma_at_peaks, kind="stable")
    if order.size > n:
        order = order[:n]
    return np.stack([ys[order], xs[order]], axis=1).astype(np.int64)


def _catalog_in_fov(
    cat: Dict[str, np.ndarray],
    wcs: Any,
    image_shape: Tuple[int, int],
    margin_deg: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Filter a catalog dict to rows whose RA/Dec project inside the image.

    Uses a two-stage filter:
      1. A cheap RA/Dec bounding-box cut drops ~99% of the catalog
         without projecting anything. Essential when `cat` has
         millions of rows.
      2. The surviving rows get projected through `wcs.all_world2pix`;
         rows whose (x, y) falls outside `[0, W) × [0, H)` expanded by
         `margin_deg` worth of pixels are dropped.

    Parameters
    ----------
    cat : dict[str, np.ndarray]
        Columns at minimum: ``ra``, ``dec``. Typically also carries
        the three Gaia magnitude columns that SPCC will use; they pass
        through unchanged.
    wcs : astropy.wcs.WCS
        The FITS-derived projection.
    image_shape : (H, W)
        Used to define the "inside the frame" test.
    margin_deg : float
        Extra slack around the frame, in degrees, to catch stars whose
        projected position lands slightly outside due to distortion.

    Returns
    -------
    dict[str, np.ndarray]
        Same columns as `cat`, filtered to the in-FOV subset.
    """
    if "ra" not in cat or "dec" not in cat:
        raise ValueError("catalog must have 'ra' and 'dec' columns")
    h, w = image_shape
    ra = np.asarray(cat["ra"], dtype=np.float64)
    dec = np.asarray(cat["dec"], dtype=np.float64)

    # Stage 1 — RA/Dec bounding box using the WCS reference point plus
    # a generous radius. Seestar FOV is ~1.3° on the diagonal; we use
    # 2× that as the bounding-box half-size, then the precise filter
    # does the real work.
    crval = np.asarray(wcs.wcs.crval, dtype=np.float64)
    # 3° box is way more than any Seestar frame can cover.
    ra0, dec0 = float(crval[0]), float(crval[1])
    box = 3.0 + margin_deg
    # Wrap-safe RA delta.
    d_ra = np.abs(((ra - ra0 + 540.0) % 360.0) - 180.0)
    d_dec = np.abs(dec - dec0)
    bbox_mask = (d_ra < box / max(np.cos(np.deg2rad(dec0)), 0.05)) & (d_dec < box)
    if not bbox_mask.any():
        return {k: v[:0] for k, v in cat.items()}

    # Stage 2 — precise WCS projection on the bbox-filtered subset.
    idx = np.nonzero(bbox_mask)[0]
    coords = np.stack([ra[idx], dec[idx]], axis=1)
    pixels = wcs.all_world2pix(coords, 0)
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    # Convert margin_deg to px using the CD scale (approximate).
    try:
        cd = wcs.wcs.cd if wcs.wcs.has_cd() else None
    except Exception:  # noqa: BLE001
        cd = None
    if cd is not None and np.any(cd):
        px_per_deg = 1.0 / np.max(np.abs(cd))
    else:
        px_per_deg = 1800.0  # fallback; ~Seestar-sized
    margin_px = float(margin_deg * px_per_deg)

    keep = (xs > -margin_px) & (xs < w + margin_px) & \
           (ys > -margin_px) & (ys < h + margin_px) & \
           np.isfinite(xs) & np.isfinite(ys)
    kept_idx = idx[keep]

    out = {k: np.asarray(v)[kept_idx] for k, v in cat.items()}
    return out


def _cross_match(
    img_radec: np.ndarray,
    cat_radec: np.ndarray,
    tol_arcsec: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour match of image stars to catalogue stars.

    Operates in a Dec-scaled flat (RA', Dec) plane: at angular scales
    of arcseconds this is accurate to ~10⁻⁶ arcsec, well inside the
    tolerance we care about. A real great-circle distance (haversine)
    would be marginally more correct at the ~100 arcsec scale but
    we're matching stars inside a 1° field of view so flat-sky is fine.

    Parameters
    ----------
    img_radec : np.ndarray
        `(N, 2)` array of detected stars' (RA, Dec) in degrees.
    cat_radec : np.ndarray
        `(M, 2)` array of catalogue stars' (RA, Dec) in degrees.
    tol_arcsec : float
        Maximum match distance. 2 arcsec is ~2/3 of a Seestar pixel,
        tight enough to reject random pair-ups but loose enough to
        absorb the WCS's quoted 1-arcsec residual.

    Returns
    -------
    (img_idx, cat_idx) : tuple of np.ndarray
        Index pairs into `img_radec` and `cat_radec` of matched stars.
        One-to-one: an image star can match at most one catalogue star
        (its nearest within tolerance).
    """
    if img_radec.ndim != 2 or img_radec.shape[-1] != 2:
        raise ValueError(f"img_radec must be (N, 2), got {img_radec.shape}")
    if cat_radec.ndim != 2 or cat_radec.shape[-1] != 2:
        raise ValueError(f"cat_radec must be (M, 2), got {cat_radec.shape}")
    if img_radec.shape[0] == 0 or cat_radec.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Dec-scale RA so a step of 1° in the scaled-RA axis corresponds
    # to 1° on the sky regardless of latitude. Use the mean catalog
    # Dec since the img/cat sets should be colocated on sky anyway.
    dec_scale = float(np.cos(np.deg2rad(cat_radec[:, 1].mean())))
    cat_flat = np.stack(
        [cat_radec[:, 0] * dec_scale, cat_radec[:, 1]], axis=1
    )
    img_flat = np.stack(
        [img_radec[:, 0] * dec_scale, img_radec[:, 1]], axis=1
    )
    tree = cKDTree(cat_flat)
    tol_deg = tol_arcsec / 3600.0
    dists, idxs = tree.query(img_flat, k=1, distance_upper_bound=tol_deg)
    good = np.isfinite(dists)
    img_idx = np.nonzero(good)[0]
    cat_idx = idxs[good]
    return img_idx.astype(np.int64), cat_idx.astype(np.int64)


def _measure_star_rgb(
    image: np.ndarray,
    stars_yx: np.ndarray,
    aperture_r: int = 3,
    annulus_inner: int = 5,
    annulus_outer: int = 8,
) -> np.ndarray:
    """Aperture photometry per star, per channel.

    For each star centre, sum the pixels inside a radius-`aperture_r`
    disk and subtract the local sky estimated from the annulus between
    `annulus_inner` and `annulus_outer`. Returns one `(R, G, B)` triple
    per star — the measured flux minus background.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3).
    stars_yx : np.ndarray
        Shape `(N, 2)` int array of `(y, x)` star centres.
    aperture_r : int
        Radius of the flux-collection disk in pixels. 3 px catches
        almost all of a typical Seestar star's PSF.
    annulus_inner, annulus_outer : int
        Inner / outer radii of the sky annulus. Pixels with
        `annulus_inner ≤ r < annulus_outer` contribute to the sky
        estimate.

    Returns
    -------
    np.ndarray
        Shape `(N, 3)` float32 — background-subtracted aperture flux
        per channel. Rows for stars too close to the image edge to
        fit an annulus are zeroed.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    if aperture_r < 1 or annulus_inner <= aperture_r or annulus_outer <= annulus_inner:
        raise ValueError(
            f"radii must satisfy 1 ≤ aperture_r < annulus_inner < annulus_outer, "
            f"got {aperture_r}, {annulus_inner}, {annulus_outer}"
        )

    h, w = image.shape[:2]
    n = stars_yx.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return out

    # Precompute circular masks relative to the aperture centre.
    rr, cc = np.mgrid[-annulus_outer : annulus_outer + 1,
                      -annulus_outer : annulus_outer + 1]
    dist2 = rr * rr + cc * cc
    aperture_mask = dist2 <= aperture_r * aperture_r
    annulus_mask = (dist2 >= annulus_inner * annulus_inner) & \
                   (dist2 <  annulus_outer * annulus_outer)

    for i, (y, x) in enumerate(stars_yx):
        y0 = y - annulus_outer
        y1 = y + annulus_outer + 1
        x0 = x - annulus_outer
        x1 = x + annulus_outer + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue  # too close to an edge; leave row at zero
        patch = image[y0:y1, x0:x1, :]  # (2R+1, 2R+1, 3)
        aperture_pixels = patch[aperture_mask]      # (Na, 3)
        annulus_pixels = patch[annulus_mask]        # (Nb, 3)
        if aperture_pixels.size == 0 or annulus_pixels.size == 0:
            continue
        # Use the median of the annulus to estimate sky, mean of the
        # aperture for the star flux (sum would work too, but mean is
        # comparable across apertures if radius ever changes).
        sky = np.median(annulus_pixels, axis=0)
        flux = aperture_pixels.mean(axis=0) - sky
        out[i, :] = np.clip(flux, 0.0, None).astype(np.float32)
    return out


# ---------- public API -----------------------------------------------------


def process(
    image: np.ndarray,
    wcs: Optional[Any] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Apply photometric colour calibration.

    Phase 2 no-op: accepts the WCS + opt-in params so the pipeline
    can be wired without committing to the final signature. Phase 4
    replaces the body with the real fit.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")
    del wcs, kwargs  # consumed in phase 4
    return image.astype(np.float32, copy=False)
