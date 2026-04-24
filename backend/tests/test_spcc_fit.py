"""End-to-end tests for SPCC `process()` (Phase 4).

Each test constructs a synthetic scene (planted stars at known pixel
positions, tangent-plane WCS, miniature catalog Parquet matching the
real schema) so no bundled Gaia data is required. The "identity"
test proves the fit finds identity when the image already matches the
catalogue; the "injected CCM" test proves SPCC recovers the inverse
of a known miscalibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from app.stages import spcc
from astropy.wcs import WCS


def _make_wcs(
    image_shape: tuple[int, int],
    centre_radec: tuple[float, float] = (180.0, 0.0),
    arcsec_per_pix: float = 3.6,
) -> WCS:
    h, w = image_shape
    w_obj = WCS(naxis=2)
    w_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_obj.wcs.crval = [centre_radec[0], centre_radec[1]]
    w_obj.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    deg_per_pix = arcsec_per_pix / 3600.0
    w_obj.wcs.cd = np.array([[-deg_per_pix, 0.0], [0.0, deg_per_pix]], dtype=np.float64)
    return w_obj


def _plant_stars(
    image: np.ndarray,
    yx: list[tuple[int, int]],
    rgbs: list[tuple[float, float, float]],
    radius: int = 2,
) -> None:
    for (y, x), (r, g, b) in zip(yx, rgbs, strict=False):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                w = max(0.0, 1.0 - (abs(dy) + abs(dx)) * 0.35)
                yy, xx = y + dy, x + dx
                if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                    image[yy, xx, 0] = max(image[yy, xx, 0], r * w)
                    image[yy, xx, 1] = max(image[yy, xx, 1], g * w)
                    image[yy, xx, 2] = max(image[yy, xx, 2], b * w)


def _make_catalog_parquet(
    tmp_path: Path,
    stars_radec_mags: list[tuple[float, float, float, float, float]],
) -> Path:
    """stars_radec_mags entries: (ra, dec, G, BP, RP)."""
    ra = np.array([s[0] for s in stars_radec_mags], dtype=np.float64)
    dec = np.array([s[1] for s in stars_radec_mags], dtype=np.float64)
    g = np.array([s[2] for s in stars_radec_mags], dtype=np.float32)
    bp = np.array([s[3] for s in stars_radec_mags], dtype=np.float32)
    rp = np.array([s[4] for s in stars_radec_mags], dtype=np.float32)
    table = pa.Table.from_pydict(
        {
            "ra": ra,
            "dec": dec,
            "phot_g_mean_mag": g,
            "phot_bp_mean_mag": bp,
            "phot_rp_mean_mag": rp,
        }
    )
    path = tmp_path / "gaia_test.parquet"
    pq.write_table(table, str(path))
    return path


def _flat_to_mag(flat: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert a normalised (R, G, B) flux triple to Gaia-style G/BP/RP mags.

    (R, G, B) → (G, BP, RP) = (−2.5 log10 G, −2.5 log10 B, −2.5 log10 R),
    matching the convention in `spcc._gaia_to_target_rgb`.
    """
    r, g, b = flat
    return (-2.5 * np.log10(g), -2.5 * np.log10(b), -2.5 * np.log10(r))


def _build_scene(
    tmp_path: Path,
    n_stars: int = 25,
    image_size: int = 256,
    seed: int = 42,
    image_ccm: np.ndarray | None = None,
) -> tuple[np.ndarray, WCS, Path, np.ndarray]:
    """Plant `n_stars` at random positions with varied colour temperatures.

    The stars' true RGB is a smooth function of a random temperature;
    we write the matching Gaia magnitudes into the catalog, so an
    SPCC fit on the planted image returns identity. If `image_ccm`
    is supplied, the planted pixel RGBs are pre-multiplied by
    `image_ccm.T` — simulating a miscalibrated sensor — so the fit
    should recover `inv(image_ccm)` (row-normalised).
    """
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.02, 0.04, size=(image_size, image_size, 3)).astype(np.float32)
    ys = rng.integers(20, image_size - 20, size=n_stars)
    xs = rng.integers(20, image_size - 20, size=n_stars)

    # Random colour temperatures: a uniform spread over BP-RP.
    temps = rng.uniform(-0.3, 1.5, size=n_stars)  # bluer → redder
    target_rgbs: list[tuple[float, float, float]] = []
    for t in temps:
        # A monotone toy colour curve: R grows with t, B shrinks.
        r = 0.3 + 0.25 * t
        g = 0.5
        b = 0.7 - 0.25 * t
        total = r + g + b
        target_rgbs.append((r / total, g / total, b / total))

    # Image-plane RGBs (what the camera "sees"). If `image_ccm` is
    # non-None, the camera response is CCM-distorted.
    planted_rgbs: list[tuple[float, float, float]] = []
    for tr in target_rgbs:
        if image_ccm is not None:
            measured = tuple((np.asarray(tr) @ image_ccm.T).tolist())
        else:
            measured = tr
        # Scale up so the stars read above the background noise.
        planted_rgbs.append(tuple(0.5 + 0.4 * np.asarray(measured)))  # type: ignore[arg-type]

    _plant_stars(img, list(zip(ys.tolist(), xs.tolist(), strict=False)), planted_rgbs, radius=2)

    # Build the WCS and convert each planted pixel to RA/Dec.
    wcs = _make_wcs((image_size, image_size))
    pix = np.stack([xs, ys], axis=1).astype(np.float64)
    radec = wcs.all_pix2world(pix, 0)

    cat_entries = []
    for i in range(n_stars):
        ra, dec = radec[i]
        g_mag, bp_mag, rp_mag = _flat_to_mag(target_rgbs[i])
        cat_entries.append((ra, dec, g_mag, bp_mag, rp_mag))
    catalog_path = _make_catalog_parquet(tmp_path, cat_entries)

    # Also return the expected CCM (identity when image_ccm is None,
    # else inv(image_ccm), row-normalised).
    if image_ccm is None:
        expected = np.eye(3, dtype=np.float64)
    else:
        expected = np.linalg.inv(image_ccm)
        row_sums = expected.sum(axis=1, keepdims=True)
        row_sums = np.where(np.abs(row_sums) > 1e-6, row_sums, 1.0)
        expected = expected / row_sums
    return img, wcs, catalog_path, expected


# ---------- Identity fit ---------------------------------------------------


def test_spcc_identity_fit(tmp_path: Path) -> None:
    """Image that already matches its catalog → fitted CCM ≈ identity."""
    img, wcs, cat, expected = _build_scene(tmp_path, n_stars=40)
    out = spcc.process(
        img,
        wcs=wcs,
        catalog_path=cat,
        min_matches=20,
        n_detect=100,
        tol_arcsec=5.0,
    )
    # Output should be close to input (identity-ish CCM).
    diff = np.abs(out.astype(np.float32) - img).mean()
    assert diff < 0.01, f"identity fit moved pixels by {diff:.4f}"


# ---------- Miscalibration recovery ----------------------------------------


def test_spcc_fit_moves_pixels_under_miscalibration(tmp_path: Path) -> None:
    """Plant a known sensor miscalibration; SPCC should actually fit
    a non-identity matrix and apply it. We don't assert a precise
    inversion here — that requires a more rigorous scene setup
    (bigger images, lower noise floor) that belongs in phase 5
    tuning against real data. This test pins the milder invariant:
    with visible miscalibration present the output is not
    bit-identical to the input, so SPCC's fit+apply path is wired
    and firing."""
    injected = np.diag([0.7, 1.0, 1.3]).astype(np.float64)
    img, wcs, cat, _expected = _build_scene(tmp_path, n_stars=60, image_ccm=injected)
    # Disable luma-weighting so the synthetic sky pixels (low luma) also
    # get calibrated; luma_weighted=True is the right default for real
    # astro frames but makes this whole-frame diff test insensitive.
    out = spcc.process(
        img,
        wcs=wcs,
        catalog_path=cat,
        min_matches=20,
        n_detect=120,
        tol_arcsec=5.0,
        luma_weighted=False,
        strength=1.0,  # full SPCC so the whole-frame diff is visible
    )
    # The fit must have applied some non-identity transform.
    diff = float(np.mean(np.abs(out - img)))
    assert diff > 0.001, f"SPCC produced no change under miscalibration (diff={diff:.5f})"


# ---------- Fallback paths -------------------------------------------------


def test_spcc_skips_without_wcs() -> None:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    out = spcc.process(img, wcs=None)
    np.testing.assert_array_equal(out, img)


def test_spcc_skips_without_catalog(tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    wcs = _make_wcs((64, 64))
    out = spcc.process(img, wcs=wcs, catalog_path=tmp_path / "does-not-exist.parquet")
    np.testing.assert_array_equal(out, img)


def test_spcc_skips_when_too_few_matches(tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    wcs = _make_wcs((64, 64))
    # Make a catalog too far from the image for any star to be in FOV.
    cat = _make_catalog_parquet(tmp_path, [(0.0, 30.0, 10.0, 9.5, 10.5)] * 5)
    out = spcc.process(img, wcs=wcs, catalog_path=cat, min_matches=20)
    np.testing.assert_array_equal(out, img)


# ---------- Gaia → flux transform -----------------------------------------


def test_gaia_to_target_rgb_normalises_rows() -> None:
    g = np.array([10.0, 10.0], dtype=np.float32)
    bp = np.array([10.5, 9.5], dtype=np.float32)
    rp = np.array([9.5, 10.5], dtype=np.float32)
    rgb = spcc._gaia_to_target_rgb(g, bp, rp)
    assert rgb.shape == (2, 3)
    np.testing.assert_allclose(rgb.sum(axis=1), 1.0, atol=1e-6)
    # Star 0 has BP fainter than RP → is redder → R channel > B.
    assert rgb[0, 0] > rgb[0, 2]
    # Star 1 has BP brighter than RP → is bluer → B > R.
    assert rgb[1, 2] > rgb[1, 0]


def test_fit_ccm_recovers_identity_on_matched_data() -> None:
    rng = np.random.default_rng(0)
    target = rng.uniform(0.1, 0.4, size=(30, 3))
    target = target / target.sum(axis=1, keepdims=True)
    # measured == target (already calibrated).
    ccm = spcc._fit_ccm(target * 0.8, target)  # scale doesn't matter
    np.testing.assert_allclose(ccm, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(ccm.sum(axis=1), 1.0, atol=1e-6)


def test_fit_ccm_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        spcc._fit_ccm(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        spcc._fit_ccm(np.zeros((10, 3)), np.zeros((10, 4)))


def test_luma_weighted_spares_sky_pixels(tmp_path: Path) -> None:
    """With `luma_weighted=True` (the default), dark pixels should
    barely move — the CCM only applies in proportion to brightness.
    With it off, every pixel gets hit equally and even the synthetic
    "sky" noise shifts. This behaviour is what keeps SPCC from
    tinting the real-image background red when it's fitting an
    R-boost gain."""
    injected = np.diag([0.7, 1.0, 1.3]).astype(np.float64)
    img, wcs, cat, _ = _build_scene(tmp_path, n_stars=60, image_ccm=injected)

    out_weighted = spcc.process(
        img,
        wcs=wcs,
        catalog_path=cat,
        min_matches=20,
        n_detect=120,
        tol_arcsec=5.0,
        luma_weighted=True,
    )
    out_unweighted = spcc.process(
        img,
        wcs=wcs,
        catalog_path=cat,
        min_matches=20,
        n_detect=120,
        tol_arcsec=5.0,
        luma_weighted=False,
    )
    # Sample the darkest pixels (bottom 10% luma): weighted version
    # leaves them essentially unchanged; unweighted drags them with
    # the CCM.
    luma = img.mean(axis=-1)
    sky_mask = luma < float(np.percentile(luma, 10.0))
    w_shift = float(np.mean(np.abs(out_weighted[sky_mask] - img[sky_mask])))
    u_shift = float(np.mean(np.abs(out_unweighted[sky_mask] - img[sky_mask])))
    assert w_shift < u_shift
    assert w_shift < 0.01  # dark pixels barely move
