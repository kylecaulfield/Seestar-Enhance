"""Tests for reference-match items 6-10:
    - Elongation metric in classifier
    - nebula_wide / nebula_filament sub-variants
    - Seestar S50 CCM
    - Dark-frame subtraction
    - HSV-space saturation mode
"""
from __future__ import annotations

import numpy as np
import pytest

from app import profiles
from app.stages import classify, color, curves, dark_subtract


def _rng_sky(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.01, 0.002, size=(h, w)).astype(np.float32)
    return np.clip(np.stack([base] * 3, axis=-1), 0.0, 1.0)


# ----- Item 7: elongation metric -----

def test_elongation_metric_exists() -> None:
    m = classify._metrics(_rng_sky())
    assert "largest_bright_elongation" in m
    assert 0.0 <= m["largest_bright_elongation"] <= 1.0


def test_elongation_distinguishes_filament_from_blob() -> None:
    h, w = 256, 256
    # Round blob in all channels → low elongation.
    blob = _rng_sky(h, w, seed=1)
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing="ij",
    )
    bump = 0.5 * np.exp(-(xx ** 2 + yy ** 2) / 0.05)
    for c in range(3):
        blob[..., c] += bump
    blob = np.clip(blob, 0.0, 1.0)
    m_blob = classify._metrics(blob)

    # Narrow filament in all channels → high elongation.
    filament = _rng_sky(h, w, seed=2)
    filament[120:130, :, :] += 0.5
    filament = np.clip(filament, 0.0, 1.0)
    m_fil = classify._metrics(filament)

    assert m_fil["largest_bright_elongation"] > m_blob["largest_bright_elongation"]
    assert m_fil["largest_bright_elongation"] > 0.8


# ----- Item 6: sub-variants registered -----

def test_sub_variant_profiles_registered() -> None:
    assert "nebula_wide" in profiles.PROFILES
    assert "nebula_filament" in profiles.PROFILES
    # Each carries the same keys as the parent nebula profile.
    for name in ("nebula_wide", "nebula_filament"):
        p = profiles.get(name)
        for key in ("stretch", "bm3d_denoise", "curves", "stars"):
            assert key in p, f"{name} missing {key}"


# ----- Item 8: sensor CCM -----

def _varied_sky(seed: int = 0) -> np.ndarray:
    """A non-constant sky with a central brighter patch. Gives
    color.process something to balance so the dark-pedestal step
    doesn't zero everything."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.05, 0.10, size=(64, 64, 3)).astype(np.float32)
    img[20:44, 20:44] += np.array([0.15, 0.08, 0.06], dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def test_ccm_shifts_channel_balance() -> None:
    img = _varied_sky(seed=1)
    without = color.process(img.copy(), wb_strength=0.0, ccm=None)
    with_ccm = color.process(img.copy(), wb_strength=0.0, ccm="seestar_s50")
    # The Seestar CCM is non-trivial, so the outputs should differ.
    assert not np.allclose(without, with_ccm, atol=1e-3)


def test_ccm_accepts_explicit_matrix() -> None:
    img = _varied_sky(seed=2)
    # Identity CCM should produce the same output as no CCM (bit-for-bit
    # because CCM is applied as a linear matrix multiplication).
    no_ccm = color.process(img.copy(), ccm=None)
    identity = color.process(img.copy(), ccm=np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(no_ccm, identity, atol=1e-5)


def test_ccm_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        color.process(
            np.zeros((16, 16, 3), dtype=np.float32),
            ccm=np.eye(4),
        )


def test_ccm_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError):
        color.process(
            np.zeros((16, 16, 3), dtype=np.float32),
            ccm="not_a_preset",
        )


# ----- Item 9: dark-frame subtraction -----

def test_dark_subtract_is_noop_when_none() -> None:
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    out = dark_subtract.process(img, dark=None)
    np.testing.assert_array_equal(out, img)


def test_dark_subtract_removes_known_dark() -> None:
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    dark = np.full((32, 32, 3), 0.1, dtype=np.float32)
    out = dark_subtract.process(img, dark=dark)
    expected = 0.4
    assert abs(out.mean() - expected) < 1e-5


def test_dark_subtract_clamps_overshoot() -> None:
    img = np.full((16, 16, 3), 0.05, dtype=np.float32)
    dark = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = dark_subtract.process(img, dark=dark)
    # Negative results after subtraction must clamp to 0.
    assert out.min() >= 0.0


def test_dark_subtract_rejects_mismatched_shape() -> None:
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dark = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        dark_subtract.process(img, dark=dark)


# ----- Item 10: HSV-space saturation -----

def test_hsv_saturation_preserves_hue_on_pure_colors() -> None:
    # Pure red pixel; bump saturation. It should stay red.
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[..., 0] = 0.5  # R=0.5, G=0, B=0
    out = curves.process(
        img,
        contrast=0.0,
        saturation=1.8,
        saturation_mode="hsv",
        star_preserve_percentile=None,
    )
    # Output should still have G=0 and B=0 (hue unchanged); R may shift
    # slightly because HSV scales value-relative saturation.
    assert out[..., 1].max() < 1e-5
    assert out[..., 2].max() < 1e-5


def test_hsv_saturation_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        curves.process(
            np.zeros((4, 4, 3), dtype=np.float32),
            saturation_mode="lab",
        )


def test_saturation_linear_shifts_channels_differently_than_hsv() -> None:
    # A slightly-off-red colour: linear-around-luma will shift hue;
    # HSV won't. They should produce different outputs.
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[..., 0] = 0.6
    img[..., 1] = 0.1
    img[..., 2] = 0.05
    out_linear = curves.process(
        img, contrast=0.0, saturation=2.0, saturation_mode="linear",
        star_preserve_percentile=None,
    )
    out_hsv = curves.process(
        img, contrast=0.0, saturation=2.0, saturation_mode="hsv",
        star_preserve_percentile=None,
    )
    assert not np.allclose(out_linear, out_hsv, atol=1e-3)
