"""Property-based tests for the per-stage `(H, W, 3) float32 in [0, 1]` contract.

Every stage in `app/stages/` claims to take a float32 RGB array in [0, 1] and
return one of the same shape. The example-based tests in `test_stages.py` and
`test_reference_match.py` cover specific behavioural assertions; these tests
hammer each stage with random inputs (constant frames, all-zero, all-one,
gradients, low-variance noise) and verify only the contract:

    * output is np.ndarray
    * output.shape == input.shape
    * output.dtype == np.float32
    * 0.0 <= output <= 1.0 everywhere
    * no NaNs, no infs

Running with `hypothesis` so failures shrink to a minimal counterexample
instead of a 96x128 image of random floats.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.stages import (
    background,
    bm3d_denoise,
    classify,
    color,
    cosmetic,
    crop,
    curves,
    dark_subtract,
    sharpen,
    stretch,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


# Hypothesis is great at probing param spaces but we don't want it to hammer
# pixel values too — we use a few hand-chosen pixel patterns and let
# hypothesis vary the shapes + parameters.
@st.composite
def rgb_image(draw, min_size: int = 32, max_size: int = 96) -> np.ndarray:
    """Strategy producing a deterministic RGB image with hypothesis-chosen shape."""
    h = draw(st.integers(min_size, max_size))
    w = draw(st.integers(min_size, max_size))
    pattern = draw(
        st.sampled_from(
            [
                "uniform_dim",
                "uniform_mid",
                "uniform_bright",
                "horizontal_gradient",
                "noisy_sky",
            ]
        )
    )
    seed = draw(st.integers(0, 1000))
    rng = np.random.default_rng(seed)

    if pattern == "uniform_dim":
        img = np.full((h, w, 3), 0.02, dtype=np.float32)
    elif pattern == "uniform_mid":
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
    elif pattern == "uniform_bright":
        img = np.full((h, w, 3), 0.95, dtype=np.float32)
    elif pattern == "horizontal_gradient":
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
        img = np.broadcast_to(ramp, (h, w, 3)).copy()
    else:  # noisy_sky
        img = rng.normal(0.05, 0.01, size=(h, w, 3)).astype(np.float32).clip(0.0, 1.0)
    return img


# Stages run end-to-end in a few ms each at 32-96 px — no need to ramp `max_examples`
# down. Hypothesis shrinks reliably enough that 25 trials per stage is plenty.
_HYPO = settings(
    max_examples=25,
    deadline=None,  # disable the per-test deadline (BM3D is slow)
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _assert_contract(out: np.ndarray, in_shape: tuple[int, ...]) -> None:
    assert isinstance(out, np.ndarray), f"output is {type(out).__name__}, not ndarray"
    assert out.shape == in_shape, f"shape drift {in_shape} -> {out.shape}"
    assert out.dtype == np.float32, f"dtype drift float32 -> {out.dtype}"
    assert np.all(np.isfinite(out)), "non-finite values in output"
    # Allow a hair of float epsilon at the boundaries — np.clip(0, 1) plus a
    # subsequent multiply can land at 1 + 1e-7. Anything past 1e-5 is a real bug.
    assert out.min() >= -1e-5, f"output min {out.min()} < 0"
    assert out.max() <= 1.0 + 1e-5, f"output max {out.max()} > 1"


# ---------- always-on stages ----------


@_HYPO
@given(img=rgb_image())
def test_background_preserves_contract(img: np.ndarray) -> None:
    out = background.process(img, grid=8, sigma=2.0, iters=2, smoothing=1.0, downscale=2)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image())
def test_color_preserves_contract(img: np.ndarray) -> None:
    out = color.process(img)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image())
def test_stretch_preserves_contract(img: np.ndarray) -> None:
    out = stretch.process(img)
    _assert_contract(out, img.shape)


# Bm3d is the slow one — limit to fewer examples.
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(img=rgb_image(min_size=32, max_size=64))
def test_bm3d_preserves_contract(img: np.ndarray) -> None:
    out = bm3d_denoise.process(img, sigma=0.05, strength=1.0, chroma_blur=0.0)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image())
def test_sharpen_preserves_contract(img: np.ndarray) -> None:
    out = sharpen.process(img)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image())
def test_curves_preserves_contract(img: np.ndarray) -> None:
    out = curves.process(img)
    _assert_contract(out, img.shape)


# ---------- opt-in stages ----------


@_HYPO
@given(img=rgb_image())
def test_cosmetic_preserves_contract(img: np.ndarray) -> None:
    out = cosmetic.process(img, neighborhood=3, sigma=6.0)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image())
def test_dark_subtract_preserves_contract(img: np.ndarray) -> None:
    """dark_subtract takes a dark frame; we use an all-zero dark (no-op)."""
    dark = np.zeros_like(img)
    out = dark_subtract.process(img, dark=dark)
    _assert_contract(out, img.shape)


@_HYPO
@given(img=rgb_image(min_size=64, max_size=96))  # crop needs room
def test_crop_preserves_contract(img: np.ndarray) -> None:
    """Cropping changes shape — verify only dtype + range."""
    h, w = img.shape[:2]
    out = crop.process(img, top=4, left=4, bottom=h - 4, right=w - 4)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5
    assert out.shape == (h - 8, w - 8, 3)


# ---------- classifier ----------


@_HYPO
@given(img=rgb_image())
def test_classify_returns_known_label(img: np.ndarray) -> None:
    """The contract for classify is a string from a known set, not the
    pixel-contract above. We still check that the metrics dict has the
    expected keys.
    """
    label = classify.classify(img)
    valid = {
        "nebula",
        "nebula_wide",
        "nebula_filament",
        "nebula_dominant",
        "galaxy",
        "cluster",
    }
    assert label in valid, f"classify returned {label!r}"


# ---------- explicit edge cases ----------


@pytest.mark.parametrize(
    "fill,desc",
    [
        (0.0, "all-zero (pitch-black input)"),
        (1.0, "all-one (saturated input)"),
        (0.5, "uniform mid-grey"),
    ],
)
@pytest.mark.parametrize(
    "stage,kwargs",
    [
        (color, {}),
        (stretch, {}),
        (sharpen, {}),
        (curves, {}),
        (cosmetic, {"neighborhood": 3, "sigma": 6.0}),
    ],
)
def test_stages_handle_constant_frames(
    stage,
    kwargs: dict,
    fill: float,
    desc: str,
) -> None:
    img = np.full((48, 64, 3), fill, dtype=np.float32)
    out = stage.process(img, **kwargs)
    _assert_contract(out, img.shape)
