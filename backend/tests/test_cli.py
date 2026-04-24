"""Tests for the CLI override + batch modes in `app.pipeline.main`."""

from __future__ import annotations

import pytest
from app.pipeline import _apply_overrides, _coerce_override_value


def test_coerce_int_float_bool_string() -> None:
    assert _coerce_override_value("12") == 12
    assert _coerce_override_value("0.75") == pytest.approx(0.75)
    assert _coerce_override_value("true") is True
    assert _coerce_override_value("False") is False
    assert _coerce_override_value("None") is None
    assert _coerce_override_value("hsv") == "hsv"


def test_coerce_comma_tuple() -> None:
    assert _coerce_override_value("1.4,1.1,0.6") == (
        pytest.approx(1.4),
        pytest.approx(1.1),
        pytest.approx(0.6),
    )


def test_apply_overrides_leaves_original_untouched() -> None:
    base = {"stretch": {"stretch": 25.0, "white_percentile": 99.9}}
    patched = _apply_overrides(base, ["stretch.stretch=30"])
    assert patched["stretch"]["stretch"] == 30
    # Original dict is unchanged.
    assert base["stretch"]["stretch"] == 25.0


def test_apply_overrides_multiple_stages() -> None:
    base = {
        "stretch": {"stretch": 25.0},
        "curves": {"contrast": 0.6},
    }
    patched = _apply_overrides(
        base,
        ["stretch.stretch=30", "curves.contrast=0.75", "curves.channel_gains=1.4,1.1,0.6"],
    )
    assert patched["stretch"]["stretch"] == 30
    assert patched["curves"]["contrast"] == pytest.approx(0.75)
    assert patched["curves"]["channel_gains"] == (
        pytest.approx(1.4),
        pytest.approx(1.1),
        pytest.approx(0.6),
    )


def test_apply_overrides_adds_new_stage() -> None:
    base = {"stretch": {"stretch": 25.0}}
    patched = _apply_overrides(base, ["curves.contrast=0.5"])
    assert patched["curves"] == {"contrast": pytest.approx(0.5)}


def test_apply_overrides_rejects_malformed() -> None:
    base = {"stretch": {}}
    with pytest.raises(ValueError, match="stage.param=value"):
        _apply_overrides(base, ["no_equals_sign"])
    with pytest.raises(ValueError, match="stage.param"):
        _apply_overrides(base, ["nodot=1"])


def test_apply_overrides_rejects_non_dict_stage() -> None:
    base = {"bad_stage": 42}
    with pytest.raises(ValueError, match="not a stage params dict"):
        _apply_overrides(base, ["bad_stage.param=1"])
