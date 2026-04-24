"""Tests for the CLI override + batch modes in `app.pipeline.main`."""

from __future__ import annotations

import pytest
from app.pipeline import (
    _apply_overrides,
    _blas_thread_env,
    _coerce_override_value,
    _resolve_batch_jobs,
)


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


# ---------- --jobs / parallel batch ----------


def test_resolve_batch_jobs_defaults_to_one() -> None:
    assert _resolve_batch_jobs(1, n_work=4) == 1


def test_resolve_batch_jobs_auto_uses_cpu_count(monkeypatch) -> None:
    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 8)
    assert _resolve_batch_jobs(0, n_work=4) == 4  # capped at n_work
    assert _resolve_batch_jobs(0, n_work=20) == 8  # capped at cpu_count


def test_resolve_batch_jobs_clamps_to_one() -> None:
    assert _resolve_batch_jobs(-3, n_work=4) in (1, 4)  # non-positive -> auto
    assert _resolve_batch_jobs(100, n_work=2) == 2  # never exceed n_work


def test_blas_thread_env_throttles_proportionally(monkeypatch) -> None:
    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 16)
    env = _blas_thread_env(n_jobs=4)
    assert env["OMP_NUM_THREADS"] == "4"  # 16 // 4 = 4 threads per worker
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    # 8 workers -> 16 // 8 = 2 threads each
    env2 = _blas_thread_env(n_jobs=8)
    assert env2["OMP_NUM_THREADS"] == "2"


def test_blas_thread_env_floors_to_one() -> None:
    # More workers than cores -> 1 thread per worker.
    env = _blas_thread_env(n_jobs=1000)
    assert env["OMP_NUM_THREADS"] == "1"
