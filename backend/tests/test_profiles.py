"""Tests for the profile registry."""

from __future__ import annotations

import pytest
from app import profiles


def test_default_profile_present() -> None:
    assert "default" in profiles.PROFILES


def test_all_profiles_cover_v1_stages() -> None:
    expected = {"background", "color", "stretch", "bm3d_denoise", "sharpen", "curves"}
    for name, profile in profiles.PROFILES.items():
        assert expected.issubset(profile.keys()), (
            f"profile {name} missing keys: {expected - profile.keys()}"
        )


def test_get_returns_a_dict() -> None:
    p = profiles.get("galaxy")
    assert isinstance(p, dict)
    assert "stretch" in p


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        profiles.get("no-such-profile")


# ---------- merge() utility ----------


def test_merge_inherits_unspecified_keys() -> None:
    parent = {"stretch": {"stretch": 20.0}, "curves": {"contrast": 0.5}}
    out = profiles.merge(parent, stretch={"stretch": 30.0})
    # stretch updated, curves inherited
    assert out["stretch"]["stretch"] == 30.0
    assert out["curves"]["contrast"] == 0.5


def test_merge_recursively_merges_nested_dicts() -> None:
    """The win over `{**PARENT, key: {...}}` — sibling keys survive."""
    parent = {
        "color": {"wb_strength": 1.0, "green_clip": 0.5, "ccm": "seestar_s50"},
    }
    # Override only one nested key.
    out = profiles.merge(parent, color={"green_clip": 0.0})
    assert out["color"]["green_clip"] == 0.0
    assert out["color"]["wb_strength"] == 1.0  # inherited
    assert out["color"]["ccm"] == "seestar_s50"  # inherited


def test_merge_none_disables_a_stage() -> None:
    """`None` is the explicit way to disable a parent's enabled stage."""
    parent = {"stars": {"radius": 7}, "curves": {"contrast": 0.5}}
    out = profiles.merge(parent, stars=None)
    assert out["stars"] is None
    assert out["curves"] == {"contrast": 0.5}


def test_merge_does_not_mutate_parent() -> None:
    parent = {"stretch": {"stretch": 20.0}, "curves": {"contrast": 0.5}}
    snapshot = {
        "stretch": dict(parent["stretch"]),
        "curves": dict(parent["curves"]),
    }
    profiles.merge(parent, stretch={"stretch": 99.0})
    assert parent["stretch"] == snapshot["stretch"]
    assert parent["curves"] == snapshot["curves"]


def test_merge_replaces_non_dict_value() -> None:
    """If the parent's value isn't a dict, override replaces it wholesale."""
    parent = {"radius": 7}
    out = profiles.merge(parent, radius=12)
    assert out["radius"] == 12


def test_merge_adds_new_top_level_key() -> None:
    parent = {"stretch": {"stretch": 20.0}}
    out = profiles.merge(parent, spcc={"min_matches": 20})
    assert out["spcc"] == {"min_matches": 20}
    assert out["stretch"]["stretch"] == 20.0


def test_merge_chains_for_multi_level_inheritance() -> None:
    """DEFAULT -> NEBULA -> NEBULA_WIDE is the actual production pattern."""
    base = {"color": {"wb_strength": 1.0, "green_clip": 0.5}}
    mid = profiles.merge(base, color={"green_clip": 0.85})
    leaf = profiles.merge(mid, color={"green_clip": 0.0})
    assert leaf["color"]["wb_strength"] == 1.0  # inherited from base
    assert leaf["color"]["green_clip"] == 0.0  # leaf override


def test_merge_handles_tuple_values() -> None:
    """channel_gains is a tuple — must replace, not merge."""
    parent = {"curves": {"channel_gains": (1.0, 1.0, 1.0)}}
    out = profiles.merge(parent, curves={"channel_gains": (1.5, 1.0, 0.5)})
    assert out["curves"]["channel_gains"] == (1.5, 1.0, 0.5)


# ---------- production profiles use merge() correctly ----------


def test_nebula_sub_profiles_inherit_default_color_keys() -> None:
    """The sibling-key-loss bug the merge() helper exists to prevent.

    `dark_percentile` is set on DEFAULT, untouched in NEBULA's color
    override, and untouched in NEBULA_WIDE's color override. With the
    old `{**PARENT, "color": {**PARENT["color"], "ccm": None}}` pattern
    you had to remember to spread the parent — the merge helper does
    it automatically.
    """
    for name in ("nebula_wide", "nebula_filament", "nebula_dominant"):
        p = profiles.get(name)
        assert p["color"]["dark_percentile"] == 25.0, f"{name} lost dark_percentile inheritance"
        assert p["color"]["mid_low"] == 40.0, f"{name} lost mid_low inheritance"


def test_nebula_wide_disables_stars_via_none() -> None:
    """`stars=None` survives the merge as `None`, not as an empty dict."""
    assert profiles.get("nebula_wide")["stars"] is None


def test_nebula_dominant_inherits_no_spcc() -> None:
    """NEBULA has no `spcc` key; NEBULA_DOMINANT shouldn't add one."""
    assert "spcc" not in profiles.get("nebula_dominant")
