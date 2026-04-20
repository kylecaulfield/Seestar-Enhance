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
