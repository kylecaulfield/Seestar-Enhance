"""Per-stage parameter profiles for the v1 pipeline.

Each profile is a mapping from stage name to keyword arguments forwarded to
that stage's `process()` call. Unspecified stages use the function defaults.

v1 pipeline stages (in order):
    background, color, stretch, bm3d_denoise, sharpen, curves
"""
from __future__ import annotations

from typing import Any, Dict

StageParams = Dict[str, Any]
Profile = Dict[str, StageParams]


DEFAULT: Profile = {
    "background": {
        "grid": 24,
        "sigma": 2.5,
        "iters": 5,
        "smoothing": 1.0,
        "downscale": 8,
    },
    "color": {
        "dark_percentile": 25.0,
        "mid_low": 40.0,
        "mid_high": 80.0,
        "wb_strength": 1.0,
    },
    "stretch": {
        "black_percentile": 0.1,
        "stretch": 25.0,
    },
    "bm3d_denoise": {
        "sigma": None,
        "strength": 1.0,
    },
    "sharpen": {
        "radius": 1.5,
        "amount": 0.4,
    },
    "curves": {
        "contrast": 0.6,
        "saturation": 1.2,
    },
}


NEBULA: Profile = {
    **DEFAULT,
    "stretch": {"black_percentile": 0.05, "stretch": 40.0},
    "curves": {"contrast": 0.5, "saturation": 1.35},
    "bm3d_denoise": {"sigma": None, "strength": 1.1},
}


GALAXY: Profile = {
    **DEFAULT,
    "stretch": {"black_percentile": 0.1, "stretch": 20.0},
    "sharpen": {"radius": 1.2, "amount": 0.5},
    "curves": {"contrast": 0.7, "saturation": 1.2},
}


CLUSTER: Profile = {
    **DEFAULT,
    "stretch": {"black_percentile": 0.2, "stretch": 12.0},
    "bm3d_denoise": {"sigma": None, "strength": 0.7},
    "sharpen": {"radius": 1.0, "amount": 0.3},
    "curves": {"contrast": 0.4, "saturation": 1.1},
}


PROFILES: Dict[str, Profile] = {
    "default": DEFAULT,
    "nebula": NEBULA,
    "galaxy": GALAXY,
    "cluster": CLUSTER,
}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; choose from {sorted(PROFILES)}"
        )
    return PROFILES[name]
