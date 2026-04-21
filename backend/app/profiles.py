"""Per-stage parameter profiles for the v1 pipeline.

Each profile is a mapping from stage name to keyword arguments forwarded to
that stage's `process()` call. Unspecified stages use the function defaults.

v1 pipeline stages (in order):
    background, color, stretch, bm3d_denoise, sharpen, curves

**v1 tuning note.** Because v1 has no star-removal step, every operation
(stretch, denoise, sharpen, curves) hits the stars too. Aggressive stretch
bloats star PSFs; aggressive sharpen ringing-halos them; aggressive curves
and saturation push star cores to clipping and desaturate colored stars.
The three target profiles below are therefore tuned **conservatively**
relative to what a v2 star-aware pipeline could get away with. When v2
lands, bump these numbers up for the starless path.

Ordering by aggressiveness (most -> least): nebula > galaxy > cluster.
Clusters are almost entirely stars, so every knob is dialed down.
"""
from __future__ import annotations

from typing import Any, Dict

StageParams = Dict[str, Any]
Profile = Dict[str, StageParams]


# Safe middle-of-the-road profile. Used as the `default` and as a base for
# the three content-tuned profiles below.
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
        "green_clip": 0.5,
        "pre_stretch_chroma_smooth": 0.0,
    },
    "stretch": {
        "black_percentile": 0.1,
        "stretch": 20.0,
    },
    "bm3d_denoise": {
        "sigma": None,
        "strength": 1.0,
        "chroma_blur": 0.0,
    },
    "sharpen": {
        "radius": 1.5,
        "amount": 0.35,
    },
    "curves": {
        "contrast": 0.5,
        "saturation": 1.15,
    },
}


# Nebulae: faint extended emission, wants the strongest stretch we dare to
# apply. Background removal is stronger (larger grid, more smoothing) since
# nebulosity is the signal and shouldn't be fit away.
NEBULA: Profile = {
    **DEFAULT,
    "background": {
        "grid": 32,
        "sigma": 2.0,
        "iters": 6,
        "smoothing": 1.5,
        "downscale": 8,
    },
    # SCNR the Seestar's RGGB green cast, but only partially: 0.7 is a
    # soft blend so OIII's green-cyan component survives. A hard clip
    # (0.9) over-suppresses OIII and turns the Veil purple instead of
    # the red-Ha + blue-teal-OIII combo that community images show.
    # pre_stretch_chroma_smooth=80: only remove very-slow chromatic
    # variation (LP residuals / Bayer blobs). The Veil's narrow filaments
    # are ~10 px wide so they fall well inside the high-pass band and
    # keep their colour; 200+ px sky blobs get flattened to neutral.
    "color": {
        "dark_percentile": 25.0,
        "mid_low": 40.0,
        "mid_high": 80.0,
        "wb_strength": 1.0,
        "green_clip": 0.7,
        "pre_stretch_chroma_smooth": 80.0,
    },
    # Aggressive black point clips the mid-dark chroma speckle to pure
    # black; filaments survive above it. stretch is moderate because
    # once the floor is clipped we don't need to compress as much.
    "stretch": {
        "black_percentile": 1.5,
        "white_percentile": 99.95,
        "stretch": 22.0,
    },
    # Heavy chroma_blur flattens remaining red/blue speckle in the
    # mid-brightness diffuse region. BM3D sigma is fixed because the
    # MAD estimator bleeds real nebula structure into its "noise".
    # Sharpen stays gentle — we don't have star-removal and we don't
    # want to re-amplify what we just denoised.
    "bm3d_denoise": {"sigma": 0.15, "strength": 1.0, "chroma_blur": 10.0},
    "sharpen": {"radius": 1.6, "amount": 0.25},
    # Low contrast preserves delicate gradient on the faint outer
    # filaments; higher saturation lifts Ha (red) and OIII (cyan) so
    # the real nebula stands out from the grey-ish noise speckle.
    "curves": {"contrast": 0.30, "saturation": 1.35},
}


# Galaxies: bright core with a faint halo. A middle-ground stretch keeps
# the core from clipping while lifting the disk. Slightly tighter sharpen
# radius favors dust-lane detail over halo softness.
GALAXY: Profile = {
    **DEFAULT,
    # Finer grid than default: M81-class targets have broadband color
    # gradients (LP residuals, amp glow) at scales of a few hundred
    # pixels. grid=20 fits those cleanly; grid must still be coarser
    # than the galaxy itself so the fit doesn't eat halo signal.
    "background": {
        "grid": 20,
        "sigma": 2.5,
        "iters": 5,
        "smoothing": 1.2,
        "downscale": 8,
    },
    # Gentler stretch + high white_percentile to preserve the M81-class
    # core: stretch=14 keeps mid-tones where they are instead of pushing
    # them into near-clipping; white_percentile=99.97 ensures the bright
    # core normalises to slightly under 1.0 so arcsinh still produces
    # gradient (yellow-tan core highlights, not a white blob).
    "stretch": {
        "black_percentile": 0.1,
        "white_percentile": 99.97,
        "stretch": 14.0,
    },
    # Pre-stretch chroma smooth: Gaussian-blur the per-channel chroma
    # deviation in linear space before arcsinh amplifies it. Sigma=50 px
    # targets the 100-200 px chromatic blobs left by LP gradient residuals
    # and Bayer demosaic. Galaxy luma structure (spiral arms, dust lanes)
    # is unaffected because we only blur the chroma, not the luma.
    "color": {
        **DEFAULT["color"],
        "pre_stretch_chroma_smooth": 50.0,
    },
    # Galaxy sky noise after stretch is meaningful; boost denoise so the
    # halo doesn't drown in chroma speckle.
    "bm3d_denoise": {"sigma": None, "strength": 1.8, "chroma_blur": 7.0},
    "sharpen": {"radius": 1.2, "amount": 0.35},
    # Lower contrast so the bright core keeps gradient instead of being
    # pushed into a white blob; saturation 1.40 keeps colour punch without
    # needing the S-curve to drive it.
    "curves": {"contrast": 0.35, "saturation": 1.40},
}


# Clusters: mostly stars. Every operation that hits a star core can either
# bloat it (stretch) or ring it (sharpen) or desaturate it (saturation).
# This profile is the gentlest of the three on every knob.
CLUSTER: Profile = {
    **DEFAULT,
    "background": {
        "grid": 24,
        "sigma": 2.5,
        "iters": 5,
        "smoothing": 1.0,
        "downscale": 8,
    },
    # Cluster cores are dense — individual bright stars blow out first.
    # stretch=11 + white_percentile=99.97 keeps the core resolved into
    # stars rather than a single saturated blob.
    "stretch": {
        "black_percentile": 0.2,
        "white_percentile": 99.97,
        "stretch": 11.0,
    },
    # Clusters have dark sky with tight star PSFs; a mild chroma blur
    # cleans sky speckle without touching star cores.
    "bm3d_denoise": {"sigma": None, "strength": 1.2, "chroma_blur": 1.5},
    "sharpen": {"radius": 1.0, "amount": 0.15},
    # Very mild S-curve so the packed cluster core keeps star-to-star
    # gradient. Saturation 1.25 lifts the yellow-white-blue star
    # population without hue-clipping.
    "curves": {"contrast": 0.15, "saturation": 1.25},
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
