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
    # Main stretch is gentler than v1 because the v2 star-split path
    # (below) applies a second stretch pass to the starless only.
    # Aggressive stretch that would bloat stars in v1; the star-split
    # (below) separates stars from diffuse structure so this level of
    # arcsinh compression can lift faint filaments without halo ringing.
    "stretch": {
        "black_percentile": 1.5,
        "white_percentile": 99.95,
        "stretch": 26.0,
    },
    # Heavy chroma_blur flattens remaining red/blue speckle in the
    # mid-brightness diffuse region. BM3D sigma is fixed because the
    # MAD estimator bleeds real nebula structure into its "noise".
    # Sharpen stays gentle — we don't have star-removal and we don't
    # want to re-amplify what we just denoised.
    "bm3d_denoise": {"sigma": 0.15, "strength": 1.0, "chroma_blur": 10.0},
    "sharpen": {"radius": 1.6, "amount": 0.25},
    # Low contrast preserves gradient; moderate saturation because
    # the starless layer already concentrates all the chroma — applying
    # 1.35x saturation on a star-free, double-stretched nebula
    # over-saturates the noise colour.
    "curves": {"contrast": 0.30, "saturation": 1.35},
    # v2: star/starless split. radius=7 (15 px window) removes the
    # Seestar PSF stars cleanly while preserving Veil filaments (~10 px
    # wide). With the split in place the main stretch no longer bloats
    # every star — we get nebula-favourable stretch aggressiveness and
    # stars stay point-like. Stars are recombined via screen blend after
    # all processing, so they carry their natural colour into the final.
    "stars": {"radius": 7},
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
    # Stretch + white-point combo that lifts the disk out of the noise
    # floor while still preserving core gradient: stretch=20 is aggressive
    # enough to make the halo visible; white_percentile=99.995 keeps the
    # very peak (M81 nucleus) out of clipping so it renders as a warm
    # yellow-tan rather than a white blob.
    "stretch": {
        "black_percentile": 0.1,
        "white_percentile": 99.995,
        "stretch": 20.0,
    },
    # Pre-stretch chroma smooth: subtract only *very* slow chromatic
    # variation (sigma=20 → touches blobs larger than ~40 px). M81 and
    # M82 are 60-80 px across, so their natural disk colour survives the
    # high-pass intact; only true LP/Bayer blobs on the larger scale get
    # flattened.
    "color": {
        **DEFAULT["color"],
        "pre_stretch_chroma_smooth": 20.0,
    },
    # Gentler chroma_blur so the galaxy's subtle blue-disk / yellow-core
    # colour separation isn't flattened. Full BM3D luma strength still
    # denoises the halo.
    "bm3d_denoise": {"sigma": None, "strength": 1.8, "chroma_blur": 2.5},
    "sharpen": {"radius": 1.2, "amount": 0.35},
    # Higher saturation so the warm core and cooler disk read clearly at
    # the galaxy's small angular size; lower contrast so the halo keeps
    # its gradient into the sky.
    "curves": {"contrast": 0.30, "saturation": 1.65},
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
