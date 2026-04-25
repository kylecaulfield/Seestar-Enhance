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

**Sub-profiles use `merge()`** — `{**PARENT, "color": {**PARENT["color"],
"ccm": None}}` was the old pattern; it's verbose and easy to mis-type
(forgetting `**PARENT["color"]` silently wipes sibling params). The
`merge()` helper does a recursive dict merge so sub-profiles only specify
the keys they're actually changing. Explicit `None` at a top-level key
disables that stage outright.
"""

from __future__ import annotations

import copy
from typing import Any

StageParams = dict[str, Any]
Profile = dict[str, Any]


def merge(parent: Profile, **overrides: Any) -> Profile:
    """Recursively merge ``overrides`` onto a deep copy of ``parent``.

    Behaviour:
      * Top-level keys not in ``overrides`` are inherited from parent.
      * For keys that exist in BOTH and where both values are dicts,
        merge recursively — this is the win over ``{**PARENT, key: {...}}``,
        which silently replaces the whole sub-dict.
      * For all other override values (including ``None``, lists, tuples,
        scalars, or where parent's value isn't a dict), the override
        value REPLACES the parent value. ``None`` is the way to disable
        a stage that the parent has enabled.

    The parent is never mutated; the returned dict is a deep copy with
    overrides applied. Safe to call repeatedly without side effects.
    """
    out: Profile = copy.deepcopy(parent)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in out and isinstance(out[key], dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``.

    Same merge rule as ``merge`` but operates on plain dicts (no kwargs
    expansion). Used for nested merges within a profile sub-dict.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
NEBULA: Profile = merge(
    DEFAULT,
    # Cosmetic pre-pass strips hot pixels / cosmic-ray hits before the
    # RBF background fit samples them; without it the arcsinh below
    # amplifies each outlier into a vivid rainbow dot.
    cosmetic={"neighborhood": 3, "sigma": 6.0},
    background={
        "grid": 32,
        "sigma": 2.0,
        "iters": 6,
        "smoothing": 1.5,
        "downscale": 8,
    },
    # SCNR the Seestar's RGGB green cast partially (0.85): a hard
    # clip over-suppresses OIII and turns the Veil purple. Mahalanobis
    # WB rejects outliers in the RGB cloud — for emission nebulae that
    # means the red Ha pixels get rejected as "non-sky" and the sky
    # sets the white point, washing out red dominance. Plain per-
    # channel medians instead, with wb_strength=0.6 so the residual
    # red signal survives. pre_stretch_chroma_lowpass=15.0 smooths
    # pixel-scale chroma in linear space before arcsinh amplifies it
    # into rainbow speckle. Static Seestar S50 CCM pre-compensates
    # the sensor's IR leak in R + RGGB green dominance; runs before
    # any WB heuristic so WB has less to do. Deconvolution is OFF
    # for nebula by default — Richardson-Lucy amplifies shot noise on
    # faint-target stacks more than it helps; galaxy profile can opt in.
    color={
        "wb_strength": 0.6,
        "green_clip": 0.85,
        "pre_stretch_chroma_lowpass": 15.0,
        "mahalanobis_wb": False,
        "star_protect_percentile": 99.0,
        "ccm": "seestar_s50",
    },
    # black=8 clips mid-dark chroma speckle to pure black, filaments
    # survive above. stretch=24 + white_pct=99.2 is stack-friendly:
    # sky-crush via black, exclude stars via white, moderate arcsinh.
    # Main stretch gentler than v1 because the star-split below
    # applies a second stretch pass to the starless layer only.
    stretch={
        "black_percentile": 8.0,
        "white_percentile": 99.2,
        "stretch": 24.0,
    },
    # chroma_blur=120 is extreme by luma-denoise standards but right
    # for Seestar stacks: per-pixel chroma SNR is much worse than luma,
    # and real nebula colour is always spatially smooth over hundreds
    # of pixels. Edge-preserving keeps Ha→sky transitions sharp;
    # luma_sigma=0.08 is the "similar luma" tolerance.
    bm3d_denoise={
        "sigma": 0.25,
        "strength": 4.0,
        "chroma_blur": 120.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.08,
    },
    sharpen={"radius": 1.6, "amount": 0.25},
    # S-curve + moderate gains. The curve pushes sky toward zero, then
    # per-channel gains put R ahead of B for the Ha-dominant look.
    # saturation_mode="hsv" preserves per-pixel hue under saturation
    # boost — linear mode shifts hue on already-saturated pixels.
    curves={
        "contrast": 0.85,
        "saturation": 1.25,
        "channel_gains": (1.4, 0.95, 0.75),
        "saturation_mode": "hsv",
    },
    stars={"radius": 7},
)


# nebula_wide — wide diffuse emission that fills most of the frame
# (Rosette, Heart/Soul, North America) AND bicolor HOO targets like
# the Crescent (NGC 6888): Wolf-Rayet bubble with OIII in G + Hα in R.
# Lighter chroma_blur than NEBULA so internal dust-lane / bicolor
# structure survives; SCNR off so the green OIII channel isn't
# clipped; SPCC enabled with low strength.
NEBULA_WIDE: Profile = merge(
    NEBULA,
    # Static CCM OFF — its built-in R-cut/B-boost was meant for the
    # pre-SPCC pipeline and was compounding with SPCC's own
    # corrections (purple-cast result). green_clip=0.0 disables SCNR;
    # the standard nebula profile clips G toward max(R, B) but that
    # destroys the OIII signal on bicolor HOO targets.
    color={
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.0,
    },
    # Selected from a 35-iteration random search scored against the
    # local NGC 6888 reference; top candidates eye-picked for sky
    # cleanliness + natural star colour over pure colour-match.
    # stretch=27 + black_pct=12 + white_pct=99.9 lifts the arc into
    # visibility without pulling sky noise into the mid-tones the
    # way stretch=40+ did.
    stretch={
        "black_percentile": 12.0,
        "white_percentile": 99.9,
        "stretch": 27.0,
    },
    bm3d_denoise={
        "sigma": 0.22,
        "strength": 3.0,
        "chroma_blur": 40.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.06,
    },
    # contrast=0.50 keeps the S-curve gentle — aggressive contrast on
    # this moderate-stretch config tipped the arc into blown-out
    # whites. saturation=1.38 delivers strong arc colour without the
    # rainbow-speckle effect seen at saturation>1.5 in the search.
    # channel_gains (1.49, 1.11, 0.55): R lifted with mild G lift
    # (orange-tilt rather than pink) and B cut to remove the residual
    # purple cast. Combined with the curves-stage star-preserve taper
    # (502d705) so stars keep their natural colour at these gains.
    curves={
        "contrast": 0.50,
        "saturation": 1.38,
        "saturation_mode": "hsv",
        "channel_gains": (1.49, 1.11, 0.55),
    },
    # SPCC strength dialled back (0.7 -> 0.35) because SPCC calibrates
    # against stellar references that assume broadband continuum. On
    # an OIII-dominant bubble the G channel carries real emission
    # that SPCC would read as "excess green sensor bias" and try to
    # suppress, which would gut the teal tones.
    spcc={
        "min_matches": 20,
        "strength": 0.35,
        "solar_reference_bp_rp": None,
    },
    # Star-split DISABLED. The median-based star detector treats the
    # Crescent's bright arc as "star-like" and preferentially pulls G
    # out of the starless channel — R/G jumped 1.11 -> 2.15 post-split
    # in the trace, killing arc colour. Keeping stars in the frame
    # preserves the arc's intrinsic R≈G≈B ratio so the downstream
    # saturation boost can pull out bicolor tones.
    stars=None,
)


# nebula_filament — narrow filament structures (Veil segments, Bubble)
# covering only a small fraction of the frame. Most pixels are sky,
# so we can afford to crush the sky to pure black and push the
# filament hard.
# nebula_filament — narrow filament structures (Veil segments, Bubble)
# covering only a small fraction of the frame. Most pixels are sky,
# so we can afford to crush the sky to pure black and push the
# filament hard. Same SPCC + CCM rationale as NEBULA_WIDE.
NEBULA_FILAMENT: Profile = merge(
    NEBULA,
    color={
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.25,
    },
    stretch={
        "black_percentile": 15.0,
        "white_percentile": 99.2,
        "stretch": 28.0,
    },
    bm3d_denoise={
        "sigma": 0.28,
        "strength": 4.5,
        "chroma_blur": 180.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.10,
    },
    # Saturation 1.0 — SPCC + stretch already supply enough chroma
    # for Veil-class targets. Saturation > 1 on top over-saturates
    # the filaments and starts showing chroma noise in the sky.
    curves={
        "contrast": 1.0,
        "saturation": 1.0,
        "saturation_mode": "hsv",
    },
    # strength=0.25 — filament scenes are dominated by sparse
    # starfield pixels, and SPCC's gains end up ±20 %-clamped on
    # the B channel for every filament sample in our set. Applying
    # even the clamped value at strength 0.7 tinted the filament
    # purple. 0.25 keeps a useful colour nudge without over-blueing.
    # `solar_reference_bp_rp=None` means use the frame's median
    # catalogue star colour as the neutral reference, auto-correcting
    # for dust-extinction reddening of Milky Way stars.
    spcc={
        "min_matches": 20,
        "strength": 0.25,
        "solar_reference_bp_rp": None,
    },
    # Smaller radius so the filament's ~5-8 px narrow structure isn't
    # caught by the median filter as a "star" — keeps it in the
    # starless layer where it can be stretched aggressively.
    stars={"radius": 9},
)


# nebula_dominant — emission nebula that fills almost the entire frame
# (Rosette, Heart/Soul, sometimes North America / California). These
# targets have no useful "sky" region for SPCC's star photometry to
# calibrate against: every bright-region measurement is contaminated
# by the nebula itself, and crushing the sky-shoulder eats the
# diffuse outer emission. This profile therefore:
#
#   - runs the static Seestar S50 CCM (well-tuned to this sensor's
#     biases already);
#   - skips SPCC entirely — no `"spcc"` key, so the stage is a no-op;
#   - keeps the generic NEBULA's sky-floor and stretch (black_pct
#     modest, not the wide-nebula value of 25) so the nebula's outer
#     halo survives;
#   - keeps a moderate saturation so the natural Ha dominance shows.
#
# Classifier routes here when largest_bright_fraction > 0.6.
NEBULA_DOMINANT: Profile = merge(
    NEBULA,
    # CCM OFF: the static Seestar CCM trims R and lifts B to correct
    # for the sensor's IR leak, but on an Ha-dominated frame that
    # trims the actual emission.
    # WB OFF (wb_strength=0.0): the mid-tone-median WB band on this
    # target is ~80 % nebula, so WB tries to flatten the Ha emission
    # to "neutral" grey — exactly backwards of what we want.
    # SCNR moderate (0.4): tame RGGB green bias without clipping G
    # all the way to (R+B)/2 (would leave a pink-violet cast).
    color={
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.4,
    },
    # Gentler than NEBULA_WIDE's black=25 crush (which would eat the
    # frame-filling emission). stretch=24 + white_pct=99.80 lifts the
    # Rosette's faint mid-nebula luma into visible range without
    # blowing the brightest knots; black_pct=1.0 pulls a little extra
    # faint signal off the floor for the outer petals.
    stretch={
        "black_percentile": 1.0,
        "white_percentile": 99.80,
        "stretch": 24.0,
    },
    # Frame-filling nebulae have no identifiable "sky" so chroma
    # noise can't be separated from signal by masking. The 150-px
    # bilateral blur flattens rainbow speckle without flattening
    # real emission structure (which is spatially smooth over
    # hundreds of pixels).
    bm3d_denoise={
        "sigma": 0.25,
        "strength": 3.5,
        "chroma_blur": 150.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.08,
    },
    # channel_gains (1.28, 1.02, 0.68): softer R-bias + slight G
    # lift to nudge Rosette toward warm brown-red instead of vivid
    # pink; saturation=0.92 pulls back the hot pink without
    # stripping dust-lane tonality.
    curves={
        "contrast": 0.60,
        "saturation": 0.92,
        "saturation_mode": "hsv",
        "channel_gains": (1.28, 1.02, 0.68),
    },
    stars={"radius": 7},
    # SPCC is intentionally NOT included — frame-filling nebulae have
    # no star-only region for the photometric fit to calibrate
    # against. The merge utility leaves the parent NEBULA's `spcc`
    # alone if the parent has one (it doesn't), so SPCC stays off.
)


# Galaxies: bright core with a faint halo. A middle-ground stretch keeps
# the core from clipping while lifting the disk. Slightly tighter sharpen
# radius favors dust-lane detail over halo softness.
GALAXY: Profile = merge(
    DEFAULT,
    # Finer grid than default: M81-class targets have broadband color
    # gradients (LP residuals, amp glow) at scales of a few hundred
    # pixels. grid=20 fits those cleanly; grid must still be coarser
    # than the galaxy itself so the fit doesn't eat halo signal.
    background={
        "grid": 20,
        "sigma": 2.5,
        "iters": 5,
        "smoothing": 1.2,
        "downscale": 8,
    },
    # Stretch lifts the disk out of the noise floor while
    # white_percentile=99.997 keeps the M81 nucleus from clipping to
    # pure white; note we intentionally DO NOT use the v2 star-split on
    # this profile because median-based removal steals the bright
    # galactic nucleus (classifies it as a "star"). The nucleus needs
    # to travel through the saturation + stretch steps for the warm-
    # yellow core colour to render; splitting it off re-adds it
    # unmodified.
    stretch={
        "black_percentile": 0.1,
        "white_percentile": 99.997,
        "stretch": 23.0,
    },
    # Pre-stretch chroma smooth: subtract only *very* slow chromatic
    # variation (sigma=20 → touches blobs larger than ~40 px). M81 and
    # M82 are 60-80 px across so disk colour survives the high-pass
    # intact; only LP/Bayer blobs at the larger scale get flattened.
    color={"pre_stretch_chroma_smooth": 20.0},
    # Gentler chroma_blur so the galaxy's subtle blue-disk / yellow-
    # core colour separation isn't flattened. Full BM3D luma strength
    # still denoises the halo.
    bm3d_denoise={"sigma": None, "strength": 1.8, "chroma_blur": 2.5},
    sharpen={"radius": 1.2, "amount": 0.35},
    # Higher saturation so the warm core reads as yellow-orange at the
    # galaxy's small angular size; contrast kept moderate so the halo
    # keeps its gradient into the sky.
    curves={"contrast": 0.35, "saturation": 1.85},
)


# Clusters: mostly stars. Every operation that hits a star core can
# either bloat it (stretch) or ring it (sharpen) or desaturate it
# (saturation). This profile is the gentlest of the three on every knob.
CLUSTER: Profile = merge(
    DEFAULT,
    background={
        "grid": 24,
        "sigma": 2.5,
        "iters": 5,
        "smoothing": 1.0,
        "downscale": 8,
    },
    # Cluster cores are dense — individual bright stars blow out
    # first. stretch=11 + white_pct=99.97 keeps the core resolved into
    # stars rather than a single saturated blob.
    stretch={
        "black_percentile": 0.2,
        "white_percentile": 99.97,
        "stretch": 11.0,
    },
    # Clusters have dark sky with tight star PSFs; a mild chroma blur
    # cleans sky speckle without touching star cores.
    bm3d_denoise={"sigma": None, "strength": 1.2, "chroma_blur": 1.5},
    sharpen={"radius": 1.0, "amount": 0.15},
    # Very mild S-curve so the packed cluster core keeps star-to-star
    # gradient. Saturation 1.25 lifts yellow-white-blue stars without
    # hue-clipping.
    curves={"contrast": 0.15, "saturation": 1.25},
)


PROFILES: dict[str, Profile] = {
    "default": DEFAULT,
    "nebula": NEBULA,
    "nebula_wide": NEBULA_WIDE,
    "nebula_filament": NEBULA_FILAMENT,
    "nebula_dominant": NEBULA_DOMINANT,
    "galaxy": GALAXY,
    "cluster": CLUSTER,
}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]
