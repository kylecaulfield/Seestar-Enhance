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
    # Cosmetic pre-pass strips hot pixels / cosmic-ray hits before the
    # RBF background fit samples them; without it the arcsinh below
    # amplifies each outlier into a vivid rainbow dot.
    "cosmetic": {"neighborhood": 3, "sigma": 6.0},
    "background": {
        "grid": 32,
        "sigma": 2.0,
        "iters": 6,
        "smoothing": 1.5,
        "downscale": 8,
    },
    # Deconvolution is OFF for nebula by default. Richardson-Lucy
    # amplifies shot noise and on faint-target stacks the amplified
    # chroma noise outweighs the modest star-PSF tightening. The
    # stage is wired and tested — galaxy profile can opt in.
    # SCNR the Seestar's RGGB green cast, but only partially: 0.7 is a
    # soft blend so OIII's green-cyan component survives. A hard clip
    # (0.9) over-suppresses OIII and turns the Veil purple instead of
    # the red-Ha + blue-teal-OIII combo that community images show.
    # pre_stretch_chroma_smooth=80: only remove very-slow chromatic
    # variation (LP residuals / Bayer blobs). The Veil's narrow filaments
    # are ~10 px wide so they fall well inside the high-pass band and
    # keep their colour; 200+ px sky blobs get flattened to neutral.
    # pre_stretch_chroma_smooth=0 for nebula: a high-pass-chroma filter
    # would flatten the very thing we want to see (Ha / OIII emission
    # spanning most of the frame on wide targets like Rosette / Crescent /
    # Veil).
    # pre_stretch_chroma_lowpass=3.0 smooths pixel-scale chroma noise
    # in linear space before the arcsinh stretch amplifies it into
    # visible rainbow speckle. Nebula features are spatially much
    # larger than 3px so their colour survives; pure per-pixel chroma
    # variance is averaged down ~3x, saving ~3x amplification downstream.
    # Mahalanobis WB rejects outliers in the RGB cloud, which for an
    # emission nebula means the red Ha pixels get rejected as non-sky,
    # and the sky-coloured pixels set the white point. Net effect: the
    # nebula's natural red dominance gets washed out. Use plain per-
    # channel medians for nebulae instead. Lower wb_strength so the
    # residual red signal survives the channel normalisation.
    "color": {
        "dark_percentile": 25.0,
        "mid_low": 40.0,
        "mid_high": 80.0,
        "wb_strength": 0.6,
        "green_clip": 0.85,
        "pre_stretch_chroma_smooth": 0.0,
        "pre_stretch_chroma_lowpass": 15.0,
        "mahalanobis_wb": False,
        "star_protect_percentile": 99.0,
        # Seestar S50 CCM pre-compensates for the sensor's known
        # IR leak into R and the RGGB green dominance. Applied before
        # any WB heuristic so the WB has less work to do.
        "ccm": "seestar_s50",
    },
    # Aggressive black point clips the mid-dark chroma speckle to pure
    # black; filaments survive above it. stretch is moderate because
    # once the floor is clipped we don't need to compress as much.
    # Main stretch is gentler than v1 because the v2 star-split path
    # (below) applies a second stretch pass to the starless only.
    # Aggressive stretch that would bloat stars in v1; the star-split
    # (below) separates stars from diffuse structure so this level of
    # arcsinh compression can lift faint filaments without halo ringing.
    # Stack-friendly stretch: sky-crush via black, exclude stars via
    # white, moderate arcsinh factor. The surviving noise band is
    # dominated by chroma, addressed by the very-large chroma_blur below.
    "stretch": {
        "black_percentile": 8.0,
        "white_percentile": 99.2,
        "stretch": 24.0,
    },
    # chroma_blur=120 is extreme by luma-denoise standards but appropriate
    # for Seestar stacked frames: the per-pixel chroma SNR is much worse
    # than the luma SNR, and real nebula colour is always spatially
    # smooth over hundreds of pixels. Sigma=120 averages the chroma
    # across ~240-pixel neighbourhoods, leaving luma detail sharp and
    # the nebula's Ha / OIII colour intact.
    # Aggressive denoise + big chroma_blur: Seestar single-stack sky
    # residuals produce enough chroma noise at our stretch levels that
    # moderate settings leave the frame speckled. strength=3 BM3D and
    # chroma_blur=100 together flatten the sky to near-neutral while
    # preserving nebula luma.
    # Edge-preserving chroma smoothing: the isotropic 200-px Gaussian
    # blurred the nebula's red into the surrounding sky (pink halo).
    # `chroma_edge_aware=True` weights the chroma-average by local
    # luma variance so smoothing stops at luma edges. `luma_sigma=0.05`
    # is the tolerance for "similar luma" — pixels whose luma differs
    # by more than that don't mix chroma.
    "bm3d_denoise": {
        "sigma": 0.25,
        "strength": 4.0,
        "chroma_blur": 120.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.08,
    },
    "sharpen": {"radius": 1.6, "amount": 0.25},
    # CLAHE is OFF for nebula by default. On noisy-sky stacked data,
    # local-tile histogram equalisation amplifies sky chroma noise
    # more than it lifts nebula structure. Available for opt-in on
    # cleaner galaxy data where the background noise floor is lower.
    # S-curve + moderate gains. The curve pushes sky toward zero, then
    # per-channel gains put R ahead of B for the Ha-dominant look. Gains
    # are moderate (1.25/0.95/0.80) so any residual R on the sky after
    # the S-curve stays visually-neutral instead of tinting pink.
    # saturation_mode="hsv": convert to HSV for the saturation scale.
    # Hue-preserving — a nebula pixel stays at its original hue, just
    # more saturated. Linear saturation (default) can shift hue on
    # already-saturated pixels, which is visible on Ha-rich regions.
    "curves": {
        "contrast": 0.85,
        "saturation": 1.25,
        "channel_gains": (1.4, 0.95, 0.75),
        "saturation_mode": "hsv",
    },
    "stars": {"radius": 7},
}


# nebula_wide — wide diffuse emission that fills most of the frame
# (Rosette, Heart/Soul, North America). Key differences from the
# generic NEBULA profile: lighter chroma_blur so internal dust-lane
# structure survives, lower black_percentile since there's no "dark
# sky" region to crush, slightly reduced R gain (the whole frame is
# already red — less headroom before posterisation).
NEBULA_WIDE: Profile = {
    **NEBULA,
    # SPCC is doing the sensor calibration on this profile, so we
    # turn OFF the static Seestar CCM and drop most of the heuristic
    # WB. Leaving both active compounds three calibrations on top of
    # each other and the result swings way too blue.
    "color": {
        **NEBULA["color"],
        # Static CCM OFF — its built-in R-cut/B-boost was meant for the
        # pre-SPCC pipeline and was compounding with SPCC's own
        # corrections (purple-cast result). With SPCC now running with
        # the G2V solar reference and clamped gains, it can do the
        # whole sensor calibration on its own.
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.20,
    },
    "stretch": {
        "black_percentile": 25.0,
        "white_percentile": 99.9,
        "stretch": 18.0,
    },
    "bm3d_denoise": {
        "sigma": 0.22,
        "strength": 3.0,
        "chroma_blur": 40.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.06,
    },
    # contrast=0.70: middle ground between 0.95 (blew the arc to
    # white) and 0.55 (left it nearly invisible). Combined with
    # white_percentile=99.9 in the stretch stage the arc keeps its
    # colour gradient — strong red but not fully clipped.
    "curves": {
        "contrast": 0.70,
        "saturation": 1.0,
        "saturation_mode": "hsv",
    },
    # strength=0.45 — apply 45% of SPCC's fitted gain. Full strength
    # (1.0) on broadband Seestar data pushes R too hard because Gaia's
    # RP band is wider / extends further into the IR than a camera's
    # red filter. Dialling back lets us keep the SPCC star-match
    # direction without overcooking the result.
    "spcc": {
        "min_matches": 20,
        "strength": 0.7,
        # `solar_reference_bp_rp=None` means "use the median catalogue
        # star colour in this frame as the neutral reference". Auto-
        # corrects for dust-extinction reddening — Milky Way stars are
        # systematically reddened by intervening dust, and an absolute
        # G2V reference reads that reddening as a sensor R-bias and
        # over-corrects to blue.
        "solar_reference_bp_rp": None,
    },
    # Wide nebulae don't benefit much from star split — the diffuse
    # signal is too embedded in the star field for median-based
    # separation to help cleanly — but the split also doesn't hurt,
    # so we keep the same radius as the generic nebula profile.
    "stars": {"radius": 7},
}


# nebula_filament — narrow filament structures (Veil segments, Bubble)
# covering only a small fraction of the frame. Most pixels are sky,
# so we can afford to crush the sky to pure black and push the
# filament hard.
NEBULA_FILAMENT: Profile = {
    **NEBULA,
    # Same idea as NEBULA_WIDE — SPCC handles colour calibration, so
    # drop the static CCM, WB, and post-stretch channel_gains.
    "color": {
        **NEBULA["color"],
        # Static CCM OFF here too — see NEBULA_WIDE for rationale.
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.25,
    },
    "stretch": {
        "black_percentile": 15.0,
        "white_percentile": 99.2,
        "stretch": 28.0,
    },
    "bm3d_denoise": {
        "sigma": 0.28,
        "strength": 4.5,
        "chroma_blur": 180.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.10,
    },
    # Saturation 1.0 — SPCC + stretch already supply enough chroma
    # for Veil-class targets. Saturation > 1 on top over-saturates
    # the filaments and starts showing chroma noise in the sky.
    "curves": {
        "contrast": 1.0,
        "saturation": 1.0,
        "saturation_mode": "hsv",
    },
    # strength=0.25 — filament scenes are dominated by sparse
    # starfield pixels, and SPCC's gains end up ±20 %-clamped on
    # the B channel for every filament sample in our set (the fit
    # wants to boost B further; clamping caps it). Applying even
    # the clamped value at strength 0.7 still tinted the filament
    # purple — the sparse real-signal pixels (the filament itself)
    # were swamped by the sky's full-frame B-boost. 0.25 keeps a
    # useful colour nudge without over-blueing the Veil.
    "spcc": {
        "min_matches": 20,
        "strength": 0.25,
        # `solar_reference_bp_rp=None` means "use the median catalogue
        # star colour in this frame as the neutral reference". Auto-
        # corrects for dust-extinction reddening — Milky Way stars are
        # systematically reddened by intervening dust, and an absolute
        # G2V reference reads that reddening as a sensor R-bias and
        # over-corrects to blue.
        "solar_reference_bp_rp": None,
    },
    # Smaller radius so the filament's ~5-8 px narrow structure isn't
    # caught by the median filter as a "star" — keeps it in the
    # starless layer where it can be stretched aggressively.
    "stars": {"radius": 9},
}


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
NEBULA_DOMINANT: Profile = {
    **NEBULA,
    "color": {
        **NEBULA["color"],
        # CCM OFF: the static Seestar CCM trims R and lifts B to
        # correct for the sensor's IR leak, but on an Ha-dominated
        # frame that trims the actual emission.
        #
        # WB OFF (wb_strength=0.0): the mid-tone-median WB band on
        # this target is ~80% nebula, so WB tries to flatten the Ha
        # emission into "neutral" grey — exactly backwards of what
        # we want. Leaving WB off keeps the raw per-channel balance
        # (R slightly above G above B), which the channel_gains
        # below amplify into proper Ha dominance.
        #
        # SCNR at moderate strength (0.4): the Seestar's RGGB green
        # dominance still needs attention — we just don't want SCNR
        # to clip G all the way down to (R+B)/2 (would leave a
        # pink-violet cast).
        "ccm": None,
        "wb_strength": 0.0,
        "green_clip": 0.4,
    },
    # Gentler than NEBULA_WIDE's black=25 crush (would eat the
    # frame-filling emission). stretch=22 with white_percentile=99.85
    # lifts the Rosette's faint mid-nebula luma into the visible range
    # without pushing the brightest knots into clipped saturation.
    # black_percentile=1.0 (vs default 2.0) pulls a little extra faint
    # signal off the floor — helps the outer petals show up.
    "stretch": {
        "black_percentile": 1.0,
        "white_percentile": 99.85,
        "stretch": 22.0,
    },
    # Frame-filling nebulae have nothing identifiable as "sky" so
    # chroma noise can't be separated from the signal by sky
    # masking. Crush it with a large chroma_blur — the nebula's
    # actual colour is spatially smooth over many hundreds of
    # pixels, so a 150-px bilateral blur flattens the rainbow
    # speckle without flattening real emission structure.
    "bm3d_denoise": {
        "sigma": 0.25,
        "strength": 3.5,
        "chroma_blur": 150.0,
        "chroma_edge_aware": True,
        "chroma_edge_luma_sigma": 0.08,
    },
    # Explicit channel_gains push R up and B down. Frame-filling
    # emission nebulae are H-alpha-dominated, so after removing the
    # static Seestar CCM (which was cutting R) and running the SCNR
    # green_clip (which flattens R and B toward the same mid-tone)
    # we land at pink. Boosting R / cutting B is the "this target
    # is genuinely red" hint that SPCC would give us on a frame
    # where it can run, but which we have no catalogue leverage
    # for here.
    "curves": {
        "contrast": 0.60,
        "saturation": 1.0,
        "saturation_mode": "hsv",
        # channel_gains (1.40, 1.00, 0.55): slightly stronger R bias
        # than the 1.35 / 0.60 middle-ground, paired with the boosted
        # stretch to keep the Rosette visible in red without tipping
        # back into the "blown-out orange" regime.
        "channel_gains": (1.40, 1.00, 0.55),
    },
    "stars": {"radius": 7},
    # No "spcc" key — SPCC stage is skipped for this profile.
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
    # Stretch lifts the disk out of the noise floor while
    # white_percentile=99.997 keeps the M81 nucleus from clipping to
    # pure white; note we intentionally DO NOT use the v2 star-split on
    # this profile because median-based removal steals the bright
    # galactic nucleus (classifies it as a "star"). The nucleus needs to
    # travel through the saturation + stretch steps for the warm-yellow
    # core colour to render; splitting it off re-adds it unmodified.
    "stretch": {
        "black_percentile": 0.1,
        "white_percentile": 99.997,
        "stretch": 23.0,
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
    # Higher saturation so the warm core reads as yellow-orange at the
    # galaxy's small angular size; contrast kept moderate so the halo
    # keeps its gradient into the sky.
    "curves": {"contrast": 0.35, "saturation": 1.85},
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
    "nebula_wide": NEBULA_WIDE,
    "nebula_filament": NEBULA_FILAMENT,
    "nebula_dominant": NEBULA_DOMINANT,
    "galaxy": GALAXY,
    "cluster": CLUSTER,
}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; choose from {sorted(PROFILES)}"
        )
    return PROFILES[name]
