"""Image-content auto-classification.

Given a debayered RGB image (float32 in [0, 1]) from `io_fits.load_fits`,
decide whether it is best processed as a diffuse nebula, a galaxy, or a
star cluster.

Three cheap metrics feed a small set of hand-tuned thresholds:

1. Median brightness of non-star pixels — a sky-background proxy. Computed
   on the luminance channel after masking out a dilated set of locally
   bright pixels so stars and their halos don't skew the median.
2. Star density per megapixel — number of local-maximum pixels above a
   MAD-based high threshold, normalized by image area. High-density
   fields (globular clusters) score an order of magnitude above the
   field-star counts of a typical galaxy or nebula frame.
3. Largest connected bright region area, as a fraction of the image —
   using a moderate MAD threshold. Diffuse nebulosity forms one very
   large connected component; galaxy bulges form a moderate one; cluster
   frames rarely produce a single dominant component at this threshold.

Thresholds are tuned empirically on the three Seestar samples and are
intentionally coarse — the goal is to pick a profile, not to replace a
real classifier.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

import numpy as np
from scipy.ndimage import binary_dilation, label, maximum_filter

logger = logging.getLogger(__name__)

Classification = Literal["nebula", "galaxy", "cluster"]


class ClassifyMetrics(TypedDict):
    non_star_median: float
    star_density_per_mpix: float
    largest_bright_fraction: float


# Empirically tuned on Seestar S50 single frames + stacks.
# Nebula is picked first because a large extended region is the strongest
# signal (veil-nebula largest_bright_fraction >> 0.1, galaxy/cluster are
# below 0.03). Cluster falls out from high point-source density
# (M92 ~1400/MP, M81 galaxy ~985/MP) COMBINED with a dim sky — stacked
# nebulae like NGC 6888 can have star densities up to ~2000/MP but
# their non-star sky isn't dark, so we gate the cluster branch on
# a dim sky threshold too.
_BRIGHT_THRESH = 0.05  # on pre-stretched luma (lower = catches faint Ha)
_NEBULA_LARGEST_FRAC = 0.10
_CLUSTER_STAR_DENSITY = 1300.0
_CLUSTER_MAX_SKY_MEDIAN = 0.03  # above this, diffuse emission is present
_GALAXY_MIN_SKY_MEDIAN = 0.03   # galaxy halos lift median above "dark sky"
_GALAXY_MAX_SKY_MEDIAN = 0.08   # but not as high as nebula diffuse median


def _prestretch(luma: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] and apply a mild arcsinh so metric thresholds
    are comparable across Seestar frames with wildly different dynamic
    range (a bright galaxy and a dim nebula otherwise sit at very
    different absolute intensities in raw data).
    """
    black = float(np.percentile(luma, 5.0))
    white = float(np.percentile(luma, 99.95))
    if white <= black:
        return np.zeros_like(luma, dtype=np.float32)
    normalized = np.clip((luma - black) / (white - black), 0.0, 1.0)
    return (np.arcsinh(normalized * 25.0) / np.arcsinh(25.0)).astype(np.float32)


def _metrics(image: np.ndarray) -> ClassifyMetrics:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    luma = _prestretch(image.mean(axis=-1).astype(np.float32, copy=False))
    h, w = luma.shape
    mpix = (h * w) / 1.0e6

    med = float(np.median(luma))
    mad = float(np.median(np.abs(luma - med))) * 1.4826
    if mad < 1e-6:
        mad = 1e-6

    high_thresh = med + 8.0 * mad
    high_mask = luma > high_thresh

    # Star count via local maxima above the high threshold. A 7x7 window
    # comfortably separates adjacent Seestar stars (typical FWHM ~3 px).
    neigh = maximum_filter(luma, size=7)
    peaks = (luma >= neigh) & high_mask
    star_density = float(peaks.sum()) / max(mpix, 1e-6)

    # Background median excludes bright pixels plus a small dilation so star
    # halos don't bias the sky estimate upward.
    star_exclusion = binary_dilation(high_mask, iterations=3)
    non_star = ~star_exclusion
    if int(non_star.sum()) > 1024:
        non_star_median = float(np.median(luma[non_star]))
    else:
        non_star_median = med

    # Moderate threshold captures diffuse emission, not just star cores.
    # Uses a fixed level in pre-stretched space because nebular MAD is
    # inflated by the diffuse emission itself, so a MAD-scaled threshold
    # would perversely reject the very pixels we want to count.
    mod_mask = luma > _BRIGHT_THRESH
    if bool(mod_mask.any()):
        lbl, _ = label(mod_mask)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0  # background label
        largest_px = int(counts.max()) if counts.size > 1 else 0
    else:
        largest_px = 0
    largest_frac = largest_px / float(h * w)

    return {
        "non_star_median": non_star_median,
        "star_density_per_mpix": star_density,
        "largest_bright_fraction": largest_frac,
    }


def classify(image: np.ndarray) -> Classification:
    """Classify an RGB image as nebula, galaxy, or cluster.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3) with values in [0, 1], as
        produced by `stages.io_fits.load_fits`. The classifier runs on
        the raw debayered image before background/stretch so thresholds
        stay comparable across frames.

    Returns
    -------
    Literal["nebula", "galaxy", "cluster"]
    """
    m = _metrics(image)

    if m["largest_bright_fraction"] > _NEBULA_LARGEST_FRAC:
        result: Classification = "nebula"
    elif (
        m["star_density_per_mpix"] > _CLUSTER_STAR_DENSITY
        and m["non_star_median"] < _CLUSTER_MAX_SKY_MEDIAN
        and m["largest_bright_fraction"] > 0.005
    ):
        # Dense stars AND dark sky AND a visible dense core region →
        # globular cluster. The largest_bright_fraction gate is what
        # separates a real cluster (M92 core fills ~1.3% of the frame)
        # from a dense starfield overlaid on a faint nebula filament
        # (NGC 6960 Western Veil has only ~0.3% bright fraction).
        result = "cluster"
    elif _GALAXY_MIN_SKY_MEDIAN < m["non_star_median"] < _GALAXY_MAX_SKY_MEDIAN:
        # Moderately elevated sky median without a dominant bright region:
        # a galaxy with a halo (M81's halo lifts the median without
        # producing the large connected region that Rosette/Veil do).
        result = "galaxy"
    elif m["non_star_median"] > _GALAXY_MAX_SKY_MEDIAN:
        # High sky median, no one-dominant region: a dim diffuse nebula
        # dispersed across the frame.
        result = "nebula"
    elif m["star_density_per_mpix"] > _CLUSTER_STAR_DENSITY:
        # Dark sky + dense stars but no visible cluster core: a faint
        # nebula filament (Veil segments) through a dense starfield.
        result = "nebula"
    else:
        # Dark sky, moderate stars → sparsely-framed galaxy.
        result = "galaxy"

    logger.info(
        "classify: %s (non_star_median=%.4f, star_density_per_mpix=%.1f, "
        "largest_bright_fraction=%.4f)",
        result,
        m["non_star_median"],
        m["star_density_per_mpix"],
        m["largest_bright_fraction"],
    )
    return result
