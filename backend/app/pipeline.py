"""End-to-end v1 pipeline.

Order (classical only, no ML):
    io_fits -> classify -> background -> color -> stretch ->
    bm3d_denoise -> sharpen -> curves -> export.

Classification runs on the debayered-but-unprocessed image so its metrics
stay comparable across frames. The result selects one of the built-in
profiles (nebula / galaxy / cluster) unless the caller overrides it.

Run as a module:
    python -m app.pipeline <input.fits> <output.png>
                           [--profile NAME] [--no-auto] [-v]

Progress callback:
    run(..., progress=lambda stage, frac: ...)
  Each stage reports its name and a 0..1 completion value; the pipeline
  emits an additional frac=1.0 tick when the whole run finishes. This is
  the interface we'll wire to the web UI in phase 5.

v2 will insert star removal (after stretch) and replace bm3d_denoise with
an ML denoiser. See stages/stars.py and stages/ml_denoise.py for stubs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from app import profiles
from app.stages import (
    background,
    bm3d_denoise,
    classify,
    color,
    curves,
    export,
    sharpen,
    stars,
    stretch,
)
from app.stages.io_fits import load_fits

PathLike = Union[str, Path]
ProgressCallback = Callable[[str, float], None]

logger = logging.getLogger(__name__)


# Stages in execution order. Progress is reported as stage-starting
# (frac=0.0) and stage-finished (frac=1.0); intermediate fractions are
# reserved for long-running stages that learn to emit their own updates.
_STAGES = ("load", "classify", "background", "color", "stretch",
           "bm3d_denoise", "sharpen", "curves", "export")


def _noop_progress(stage: str, frac: float) -> None:  # noqa: ARG001
    return None


def run(
    input_fits: PathLike,
    output_png: PathLike,
    profile: Optional[str] = None,
    verbose: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> np.ndarray:
    """Run the full v1 pipeline.

    Parameters
    ----------
    input_fits : str | Path
        Input Seestar FITS file.
    output_png : str | Path
        Output 16-bit PNG file.
    profile : str | None
        Named profile from `app.profiles`. If None, the image is classified
        and the matching profile ("nebula" | "galaxy" | "cluster") is used.
    verbose : bool
        Mirror stage messages to stderr in addition to the logger.
    progress : Callable[[str, float], None] | None
        Optional callback invoked as (stage_name, fraction_complete). Called
        once with frac=0.0 at the start of each stage and once with frac=1.0
        at its end, plus a final ("done", 1.0).
    """
    cb: ProgressCallback = progress or _noop_progress

    def log(msg: str) -> None:
        logger.info(msg)
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    log(f"[1/{len(_STAGES)}] loading {input_fits}")
    cb("load", 0.0)
    img = load_fits(input_fits)
    cb("load", 1.0)

    log(f"[2/{len(_STAGES)}] classify")
    cb("classify", 0.0)
    if profile is None:
        selected = classify.classify(img)
        log(f"auto-selected profile: {selected}")
    else:
        selected = profile
        log(f"using requested profile: {selected}")
    params = profiles.get(selected)
    cb("classify", 1.0)

    log(f"[3/{len(_STAGES)}] background removal")
    cb("background", 0.0)
    img = background.process(img, **params.get("background", {}))
    cb("background", 1.0)

    log(f"[4/{len(_STAGES)}] color neutralization + white balance")
    cb("color", 0.0)
    img = color.process(img, **params.get("color", {}))
    cb("color", 1.0)

    log(f"[5/{len(_STAGES)}] stretch")
    cb("stretch", 0.0)
    img = stretch.process(img, **params.get("stretch", {}))
    cb("stretch", 1.0)

    # v2 star-split: for targets where star bloat is the main ceiling on
    # stretch aggressiveness (i.e. nebulae), split the frame into a
    # compact-star channel and an extended-structure channel, push the
    # starless much harder, then screen-blend the original stars back in.
    # Opt-in via profile param so galaxy and cluster profiles keep the
    # pure classical path.
    stars_params = params.get("stars")
    starless_stretch_params = params.get("starless_stretch")
    stars_only = None
    if stars_params is not None:
        log(f"[5a/{len(_STAGES)}] star/starless split")
        stars_only, img = stars.process(img, **stars_params)
        if starless_stretch_params is not None:
            log("[5b] additional stretch on starless")
            img = stretch.process(img, **starless_stretch_params)

    log(f"[6/{len(_STAGES)}] bm3d denoise (classical)")
    cb("bm3d_denoise", 0.0)
    img = bm3d_denoise.process(img, **params.get("bm3d_denoise", {}))
    cb("bm3d_denoise", 1.0)

    log(f"[7/{len(_STAGES)}] sharpen")
    cb("sharpen", 0.0)
    img = sharpen.process(img, **params.get("sharpen", {}))
    cb("sharpen", 1.0)

    log(f"[8/{len(_STAGES)}] curves")
    cb("curves", 0.0)
    img = curves.process(img, **params.get("curves", {}))
    cb("curves", 1.0)

    # Screen-blend the unmodified stars back onto the heavily-processed
    # starless. Must happen after curves so stars don't get double-boosted.
    if stars_only is not None:
        log("[8a] recombine starless + stars (screen blend)")
        img = stars.recombine(stars_only, img)

    log(f"[9/{len(_STAGES)}] export -> {output_png}")
    cb("export", 0.0)
    export.process(img, output_png)
    cb("export", 1.0)

    cb("done", 1.0)
    log(f"wrote {output_png}")
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline",
        description="Run the Seestar enhancement pipeline end-to-end (v1).",
    )
    parser.add_argument("input", help="input Seestar FITS file")
    parser.add_argument("output", help="output PNG file (16-bit)")
    parser.add_argument(
        "--profile",
        default=None,
        choices=sorted(profiles.PROFILES),
        help="parameter profile (default: auto-classify)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="disable auto-classify; use --profile or fall back to default",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    selected = args.profile
    if selected is None and args.no_auto:
        selected = "default"

    run(args.input, args.output, profile=selected, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
