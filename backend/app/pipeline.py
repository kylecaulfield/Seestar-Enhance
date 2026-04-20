"""End-to-end v1 pipeline.

Order (classical only, no ML): io_fits -> background -> color -> stretch ->
bm3d_denoise -> sharpen -> curves -> export.

Run as a module:
    python -m app.pipeline <input.fits> <output.png> [--profile NAME] [-v]

v2 will insert star removal (after stretch) and replace bm3d_denoise with
an ML denoiser. See stages/stars.py and stages/ml_denoise.py for stubs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np

from app import profiles
from app.stages import (
    background,
    bm3d_denoise,
    color,
    curves,
    export,
    sharpen,
    stretch,
)
from app.stages.io_fits import load_fits

PathLike = Union[str, Path]


def run(
    input_fits: PathLike,
    output_png: PathLike,
    profile: str = "default",
    verbose: bool = False,
) -> np.ndarray:
    params = profiles.get(profile)

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    log(f"[1/7] loading {input_fits}")
    img = load_fits(input_fits)

    log("[2/7] background removal")
    img = background.process(img, **params.get("background", {}))

    log("[3/7] color neutralization + white balance")
    img = color.process(img, **params.get("color", {}))

    log("[4/7] stretch")
    img = stretch.process(img, **params.get("stretch", {}))

    # TODO(v2): insert stars.process(img) here to split stars/starless, then
    # run denoise on starless only and recombine with screen blend before
    # sharpen. See backend/app/stages/stars.py and ml_denoise.py stubs.

    log("[5/7] bm3d denoise (classical)")
    img = bm3d_denoise.process(img, **params.get("bm3d_denoise", {}))

    log("[6/7] sharpen")
    img = sharpen.process(img, **params.get("sharpen", {}))

    log("[7/7] curves + export")
    img = curves.process(img, **params.get("curves", {}))
    export.process(img, output_png)
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
        default="default",
        choices=sorted(profiles.PROFILES),
        help="parameter profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    run(args.input, args.output, profile=args.profile, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
