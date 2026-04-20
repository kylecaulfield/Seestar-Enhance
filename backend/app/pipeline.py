"""End-to-end pipeline: io_fits -> background -> color -> stretch -> sharpen
-> curves -> export.

Also runnable as a module:

    python -m app.pipeline <input.fits> <output.png>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np

from app.stages import (
    background,
    color,
    curves,
    denoise,
    export,
    sharpen,
    stars,
    stretch,
)
from app.stages.io_fits import load_fits

PathLike = Union[str, Path]


def run(input_fits: PathLike, output_png: PathLike, verbose: bool = False) -> np.ndarray:
    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    log(f"[1/7] loading {input_fits}")
    img = load_fits(input_fits)

    log("[2/7] background removal")
    img = background.process(img)

    log("[3/7] color neutralization + white balance")
    img = color.process(img)

    log("[4/7] stars (placeholder, pass-through)")
    img = stars.process(img)

    log("[5/7] denoise (placeholder, pass-through)")
    img = denoise.process(img)

    log("[6/7] stretch")
    img = stretch.process(img)

    log("[6.5/7] sharpen")
    img = sharpen.process(img)

    log("[7/7] curves + export")
    img = curves.process(img)
    export.process(img, output_png)
    log(f"wrote {output_png}")
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline",
        description="Run the Seestar enhancement pipeline end-to-end.",
    )
    parser.add_argument("input", help="input Seestar FITS file")
    parser.add_argument("output", help="output PNG file (16-bit)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    run(args.input, args.output, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
