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
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app import profiles
from app.stages import (
    background,
    bm3d_denoise,
    clahe,
    classify,
    color,
    cosmetic,
    curves,
    dark_subtract,
    deconv,
    export,
    sharpen,
    spcc,
    stars,
    stretch,
)
from app.stages.io_fits import load_fits_with_wcs

PathLike = str | Path
ProgressCallback = Callable[[str, float], None]

logger = logging.getLogger(__name__)


# Stages in execution order. Progress is reported as stage-starting
# (frac=0.0) and stage-finished (frac=1.0); intermediate fractions are
# reserved for long-running stages that learn to emit their own updates.
_STAGES = (
    "load",
    "classify",
    "background",
    "color",
    "stretch",
    "bm3d_denoise",
    "sharpen",
    "curves",
    "export",
)


def _noop_progress(stage: str, frac: float) -> None:  # noqa: ARG001
    return None


def _save_stage_preview(
    img: np.ndarray,
    path: Path,
    size: tuple[int, int] = (300, 300),
) -> None:
    """Write a thumbnail PNG of the in-flight pipeline image.

    Pre-stretch stages have luma in [0, 0.05] and would render as a
    black square. We auto-apply a quick log preview when the image
    looks linear so users can actually see what each stage is doing.
    Same approach as `main._save_before_preview`. ~50 KB per thumb.
    """
    from PIL import Image as _PILImage

    # Downscale via numpy slicing FIRST, then convert to PIL. Avoids
    # passing big numpy arrays (potentially non-contiguous views from
    # the pipeline) through Pillow's `thumbnail()`, which has caused
    # SIGABRT crashes in repeated within-process calls — likely a
    # libjpeg/libpng state issue when stacked on top of the API's
    # earlier PIL operations. Numpy slicing is safe + fast.
    h, w = img.shape[:2]
    sh, sw = size
    step = max(1, min(h // sh, w // sw))
    sub = np.ascontiguousarray(img[::step, ::step, :3])

    arr = np.clip(sub, 0.0, 1.0).astype(np.float32, copy=True)
    luma_max = float(arr.mean(axis=-1).max()) if arr.size else 1.0
    if luma_max < 0.2:
        # Linear pre-stretch image — apply a log-stretch preview.
        luma = arr.mean(axis=-1)
        black = float(np.percentile(luma, 5.0))
        white = float(np.percentile(luma, 99.9))
        denom = max(white - black, 1e-6)
        normalized = np.clip((arr - black) / denom, 0.0, 1.0)
        arr = np.log1p(normalized * 200.0)
        peak = float(arr.max()) if arr.size else 1.0
        if peak > 0:
            arr = arr / peak

    as_u8 = np.ascontiguousarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    pil = _PILImage.fromarray(as_u8, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: PIL writes incrementally to `path` if we save directly,
    # so a /preview/{id}/stage/{name} request that races the worker thread
    # can hit a partial PNG and 500. Save to a `.tmp` sibling and rename
    # onto the final name — POSIX rename swaps the inode atomically so
    # readers always see either the previous PNG or the new one. The
    # `_refresh_stages_done` glob is `*.png` so the `.png.tmp` siblings
    # are invisible to the API.
    tmp_path = path.with_name(path.name + ".tmp")
    pil.save(str(tmp_path), format="PNG", optimize=True)
    tmp_path.replace(path)


def run(
    input_fits: PathLike,
    output_png: PathLike,
    profile: str | None = None,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    dark: np.ndarray | None = None,
    stage_preview_dir: PathLike | None = None,
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
    stage_preview_dir : str | Path | None
        When set, writes a 300×300 thumbnail PNG named ``<stage>.png`` into
        this directory after each stage runs. Used by the web UI's
        stage-by-stage preview strip. None (default) skips all dumping.
    """
    cb: ProgressCallback = progress or _noop_progress

    # Stage-preview helper: bound to the configured dir + does nothing
    # when no dir is set, so the inner pipeline body stays uncluttered.
    if stage_preview_dir is not None:
        preview_root = Path(stage_preview_dir)
        preview_root.mkdir(parents=True, exist_ok=True)

        def dump(stage: str, image: np.ndarray) -> None:
            try:
                _save_stage_preview(image, preview_root / f"{stage}.png")
            except Exception:  # noqa: BLE001 — preview is best-effort
                logger.exception("stage-preview dump failed for %s", stage)
    else:

        def dump(stage: str, image: np.ndarray) -> None:  # noqa: ARG001
            return None

    def log(msg: str) -> None:
        logger.info(msg)
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    log(f"[1/{len(_STAGES)}] loading {input_fits}")
    cb("load", 0.0)
    img, wcs = load_fits_with_wcs(input_fits)
    if wcs is not None:
        log("  FITS includes valid WCS (SPCC-capable)")
    cb("load", 1.0)
    dump("load", img)

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

    # Dark-frame subtraction: first opportunity, before any processing.
    # Caller supplies the master dark via `dark=`; profile can set
    # `dark_scale` to tune it.
    if dark is not None:
        log("[2a] dark-frame subtraction")
        dark_params = params.get("dark_subtract", {})
        img = dark_subtract.process(img, dark=dark, **dark_params)
        dump("dark_subtract", img)

    # Cosmetic: hot-pixel / cosmic-ray rejection. Pre-background so the
    # fit samples aren't biased by single-pixel outliers. Opt-in.
    cosmetic_params = params.get("cosmetic")
    if cosmetic_params is not None:
        log("[3a] cosmetic correction")
        img = cosmetic.process(img, **cosmetic_params)
        dump("cosmetic", img)

    log(f"[3/{len(_STAGES)}] background removal")
    cb("background", 0.0)
    img = background.process(img, **params.get("background", {}))
    cb("background", 1.0)
    dump("background", img)

    # SPCC (v2) runs BETWEEN background and color: the background fit
    # has already removed gradients (so the frame is flat enough for
    # aperture photometry on stars), and we're still pre-WB so the
    # fitted CCM describes the sensor's response directly — not the
    # sensor composed with whatever heuristic gains color.process
    # applies. When SPCC is enabled, profiles should lower
    # `color.wb_strength` (or set it to 0) so the heuristic WB
    # doesn't fight the catalogue fit.
    spcc_params = params.get("spcc")
    if spcc_params is not None and wcs is not None:
        log("[3a] SPCC photometric calibration")
        img = spcc.process(img, wcs=wcs, **spcc_params)
        dump("spcc", img)
    elif spcc_params is not None:
        log("[3a] SPCC: skipped (no WCS in FITS header)")

    log(f"[4/{len(_STAGES)}] color neutralization + white balance")
    cb("color", 0.0)
    img = color.process(img, **params.get("color", {}))
    cb("color", 1.0)
    dump("color", img)

    # Deconvolution: tighten star PSFs in linear space before the
    # arcsinh stretch turns wide Gaussians into soft bright halos.
    # Luma-only — chroma wouldn't benefit and could take artefacts.
    deconv_params = params.get("deconv")
    if deconv_params is not None:
        log("[4a] star PSF deconvolution")
        img = deconv.process(img, **deconv_params)
        dump("deconv", img)

    log(f"[5/{len(_STAGES)}] stretch")
    cb("stretch", 0.0)
    img = stretch.process(img, **params.get("stretch", {}))
    cb("stretch", 1.0)
    dump("stretch", img)

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
        dump("stars_split", img)
        if starless_stretch_params is not None:
            log("[5b] additional stretch on starless")
            img = stretch.process(img, **starless_stretch_params)
            dump("starless_stretch", img)

    log(f"[6/{len(_STAGES)}] bm3d denoise (classical)")
    cb("bm3d_denoise", 0.0)
    img = bm3d_denoise.process(img, **params.get("bm3d_denoise", {}))
    cb("bm3d_denoise", 1.0)
    dump("bm3d_denoise", img)

    log(f"[7/{len(_STAGES)}] sharpen")
    cb("sharpen", 0.0)
    img = sharpen.process(img, **params.get("sharpen", {}))
    cb("sharpen", 1.0)
    dump("sharpen", img)

    # CLAHE: local contrast on luma. When star_split is active this
    # lands on the starless layer so stars don't get their cores
    # crushed by local-tile equalisation. Opt-in per profile.
    clahe_params = params.get("clahe")
    if clahe_params is not None:
        log("[7a] CLAHE local contrast")
        img = clahe.process(img, **clahe_params)
        dump("clahe", img)

    log(f"[8/{len(_STAGES)}] curves")
    cb("curves", 0.0)
    img = curves.process(img, **params.get("curves", {}))
    cb("curves", 1.0)
    dump("curves", img)

    # Screen-blend the unmodified stars back onto the heavily-processed
    # starless. Must happen after curves so stars don't get double-boosted.
    if stars_only is not None:
        log("[8a] recombine starless + stars (screen blend)")
        img = stars.recombine(stars_only, img)
        dump("recombine", img)

    log(f"[9/{len(_STAGES)}] export -> {output_png}")
    cb("export", 0.0)
    export.process(img, output_png)
    cb("export", 1.0)

    cb("done", 1.0)
    log(f"wrote {output_png}")
    return img


def _coerce_override_value(raw: str) -> object:
    """Parse an `--override stage.param=value` value into a Python literal.

    Tries int, then float, then bool ("true"/"false"), else returns the
    raw string. Lists/tuples are parsed through a minimal comma split —
    enough for `channel_gains=1.4,1.1,0.7` which is the dominant shape
    of overrides we need.
    """
    if "," in raw:
        return tuple(_coerce_override_value(part.strip()) for part in raw.split(","))
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_overrides(profile_dict: dict, overrides: list[str]) -> dict:
    """Apply `stage.param=value` overrides to a profile dict.

    Returns a *new* dict so the shared profile registry isn't mutated.
    """
    import copy

    result = copy.deepcopy(profile_dict)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"override {entry!r} must be of the form stage.param=value")
        key, value_raw = entry.split("=", 1)
        if "." not in key:
            raise ValueError(f"override key {key!r} must be of the form stage.param")
        stage, param = key.split(".", 1)
        value = _coerce_override_value(value_raw)
        result.setdefault(stage, {})
        if not isinstance(result[stage], dict):
            raise ValueError(f"override target {stage!r} is not a stage params dict")
        result[stage][param] = value
    return result


def _run_one(
    input_path: PathLike,
    output_path: PathLike,
    profile: str | None,
    overrides: list[str],
    verbose: bool,
) -> None:
    """Single-file run that threads --override through the profile layer.

    Resolves the profile (auto-classify if not supplied) then deep-copies
    it, applies overrides, and monkey-patches it into the profile registry
    for the duration of the call so `pipeline.run()` picks it up.
    """
    if not overrides:
        run(input_path, output_path, profile=profile, verbose=verbose)
        return

    if profile is None:
        img, _ = load_fits_with_wcs(input_path)
        profile = classify.classify(img)
    base = profiles.get(profile)
    patched = _apply_overrides(base, overrides)

    # Patch the registry entry for this one call, then restore.
    original = profiles.PROFILES[profile]
    profiles.PROFILES[profile] = patched
    try:
        run(input_path, output_path, profile=profile, verbose=verbose)
    finally:
        profiles.PROFILES[profile] = original


def _resolve_batch_jobs(requested: int, n_work: int) -> int:
    """Clamp `--jobs` to [1, min(os.cpu_count(), n_work)].

    `--jobs 0` (or any non-positive value) asks for one-per-core. We
    cap at the number of inputs so we don't spin up empty workers.
    """
    import os as _os

    cpu = _os.cpu_count() or 1
    if requested <= 0:
        return max(1, min(cpu, n_work))
    return max(1, min(requested, n_work))


# Module-level entry for ProcessPoolExecutor — cloudpickle doesn't like
# nested closures, so we need a proper function.
def _batch_worker(args: tuple) -> tuple[str, str | None]:
    src, dst, profile, overrides, verbose = args
    try:
        _run_one(src, dst, profile=profile, overrides=overrides, verbose=verbose)
        return src, None
    except Exception as exc:  # noqa: BLE001
        return src, f"{type(exc).__name__}: {exc}"


def _blas_thread_env(n_jobs: int) -> dict[str, str]:
    """Env-var dict that throttles BLAS threads per worker process.

    numpy/scipy wheels autothread across all cores by default. Without
    throttling, `N_jobs` worker processes × `N_cpu` BLAS threads each
    = N*N oversubscription, which burns context-switch time instead of
    making forward progress. Dividing cores across workers removes the
    overlap. Per-process BLAS thread budget is `max(1, cpu // n_jobs)`.

    The vars are: OMP (OpenMP), MKL (Intel MKL — used by conda numpy
    and by some pip numpy wheels), OPENBLAS (most pip numpy wheels),
    NUMEXPR (used by some downstream libs), VECLIB (Apple Accelerate).
    """
    import os as _os

    cpu = _os.cpu_count() or 1
    per_worker = max(1, cpu // max(n_jobs, 1))
    val = str(per_worker)
    return {
        "OMP_NUM_THREADS": val,
        "MKL_NUM_THREADS": val,
        "OPENBLAS_NUM_THREADS": val,
        "NUMEXPR_NUM_THREADS": val,
        "VECLIB_MAXIMUM_THREADS": val,
    }


def _run_batch_parallel(
    work: list[tuple[str, str]],
    profile: str | None,
    overrides: list[str],
    verbose: bool,
    n_jobs: int,
) -> int:
    """Run `work` pipelines across `n_jobs` worker processes.

    Returns 0 if all succeeded, 1 if any failed. Failures are logged as
    they return; the batch continues on individual errors so a single
    bad FITS doesn't abort a night's worth of processing.

    BLAS thread count per worker is auto-throttled so N processes ×
    all-cores-each doesn't oversubscribe. See `_blas_thread_env`.
    """
    import os as _os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    env_throttle = _blas_thread_env(n_jobs)
    logger.info(
        "batch: %d images across %d workers (BLAS threads/worker=%s)",
        len(work),
        n_jobs,
        env_throttle["OMP_NUM_THREADS"],
    )

    # Apply the throttle to the parent so workers inherit. Restore on
    # exit so a caller running parallel then sequential doesn't drag
    # single-process BLAS threads down.
    saved = {k: _os.environ.get(k) for k in env_throttle}
    _os.environ.update(env_throttle)
    try:
        payloads = [(src, dst, profile, overrides, verbose) for src, dst in work]
        failures = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_batch_worker, p): p[0] for p in payloads}
            for fut in as_completed(futures):
                src, err = fut.result()
                if err is None:
                    logger.info("ok: %s", src)
                else:
                    logger.error("failed on %s: %s", src, err)
                    failures += 1
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline",
        description="Run the Seestar enhancement pipeline end-to-end (v1).",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "input FITS file, or (with --batch) multiple FITS files. "
            "Without --batch, exactly one input + one output is required."
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "output PNG file (single-mode). Ignored in --batch mode; "
            "outputs land next to the inputs with .png suffix, or under "
            "--output-dir when specified."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "process every positional arg as an input FITS. Each input "
            "generates a matching .png next to it (or under --output-dir)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to place outputs in --batch mode (default: beside input).",
    )
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
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="stage.param=value",
        help=(
            "override a single profile parameter without editing "
            "profiles.py. Repeatable. Examples: "
            "--override stretch.stretch=22 "
            "--override curves.contrast=0.75 "
            "--override 'curves.channel_gains=1.4,1.1,0.6'"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "batch-mode parallelism: run N pipelines concurrently in "
            "separate processes. Defaults to 1 (sequential). "
            "`--jobs 0` picks one per CPU core. Each job costs ~300 MB "
            "of RSS plus whatever BM3D allocates, so keep N*350 MB "
            "under free memory. Ignored in single-file mode."
        ),
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

    if args.batch:
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        # In batch mode, `output` is ignored, and every positional slot
        # (including the "output" slot) is treated as an input.
        inputs = list(args.input)
        if args.output is not None:
            inputs.append(args.output)
        work = [
            (str(Path(p)), str((out_dir or Path(p).parent) / f"{Path(p).stem}.png")) for p in inputs
        ]
        n_jobs = _resolve_batch_jobs(args.jobs, n_work=len(work))
        if n_jobs == 1:
            failures = 0
            for src, dst in work:
                try:
                    _run_one(
                        src,
                        dst,
                        profile=selected,
                        overrides=args.override,
                        verbose=args.verbose,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("failed on %s: %s", src, exc)
                    failures += 1
            return 0 if failures == 0 else 1
        return _run_batch_parallel(
            work,
            selected,
            args.override,
            args.verbose,
            n_jobs,
        )

    # Single-file mode
    if args.output is None or len(args.input) != 1:
        parser.error(
            "single-file mode needs exactly one input and one output; "
            "use --batch for multiple inputs."
        )
    _run_one(
        args.input[0],
        args.output,
        profile=selected,
        overrides=args.override,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
