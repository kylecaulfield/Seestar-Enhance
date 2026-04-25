"""FastAPI app for Seestar Enhance.

HTTP surface:

- GET  /health                     — liveness probe.
- POST /process                    — upload a FITS file, start a background
                                     pipeline job, return a job id.
- GET  /status/{job_id}            — current stage name + 0..1 progress.
- GET  /result/{job_id}            — download the enhanced 16-bit PNG.
- GET  /preview/{job_id}/before    — simple-stretched preview of the input
                                     (for a before/after slider).
- Static SPA served at / (if the frontend build exists).

Job state lives in an in-process dict and pipelines run in a bounded
thread pool. This is deliberately unpretentious — no Celery, no Redis;
v1 is a single process on a single box. See BACKLOG.md for the async
upgrade path.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

from app.pipeline import run
from app.stages.io_fits import load_fits

logger = logging.getLogger(__name__)

_WORK_ROOT = Path(tempfile.gettempdir()) / "seestar-enhance-jobs"
_WORK_ROOT.mkdir(parents=True, exist_ok=True)

# FITS files start with an ASCII "SIMPLE  =" card in the primary HDU.
# We peek at the first chunk of each upload and reject anything else
# before scheduling a worker — otherwise non-FITS bytes crash deep inside
# astropy with an implementation-flavoured error that leaks to the client.
_FITS_MAGIC = b"SIMPLE  ="
_MAGIC_LEN = len(_FITS_MAGIC)

# Upload size cap. Real Seestar S50 stacked FITS top out around ~250 MB;
# 500 MB gives headroom and still rejects obvious DoS uploads.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

# 2 parallel pipelines is plenty on a typical box; BM3D already uses
# multiple cores internally and running too many in parallel thrashes.
_MAX_WORKERS = 2
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="pipeline")

# Cap active+queued pipeline work so a single flood of uploads can't queue
# hours of backlog or fill disk with temp dirs faster than the reaper can
# clear them. Cap is deliberately small: 2 workers × 3 = 6 in flight is
# enough for a handful of legitimate concurrent users.
_MAX_INFLIGHT_JOBS = _MAX_WORKERS * 3
_inflight_count = 0

# Static SPA is built into backend/app/static by the Dockerfile. If it's
# missing we skip the static mount (dev runs use vite on :5173 instead).
_STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    input_path: Path = field(default_factory=Path)
    output_path: Path = field(default_factory=Path)
    before_path: Path = field(default_factory=Path)
    # Stage-preview thumbnails land in `stages_dir/<stage>.png` as the
    # pipeline runs. `stages_done` lists the names that have a thumb on
    # disk so the UI knows which to fetch (and in what order).
    stages_dir: Path = field(default_factory=Path)
    stages_done: list[str] = field(default_factory=list)
    classification: str | None = None
    # Unix timestamp when the job reached a terminal state (done/error).
    # None while queued/running. Used by the reaper to enforce the TTL.
    terminated_at: float | None = None


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()

# Jobs in a terminal state (done/error) are reaped this many seconds after
# they finish. An hour gives the user plenty of time to download the PNG;
# the reaper runs frequently enough that disk doesn't accumulate.
_JOB_TTL_SECONDS = 3600
_REAPER_INTERVAL_SECONDS = 60


# Canonical execution order of stage-preview thumbnails. The UI uses
# this to display the strip left-to-right in pipeline order rather than
# whatever filesystem ordering ls returns. Stages not listed here (none
# right now) would be appended at the end.
_STAGE_PREVIEW_ORDER = (
    "load",
    "dark_subtract",
    "cosmetic",
    "background",
    "spcc",
    "color",
    "deconv",
    "stretch",
    "stars_split",
    "starless_stretch",
    "bm3d_denoise",
    "sharpen",
    "clahe",
    "curves",
    "recombine",
)


def _refresh_stages_done(job: Job) -> None:
    """Sweep the job's stage-preview directory and rebuild stages_done.

    Filesystem-driven so the pipeline doesn't need to call back into
    the API layer; the previews are the source of truth.
    """
    if not job.stages_dir.is_dir():
        return
    seen = {p.stem for p in job.stages_dir.glob("*.png")}
    ordered = [s for s in _STAGE_PREVIEW_ORDER if s in seen]
    extras = sorted(seen - set(_STAGE_PREVIEW_ORDER))
    new = ordered + extras
    with _JOBS_LOCK:
        # Avoid a write when nothing has changed — keeps polling cheap.
        if new != job.stages_done:
            job.stages_done = new


def _save_before_preview(fits_path: Path, png_path: Path) -> None:
    """Render a quick log-stretched PNG of the raw input for the slider.

    Intentionally simple — a percentile-normalized log stretch. It should
    look obviously worse than the processed output so the before/after
    comparison is meaningful.
    """
    img = load_fits(fits_path)
    luma = img.mean(axis=-1)
    black = float(np.percentile(luma, 5.0))
    white = float(np.percentile(luma, 99.9))
    denom = max(white - black, 1e-6)
    normalized = np.clip((img - black) / denom, 0.0, 1.0)
    stretched = np.log1p(normalized * 200.0)
    peak = float(stretched.max()) if stretched.size else 1.0
    if peak > 0:
        stretched = stretched / peak
    as_u8 = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(as_u8, mode="RGB").save(png_path)


def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    try:
        with _JOBS_LOCK:
            job.status = "running"
            job.stage = "load"
            job.progress = 0.0

        _save_before_preview(job.input_path, job.before_path)

        def progress(stage: str, frac: float) -> None:
            with _JOBS_LOCK:
                job.stage = stage
                job.progress = float(frac)
            # The pipeline writes <stages_dir>/<stage>.png after each
            # stage. We refresh stages_done at frac=1.0 (stage end)
            # only — saves O(N) stat calls per progress tick. Critical
            # to call this OUTSIDE the lock above: _refresh_stages_done
            # acquires _JOBS_LOCK itself, and threading.Lock() is not
            # reentrant.
            if frac >= 0.999:
                _refresh_stages_done(job)

        # Classify up front so the UI can surface the picked profile in
        # /status while the long BM3D stage runs. pipeline.run() will pick
        # the same one since we pass it explicitly.
        # NB: classify() is CPU-bound (~1s on real Seestar FITS) — do not
        # hold _JOBS_LOCK across it or every /status poll blocks.
        from app.stages.classify import classify

        img = load_fits(job.input_path)
        classification = classify(img)
        del img
        with _JOBS_LOCK:
            job.classification = classification

        run(
            job.input_path,
            job.output_path,
            profile=job.classification,
            progress=progress,
            stage_preview_dir=job.stages_dir,
        )

        # Final sweep — `recombine` and `export` may produce thumbs
        # after the last progress callback fires.
        _refresh_stages_done(job)

        with _JOBS_LOCK:
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
            job.terminated_at = time.time()
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        # Surface a user-friendly message; the full traceback is in the
        # server log. Astropy raises a dense multi-line OSError on malformed
        # FITS that does not belong in a web UI.
        with _JOBS_LOCK:
            job.status = "error"
            job.error = _friendly_error(exc)
            job.terminated_at = time.time()
    finally:
        global _inflight_count
        with _JOBS_LOCK:
            _inflight_count -= 1


def _friendly_error(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "OSError" or name.endswith("VerifyError") or name.endswith("HeaderParsingError"):
        return "Could not parse the uploaded file as a Seestar FITS image."
    # Strip the per-job temp directory from any exception message —
    # otherwise astropy's "Could not parse FITS at /tmp/seestar-
    # enhance-jobs/<uuid>/input.fits" leaks our internal layout to
    # users. Pattern is fixed so a simple replace suffices.
    msg = str(exc)[:400]
    msg = msg.replace(str(_WORK_ROOT), "[job]")
    return f"{name}: {msg[:200]}"


def _sweep_orphan_dirs() -> None:
    """Delete any job directories left behind by a previous process.

    Called at import time. Anything under _WORK_ROOT is stale by definition
    since the in-memory _JOBS dict starts empty on every boot.
    """
    if not _WORK_ROOT.is_dir():
        return
    for entry in _WORK_ROOT.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def _reap_expired_jobs() -> None:
    """Remove terminal jobs older than _JOB_TTL_SECONDS and their temp dirs."""
    now = time.time()
    to_remove: list[str] = []
    with _JOBS_LOCK:
        for job_id, job in _JOBS.items():
            if job.terminated_at is None:
                continue
            if now - job.terminated_at >= _JOB_TTL_SECONDS:
                to_remove.append(job_id)
        for job_id in to_remove:
            job = _JOBS.pop(job_id)
    # rmtree runs outside the lock; disk IO should not block status polls.
    for job_id in to_remove:
        job_dir = _WORK_ROOT / job_id
        shutil.rmtree(job_dir, ignore_errors=True)
    if to_remove:
        logger.info("reaped %d expired jobs", len(to_remove))


def _reaper_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(_REAPER_INTERVAL_SECONDS):
        try:
            _reap_expired_jobs()
        except Exception:  # noqa: BLE001
            logger.exception("reaper iteration failed")


_REAPER_STOP = threading.Event()
_REAPER_THREAD: threading.Thread | None = None


def _start_reaper() -> None:
    global _REAPER_THREAD
    if _REAPER_THREAD is not None:
        return
    _sweep_orphan_dirs()
    t = threading.Thread(
        target=_reaper_loop,
        args=(_REAPER_STOP,),
        name="job-reaper",
        daemon=True,
    )
    t.start()
    _REAPER_THREAD = t


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ANN001 — FastAPI lifespan signature
    _start_reaper()
    try:
        yield
    finally:
        _REAPER_STOP.set()


app = FastAPI(title="Seestar Enhance API", version="0.1.0", lifespan=_lifespan)


# CORS allowlist. v1 has no auth, so the default is restrictive: only
# same-origin (FastAPI serves the SPA) plus the local Vite dev port.
# Operators that front the API with a known SPA host set the env var
# (comma-separated). The literal "*" remains supported for users who
# really want the old open behaviour, with a warning in HOSTING.md.
_DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:8000,http://localhost:5173,http://127.0.0.1:8000,http://127.0.0.1:5173"
)
_allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS).strip()
if _allowed_origins_raw == "*":
    _allowed_origins: list[str] = ["*"]
else:
    _allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Content-Security-Policy + a small set of standard hardening headers.
# v1 is fully self-hosted (no third-party JS, no inline scripts beyond
# what Vite emits in the bundle, no external images) so the policy can
# be tight. Defence-in-depth — we don't have any XSS today (React auto-
# escapes everything we render) but a future contributor adding
# dangerouslySetInnerHTML would inherit a CSP-protected baseline.
class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            # `unsafe-inline` is unavoidable because Vite's preload-
            # link injection emits a tiny inline script. If Vite ever
            # ships a strict-CSP-compatible build mode, drop it.
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        # X-Frame-Options is a legacy duplicate of frame-ancestors but
        # some hosts/CDNs still strip CSP; keep both.
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response


app.add_middleware(_SecurityHeadersMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_endpoint(file: UploadFile) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    name_lower = file.filename.lower()
    if not (
        name_lower.endswith(".fit") or name_lower.endswith(".fits") or name_lower.endswith(".fts")
    ):
        raise HTTPException(status_code=400, detail="expected a .fit/.fits/.fts file")

    # Queue cap: reject uploads when there's already enough in-flight work
    # to keep every pipeline worker busy for a while. Stops a single client
    # from queueing hours of backlog and filling /tmp faster than the reaper
    # can keep up. Counter is bumped here and decremented in _run_job.
    global _inflight_count
    with _JOBS_LOCK:
        if _inflight_count >= _MAX_INFLIGHT_JOBS:
            raise HTTPException(
                status_code=429,
                detail="server busy; please retry in a moment",
            )
        _inflight_count += 1

    job_id = uuid.uuid4().hex
    job_dir = _WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.fits"
    output_path = job_dir / "output.png"
    before_path = job_dir / "before.png"
    stages_dir = job_dir / "stages"

    # Stream to disk while enforcing a size cap and validating the first
    # bytes look like a FITS primary HDU. A leading buffer accumulates
    # until we have enough bytes to compare against the magic, then the
    # bytes flush to disk; all further chunks pass through.
    total = 0
    header_buf = bytearray()
    magic_ok = False
    try:
        with input_path.open("wb") as f:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap",
                    )
                if not magic_ok:
                    header_buf.extend(chunk)
                    if len(header_buf) >= _MAGIC_LEN:
                        if bytes(header_buf[:_MAGIC_LEN]) != _FITS_MAGIC:
                            raise HTTPException(
                                status_code=400,
                                detail="not a valid FITS file (missing SIMPLE header)",
                            )
                        magic_ok = True
                        f.write(bytes(header_buf))
                        header_buf.clear()
                else:
                    f.write(chunk)
            if not magic_ok:
                # Stream ended before we saw enough bytes.
                raise HTTPException(status_code=400, detail="file is too small to be a FITS image")
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        with _JOBS_LOCK:
            _inflight_count -= 1
        raise

    job = Job(
        id=job_id,
        input_path=input_path,
        output_path=output_path,
        before_path=before_path,
        stages_dir=stages_dir,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    _EXECUTOR.submit(_run_job, job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str) -> dict[str, object]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    # Refresh under the read path too: jobs run in a worker thread but
    # the poll loop on the UI wants thumbnails as soon as they land.
    # Cheap O(stages) glob; only triggered while the job is running.
    if job.status == "running":
        _refresh_stages_done(job)
    with _JOBS_LOCK:
        return {
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "classification": job.classification,
            "error": job.error,
            "stages_done": list(job.stages_done),
        }


@app.get("/result/{job_id}")
def result_endpoint(job_id: str) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job not ready: {job.status}")
    if not job.output_path.is_file():
        raise HTTPException(status_code=500, detail="output missing")
    return FileResponse(
        job.output_path,
        media_type="image/png",
        filename=f"seestar-enhance-{job_id[:8]}.png",
    )


@app.get("/preview/{job_id}/before")
def preview_before_endpoint(job_id: str) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if not job.before_path.is_file():
        raise HTTPException(status_code=409, detail="preview not ready")
    return FileResponse(job.before_path, media_type="image/png")


# Allow only the canonical stage names through to the filesystem so a
# crafted /preview/{job}/stage/../../etc/passwd request can't escape the
# job's stages_dir. Easier than doing path resolution + a containment
# check, and the set is tiny.
_VALID_STAGE_NAMES = frozenset(_STAGE_PREVIEW_ORDER)


@app.get("/preview/{job_id}/stage/{stage}")
def preview_stage_endpoint(job_id: str, stage: str) -> FileResponse:
    """Per-stage thumbnail PNG dumped during pipeline execution.

    The thumbnails (300x300 PNG, ~50 KB each) are written by
    `pipeline.run()` after each stage runs. The frontend's stage-strip
    component fetches each one as it lands. Returns 404 for an unknown
    job/stage and 409 if the stage hasn't produced a thumbnail yet.
    """
    if stage not in _VALID_STAGE_NAMES:
        raise HTTPException(status_code=404, detail="unknown stage")
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    thumb_path = job.stages_dir / f"{stage}.png"
    if not thumb_path.is_file():
        raise HTTPException(status_code=409, detail="stage preview not ready")
    return FileResponse(thumb_path, media_type="image/png")


if _STATIC_DIR.is_dir():
    # mounted last so API routes above win on collision.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="spa")
