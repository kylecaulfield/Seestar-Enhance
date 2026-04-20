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
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.pipeline import run
from app.stages.io_fits import load_fits

logger = logging.getLogger(__name__)

_WORK_ROOT = Path(tempfile.gettempdir()) / "seestar-enhance-jobs"
_WORK_ROOT.mkdir(parents=True, exist_ok=True)

# 2 parallel pipelines is plenty on a typical box; BM3D already uses
# multiple cores internally and running too many in parallel thrashes.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")

# Static SPA is built into backend/app/static by the Dockerfile. If it's
# missing we skip the static mount (dev runs use vite on :5173 instead).
_STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    input_path: Path = field(default_factory=Path)
    output_path: Path = field(default_factory=Path)
    before_path: Path = field(default_factory=Path)
    classification: Optional[str] = None


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


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

        # Classify up front so the UI can surface the picked profile in
        # /status while the long BM3D stage runs. pipeline.run() will pick
        # the same one since we pass it explicitly.
        from app.stages.classify import classify

        img = load_fits(job.input_path)
        with _JOBS_LOCK:
            job.classification = classify(img)
        del img

        run(
            job.input_path,
            job.output_path,
            profile=job.classification,
            progress=progress,
        )

        with _JOBS_LOCK:
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        with _JOBS_LOCK:
            job.status = "error"
            job.error = str(exc)


app = FastAPI(title="Seestar Enhance API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_endpoint(file: UploadFile) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    name_lower = file.filename.lower()
    if not (
        name_lower.endswith(".fit")
        or name_lower.endswith(".fits")
        or name_lower.endswith(".fts")
    ):
        raise HTTPException(status_code=400, detail="expected a .fit/.fits/.fts file")

    job_id = uuid.uuid4().hex
    job_dir = _WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.fits"
    output_path = job_dir / "output.png"
    before_path = job_dir / "before.png"

    with input_path.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job = Job(
        id=job_id,
        input_path=input_path,
        output_path=output_path,
        before_path=before_path,
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
    return {
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "classification": job.classification,
        "error": job.error,
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


if _STATIC_DIR.is_dir():
    # mounted last so API routes above win on collision.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="spa")
