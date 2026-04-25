"""Smoke tests for the HTTP API."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from app.main import app
from astropy.io import fits
from fastapi.testclient import TestClient


@pytest.fixture
def synthetic_fits_bytes(tmp_path: Path) -> bytes:
    rng = np.random.default_rng(1)
    h, w = 64, 96
    mosaic = (rng.random((h, w)) * 30000 + 2000).astype(np.uint16)
    hdu = fits.PrimaryHDU(mosaic)
    hdu.header["BAYERPAT"] = "RGGB"
    path = tmp_path / "t.fits"
    hdu.writeto(path)
    return path.read_bytes()


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_process_rejects_non_fits() -> None:
    with TestClient(app) as client:
        r = client.post("/process", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_process_status_result_flow(synthetic_fits_bytes: bytes) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/process",
            files={"file": ("t.fits", synthetic_fits_bytes, "application/fits")},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Poll until done (synthetic image is tiny so this is fast).
        deadline = time.time() + 60
        while time.time() < deadline:
            s = client.get(f"/status/{job_id}").json()
            if s["status"] in ("done", "error"):
                break
            time.sleep(0.2)

        status = client.get(f"/status/{job_id}").json()
        assert status["status"] == "done", status
        assert status["progress"] == 1.0
        assert status["classification"] in (
            "nebula",
            "nebula_wide",
            "nebula_filament",
            "nebula_dominant",
            "galaxy",
            "cluster",
        )

        before = client.get(f"/preview/{job_id}/before")
        assert before.status_code == 200
        assert before.headers["content-type"].startswith("image/png")

        # Stage previews: status should report a non-empty list of
        # stages that produced thumbnails, and each name should be
        # fetchable as a PNG.
        stages_done = status["stages_done"]
        assert isinstance(stages_done, list) and len(stages_done) >= 4, stages_done
        # Pipeline order — load comes before background, etc.
        assert stages_done[0] == "load"
        assert "curves" in stages_done

        for stage in stages_done:
            r = client.get(f"/preview/{job_id}/stage/{stage}")
            assert r.status_code == 200, f"stage {stage} fetch failed: {r.status_code}"
            assert r.headers["content-type"].startswith("image/png")

        result = client.get(f"/result/{job_id}")
        assert result.status_code == 200
        assert result.headers["content-type"] == "image/png"


def test_status_404() -> None:
    with TestClient(app) as client:
        r = client.get("/status/nosuch")
    assert r.status_code == 404


def test_stage_preview_unknown_stage() -> None:
    """Path-traversal + unknown-name guard.

    The endpoint validates the stage name before consulting the job
    registry, so we don't need to start a real pipeline (which would
    leave a worker thread running between tests and race with the
    next test's BM3D call). A nonexistent job_id is fine here because
    the stage-name check 404s first.
    """
    with TestClient(app) as client:
        r = client.get("/preview/anyjob/stage/etc-passwd")
    assert r.status_code == 404


def test_stage_preview_unknown_job() -> None:
    """Valid stage name but unknown job — second 404 path."""
    with TestClient(app) as client:
        r = client.get("/preview/nosuch/stage/load")
    assert r.status_code == 404


# ---------- security headers ----------


def test_response_carries_csp_and_hardening_headers() -> None:
    """Every response carries the CSP + standard hardening headers."""
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"


def test_error_message_strips_internal_temp_path() -> None:
    """`_friendly_error` should not leak the per-job temp path in
    user-visible error messages.
    """
    from app.main import _WORK_ROOT, _friendly_error

    leak = OSError(f"failed to read {_WORK_ROOT / 'abc123' / 'input.fits'}")
    msg = _friendly_error(leak)
    assert str(_WORK_ROOT) not in msg
    assert "[job]" in msg or "Could not parse" in msg
