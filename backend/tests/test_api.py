"""Smoke tests for the HTTP API."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app.main import app


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
            "galaxy",
            "cluster",
        )

        before = client.get(f"/preview/{job_id}/before")
        assert before.status_code == 200
        assert before.headers["content-type"].startswith("image/png")

        result = client.get(f"/result/{job_id}")
        assert result.status_code == 200
        assert result.headers["content-type"] == "image/png"


def test_status_404() -> None:
    with TestClient(app) as client:
        r = client.get("/status/nosuch")
    assert r.status_code == 404
