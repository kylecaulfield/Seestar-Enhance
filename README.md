# Seestar-Enhance

Web app for enhancing ZWO Seestar astrophotography images.

## Status

Phase 1 scaffolding: backend health check + FITS loader. No image pipeline yet.

## Layout

```
backend/    FastAPI service (Python 3.11)
frontend/   React + TypeScript + Vite
samples/    Drop Seestar .fit / .fits files here for tests
```

## Run

```sh
docker-compose up --build
```

- Backend: http://localhost:8000/health
- Frontend: http://localhost:5173

## Backend tests

```sh
cd backend
pip install -r requirements.txt
pytest
```

Tests for `io_fits` are skipped automatically when `samples/` is empty. Drop
one or more Seestar FITS files into `samples/` to enable them.

## Implemented

- `GET /health` — liveness probe
- `backend/app/stages/io_fits.py`
  - `load_fits(path) -> np.ndarray` — reads a Seestar FITS file, detects the
    Bayer pattern from the header (`BAYERPAT`, default `RGGB`), debayers with
    Malvar-2004, returns float32 RGB normalized to `[0, 1]`.
  - `save_preview_png(arr, path)` — log-stretched 8-bit PNG for debugging.
