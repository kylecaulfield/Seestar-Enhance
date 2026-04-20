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

- `GET /health` — liveness probe.
- `backend/app/stages/io_fits.py` — `load_fits`, `save_preview_png`.
- Non-ML pipeline stages in `backend/app/stages/`:
  - `background.py` — RBF thin-plate-spline background fit with
    sigma-clipping, applied per channel.
  - `color.py` — dark-pixel neutralization + mid-tone white balance.
  - `stretch.py` — arcsinh stretch with auto black-point (0.1st percentile).
  - `sharpen.py` — gentle unsharp mask.
  - `curves.py` — mild S-curve contrast + saturation boost.
  - `export.py` — 16-bit RGB PNG (via pypng).
  - `stars.py`, `denoise.py` — pass-through placeholders for ML stages.
- `backend/app/pipeline.py` — chains `io_fits → background → color → stars →
  denoise → stretch → sharpen → curves → export` and exposes a CLI.

## CLI

```sh
cd backend
python -m app.pipeline ../samples/nebula.fits ../samples/outputs/nebula.png -v
```
