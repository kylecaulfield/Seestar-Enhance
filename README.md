# Seestar-Enhance

Web app for enhancing ZWO Seestar astrophotography images.

Licensed under the [MIT License](LICENSE).

## Status: v1 (classical only)

**v1 is classical-only by design.** Shipping a star-remover or an AI denoiser
requires bundling model weights, and the widely-used options are not
compatible with this repo's MIT license:

- StarNet++ v2 weights are **CC BY-NC-SA 4.0** (non-commercial, ShareAlike)
  and no official ONNX build exists.
- GraXpert's code is **GPL-3.0**; its model weights have no explicit license
  and their redistribution terms are ambiguous.

Rather than silently depend on incompatible weights, v1 uses classical
algorithms throughout. See the v2 roadmap below.

## Layout

```
backend/    FastAPI service (Python 3.11) + image pipeline
frontend/   React + TypeScript + Vite (UI placeholder for now)
samples/    Drop Seestar .fit / .fits files here; outputs land in samples/outputs/
```

## Pipeline (v1)

`io_fits → background → color → stretch → bm3d_denoise → sharpen → curves → export`

Stages (all under `backend/app/stages/`):

| Stage | Module | What it does |
| --- | --- | --- |
| io_fits | `io_fits.py` | Reads a Seestar FITS, detects Bayer pattern (default `RGGB`), debayers (Malvar-2004), returns float32 RGB in `[0,1]`. |
| background | `background.py` | Per-channel RBF thin-plate-spline fit on sigma-clipped grid samples, subtracted with a median pedestal. |
| color | `color.py` | Dark-pixel pedestal subtract + mid-tone channel equalization. |
| stretch | `stretch.py` | Arcsinh with auto black-point at the 0.1st luminance percentile. |
| bm3d_denoise | `bm3d_denoise.py` | Classical BM3D via the `bm3d` pip package with MAD-based sigma estimation. Applied to the full image. |
| sharpen | `sharpen.py` | Gaussian unsharp mask. |
| curves | `curves.py` | Tanh S-curve contrast + luma-preserving saturation. |
| export | `export.py` | 16-bit RGB PNG via `pypng`. |

Per-stage parameters live in `backend/app/profiles.py` (`default`, `nebula`,
`galaxy`, `cluster`). Pass `--profile NAME` to the CLI.

### Stubs kept for v2

`backend/app/stages/stars.py` and `backend/app/stages/ml_denoise.py` are
kept as stubs with stable signatures. Both currently raise
`NotImplementedError`. When permissively-licensed or self-trained ONNX
weights are available, filling these in and wiring them into
`pipeline.run()` is mechanical. See `TODO(v2)` markers in both files and
the placeholder comment in `pipeline.py`.

## Run

```sh
docker-compose up --build
```

- Backend: http://localhost:8000/health
- Frontend: http://localhost:5173 (placeholder)

## CLI

```sh
cd backend
python -m app.pipeline ../samples/nebula.fits ../samples/outputs/nebula.png --profile nebula -v
```

`--profile` accepts `default`, `nebula`, `galaxy`, or `cluster`.

## Backend tests

```sh
cd backend
pip install -r requirements.txt
pytest
```

Tests for `io_fits` are skipped when `samples/` is empty. Drop Seestar FITS
files into `samples/` to enable them.

## v2 roadmap

Adding ML stages without changing the v1 public surface:

1. **Self-trained star removal** — small U-Net trained on public star-field
   data, exported to ONNX, bundled under MIT (or a compatible permissive
   license). Fills `stages/stars.py` (returns `(stars_only, starless)`) and
   slots in after `stretch` with a screen-blend recombine before `curves`.
2. **ML denoise** — either a self-trained denoiser or a permissively
   licensed third-party model. Fills `stages/ml_denoise.py` and runs on the
   starless layer only; if absent, `bm3d_denoise` remains the default.
3. **Tiled inference helper** — shared utility for 256/512-px tiles with
   cosine-window seam blending, used by both ML stages once they land.
