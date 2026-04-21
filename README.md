# Seestar-Enhance

A web app and reproducible image pipeline for enhancing astrophotography
from the ZWO Seestar smart telescope. Takes raw Bayer-pattern FITS files,
runs a stackable pipeline of classical image-processing stages, and writes
a 16-bit RGB PNG.

Licensed under the [MIT License](LICENSE).

---

## Table of contents

- [Status & design philosophy](#status--design-philosophy)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
  - [Docker Compose](#docker-compose)
  - [Prebuilt image (ghcr.io)](#prebuilt-image-ghcrio)
  - [Unraid](#unraid)
  - [Native (development)](#native-development)
- [Using the web app](#using-the-web-app)
- [Hosting](#hosting)
- [Pipeline](#pipeline)
  - [Stage reference](#stage-reference)
  - [Profiles & auto-classification](#profiles--auto-classification)
- [CLI](#cli)
- [Backend API](#backend-api)
- [FITS format notes](#fits-format-notes)
- [Testing](#testing)
- [Project structure in detail](#project-structure-in-detail)
- [v2 roadmap](#v2-roadmap)
- [Bundled model licenses](#bundled-model-licenses)
- [Contributing](#contributing)
- [References](#references)

---

## Status & design philosophy

**v1 is classical-only by design.** Star removal and ML-based denoising are
popular in astrophotography, but shipping a model means bundling weights,
and the widely-used options are not MIT-compatible:

| Model | Code license | Weights license | ONNX available? |
| --- | --- | --- | --- |
| StarNet++ v2 | MIT | **CC BY-NC-SA 4.0** (non-commercial, ShareAlike) | No official ONNX |
| GraXpert AI denoise | GPL-3.0 | Unspecified; code is GPL | Yes (runtime download from S3) |

Rather than adopt incompatible weights silently or relicense the project,
v1 uses classical algorithms throughout (BM3D for denoise, no star
removal). Hooks for v2 ML stages are already wired up as
`NotImplementedError` stubs with stable signatures — see the
[v2 roadmap](#v2-roadmap).

## Repository layout

```
.
├── LICENSE                 MIT.
├── README.md               This file.
├── BACKLOG.md              Planned work, tracked by area.
├── Dockerfile              Multi-stage: builds SPA, serves from FastAPI.
├── docker-compose.yml      Single service on :8000 (API + UI).
├── .github/workflows/      GHA: docker build+push to ghcr.io on main.
├── backend/                FastAPI service + image pipeline (Python 3.11).
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py         FastAPI app, job API, static SPA mount.
│   │   ├── pipeline.py     Orchestrator + CLI entrypoint.
│   │   ├── profiles.py     Per-stage parameter bundles.
│   │   ├── models/         (empty; v2 ONNX weights will land here)
│   │   ├── static/         Built SPA lands here in Docker builds.
│   │   └── stages/         One module per pipeline stage.
│   └── tests/              pytest suite.
├── frontend/               React 18 + TypeScript + Vite SPA.
└── samples/                Drop Seestar FITS files here; outputs in samples/outputs/.
```

## Quickstart

### Docker Compose

```sh
docker compose up --build
```

Open http://localhost:8000. A single container serves both the web UI
and the API. Drop a Seestar `.fit` / `.fits` / `.fts` file on the page,
watch the progress view, scrub the before/after slider, hit **Download**.

### Prebuilt image (ghcr.io)

Each push to `main` and every `v*.*.*` tag builds a multi-arch image
(`linux/amd64` + `linux/arm64`) on GitHub Actions and publishes it to
GitHub Container Registry:

```
ghcr.io/kylecaulfield/seestar-enhance:latest
ghcr.io/kylecaulfield/seestar-enhance:sha-<short>     # per-commit, immutable
ghcr.io/kylecaulfield/seestar-enhance:v1.2.3          # on release tags
```

Run it anywhere `docker` works — no git clone required:

```sh
docker run -d \
  --name seestar-enhance \
  -p 8000:8000 \
  --restart unless-stopped \
  ghcr.io/kylecaulfield/seestar-enhance:latest
```

The image bakes in a `HEALTHCHECK` against `/health`, so
`docker ps` will show container health as `healthy` once it's ready.

Update with:

```sh
docker pull ghcr.io/kylecaulfield/seestar-enhance:latest && \
docker rm -f seestar-enhance && \
docker run -d --name seestar-enhance -p 8000:8000 \
  --restart unless-stopped \
  ghcr.io/kylecaulfield/seestar-enhance:latest
```

### Unraid

The prebuilt image above is the path of least resistance on Unraid.

**Unraid → Docker → Add Container**, then fill in:

| Field | Value |
| --- | --- |
| Name | `seestar-enhance` |
| Repository | `ghcr.io/kylecaulfield/seestar-enhance:latest` |
| Network Type | `Bridge` (default) |
| Console shell command | `sh` |
| Privileged | off |
| WebUI | `http://[IP]:[PORT:8000]/` |

Add one **Port** mapping:

| Config Type | Name | Container Port | Host Port | Connection Type |
| --- | --- | --- | --- | --- |
| Port | Web | `8000` | `8000` | TCP |

No volume mounts are strictly required — per-job temp files live in
the container's `/tmp` and are cleaned up an hour after each job
finishes (see [BACKLOG.md](BACKLOG.md) / the `_JOB_TTL_SECONDS` constant
in `backend/app/main.py`). If you want jobs to survive container
restarts, add a path mapping:

| Config Type | Name | Container Path | Host Path | Access Mode |
| --- | --- | --- | --- | --- |
| Path | Temp jobs | `/tmp/seestar-enhance-jobs` | `/mnt/user/appdata/seestar-enhance/jobs` | Read/Write |

The image runs as a non-root user (UID 10001). If you use a custom host
path, make sure Unraid's `nobody`/`users` permissions (`99:100`) on
`/mnt/user/appdata/seestar-enhance/` are writable by that UID — easiest
fix is:

```sh
chown -R 10001:10001 /mnt/user/appdata/seestar-enhance
```

**Updating**: Unraid's Docker tab → click the container → **Force Update**.
The `:latest` tag on GHCR tracks the default branch automatically.

For a more detailed Unraid walkthrough (Compose Manager, NGINX Proxy
Manager, User Scripts for auto-update), see [HOSTING.md](HOSTING.md).

### Native (development)

You'll want two terminals — one for the backend, one for the Vite dev
server:

```sh
# terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev       # http://localhost:5173, proxies API calls to :8000
```

Production-style single-container run without compose:

```sh
docker build -t seestar-enhance .
docker run --rm -p 8000:8000 seestar-enhance
```

## Using the web app

1. **Drop a FITS file** on the zone in the middle of the page (or click
   to open a file picker). Only `.fit`, `.fits`, and `.fts` are accepted.
2. A full-screen progress view shows the current stage name and an
   overall bar. The UI polls `/status/{job_id}` every 500ms.
3. When the pipeline finishes, a **before/after slider** appears over
   the full viewport. Drag the handle to compare a simple log-stretched
   preview of the raw input against the enhanced result.
4. **Download PNG** saves the 16-bit RGB output. **Process another**
   returns to the drop zone.

There are no settings, no sliders, no profile picker. The backend
auto-classifies the image (nebula / galaxy / cluster) and selects
parameters for you. If you want knobs, use the [CLI](#cli) or edit
`profiles.py`.

## Hosting

For self-hosting on a VPS, home server, or **Unraid**, see
[HOSTING.md](HOSTING.md). It covers:

- Generic Docker hosting on any Linux box (prod-mode compose override,
  Caddy/Nginx reverse proxy, firewall, backups, updates).
- A step-by-step Unraid tutorial using Compose Manager, including
  appdata layout, user-share mounting, NGINX Proxy Manager, and a
  User Script for automatic git-pull updates.

## Pipeline

```
io_fits → background → color → stretch → bm3d_denoise → sharpen → curves → export
```

Every stage in `backend/app/stages/` exposes a single entry point:

```python
def process(image: np.ndarray, **params) -> np.ndarray: ...
```

`image` is always float32 RGB of shape `(H, W, 3)` with values in `[0, 1]`.
Stages are pure and composable; adding, reordering, or swapping a stage is
a matter of editing `pipeline.py`.

### Stage reference

| Order | Stage | Module | Summary |
| --- | --- | --- | --- |
| 1 | `io_fits` | `stages/io_fits.py` | Reads FITS, detects Bayer pattern, debayers, returns float32 RGB in `[0,1]`. |
| 2 | `background` | `stages/background.py` | Removes sky gradients via a thin-plate-spline RBF fit on sigma-clipped grid samples, per channel. |
| 3 | `color` | `stages/color.py` | Neutralizes dark-region tint, then equalizes mid-tone channel medians as a simple white balance. |
| 4 | `stretch` | `stages/stretch.py` | Arcsinh stretch with an auto black-point at the 0.1st luminance percentile. |
| 5 | `bm3d_denoise` | `stages/bm3d_denoise.py` | Block-Matching 3D denoise via the `bm3d` pip package. Sigma auto-estimated from the luminance Laplacian. |
| 6 | `sharpen` | `stages/sharpen.py` | Gaussian unsharp mask with gentle defaults. |
| 7 | `curves` | `stages/curves.py` | Tanh S-curve for contrast, luma-preserving saturation boost. |
| 8 | `export` | `stages/export.py` | Writes a 16-bit RGB PNG via `pypng`. |

Each `process(image, **params)` accepts keyword overrides — the defaults
in each module are a reasonable starting point; the more opinionated
bundles live in `profiles.py`.

#### Stage details

- **io_fits.** Reads via `astropy.io.fits`. Bayer pattern comes from the
  `BAYERPAT` header (falls back to `BAYRPAT`, `COLORTYP`, `CFAPAT`, then
  defaults to `RGGB` which matches the Seestar S50 sensor). Normalization
  uses `BITPIX` when present. Debayers with Malvar-2004 from
  `colour_demosaicing`.
- **background.** Samples on a `grid × grid` lattice, sigma-clips
  iteratively (MAD-based), fits `scipy.interpolate.RBFInterpolator` with
  `thin_plate_spline` kernel, queries on a downscaled grid, then cubic
  zooms back to full resolution. The zoom-back is what makes the stage
  affordable on large images.
- **color.** Two-phase: subtract per-channel median of the bottom
  `dark_percentile` of luminance, then scale channels so their medians
  inside the mid-tone luminance band match. `wb_strength < 1.0` damps the
  white-balance pull.
- **stretch.** `arcsinh(x * k) / arcsinh(k)` with `k = stretch`. Higher
  `stretch` pulls more faint detail up; typical values `10`–`40`.
- **bm3d_denoise.** Uses `bm3d.bm3d_rgb`. Sigma is passed directly if
  supplied; otherwise estimated from the MAD of `scipy.ndimage.laplace`
  on the luminance channel, divided by a calibrated constant. A
  `strength` multiplier scales that estimate.
- **sharpen.** `image + amount * (image - gaussian(image, sigma=radius))`.
- **curves.** S-curve is `tanh(a * (2x - 1)) / tanh(a)` rescaled to
  `[0,1]`, with `a = 1 + contrast`. Saturation is a luma-preserving
  scalar around the per-pixel luma mean.
- **export.** Clips to `[0,1]`, scales to uint16, writes via `pypng`.
  Pillow's high-level API does not support 16-bit RGB output, which is
  why we use `pypng` here.

### Profiles & auto-classification

`backend/app/profiles.py` exposes four named profiles:

- `default` — conservative settings that look acceptable on most targets.
- `nebula` — stronger stretch, boosted saturation, heavier denoise +
  chroma-blur for faint diffuse targets.
- `galaxy` — moderate stretch, more sharpening, default saturation.
- `cluster` — gentler stretch (preserves star colors), lighter denoise,
  minimal sharpening (avoids ringing on star PSFs).

The web app and CLI pick one automatically when you don't specify
`--profile`. `backend/app/stages/classify.py` computes three cheap
metrics on the debayered image — median brightness of non-star pixels,
star density per megapixel, and largest connected bright-region area —
and applies hand-tuned thresholds to pick `nebula`, `galaxy`, or
`cluster`. The classification and raw metrics are logged at INFO level
and surfaced in the `/status` response so the UI can show which profile
was selected.

Add your own profile by appending to `PROFILES`. Each profile is a
nested dict `{stage_name: {param: value}}`. Any stage parameter you omit
falls back to the stage's built-in default.

## CLI

```sh
cd backend
python -m app.pipeline <input.fits> <output.png> [--profile NAME] [--no-auto] [-v]
```

By default the profile is picked automatically by the classifier. Pass
`--profile` to override, or `--no-auto` to fall back to `default` when
no profile is given.

Examples:

```sh
# auto-classify
python -m app.pipeline ../samples/NGC6992.fits ../samples/outputs/NGC6992.png -v

# explicit
python -m app.pipeline ../samples/nebula.fits  ../samples/outputs/nebula.png  --profile nebula -v
python -m app.pipeline ../samples/galaxy.fits  ../samples/outputs/galaxy.png  --profile galaxy
python -m app.pipeline ../samples/cluster.fits ../samples/outputs/cluster.png --profile cluster
```

`-v` prints per-stage progress to stderr. Exit code `0` on success.

## Backend API

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/health`                  | Liveness probe, returns `{"status": "ok"}`. |
| POST | `/process`                 | Multipart FITS upload. Starts a background pipeline job, returns `{"job_id": "…"}`. |
| GET  | `/status/{job_id}`         | `{status, stage, progress 0..1, classification, error}`. `status` is `queued`, `running`, `done`, or `error`. |
| GET  | `/result/{job_id}`         | The processed 16-bit RGB PNG. 409 if the job hasn't finished. |
| GET  | `/preview/{job_id}/before` | A simple log-stretched 8-bit preview of the input, for the slider. |

Job state lives in an in-process `dict` and pipelines run in a bounded
`ThreadPoolExecutor` (2 workers). No Celery, no Redis — v1 is a single
box. Scaling guidance is in [BACKLOG.md](BACKLOG.md).

## FITS format notes

- Input is expected to be a **2D Bayer mosaic** with a `BAYERPAT` header
  (or one of the accepted aliases). Seestar S30 and S50 both emit
  `RGGB`; this is also the assumed default when the header is silent.
- Normalization divides by `2^BITPIX - 1` when `BITPIX` is positive and
  integer; otherwise falls back to the data's own dtype range. Float
  inputs are not rescaled beyond `clip(0, 1)`.
- Already-debayered 3-channel FITS (shape `(3, H, W)` or `(H, W, 3)`) is
  accepted and skipped through the debayer step.

## Testing

```sh
cd backend
pip install -r requirements.txt
pytest -v
```

The suite covers:

- `tests/test_io_fits.py` — FITS load, shape/dtype/range, preview PNG.
- `tests/test_stages.py` — per-stage shape/range, plus behavioral checks
  (background reduces variance, color balances medians, stretch
  brightens, bm3d actually denoises noise, stubs raise NotImplementedError,
  export is 16-bit RGB).
- `tests/test_pipeline.py` — end-to-end run against a synthetic FITS,
  plus the progress-callback contract.
- `tests/test_profiles.py` — profile registry shape.
- `tests/test_classify.py` — metrics keys + nebula/galaxy/cluster rules.
- `tests/test_api.py` — FastAPI smoke: `/health`, upload → poll → result,
  404 / 400 error paths.

Tests that consume real sample files auto-skip when `samples/` is empty.

## Project structure in detail

```
backend/app/
├── __init__.py
├── main.py               FastAPI app. /health, /process, /status,
│                         /result, /preview. Mounts the built SPA at /.
├── pipeline.py           run(input, output, profile=None,
│                             progress=None, verbose=False) + CLI.
├── profiles.py           PROFILES dict + get(name).
├── models/               Reserved for v2 ONNX model weights.
├── static/               Built SPA (populated by Docker build).
└── stages/
    ├── io_fits.py
    ├── classify.py       Auto-picks nebula / galaxy / cluster profile.
    ├── background.py
    ├── color.py
    ├── stretch.py
    ├── bm3d_denoise.py
    ├── sharpen.py
    ├── curves.py
    ├── export.py
    ├── stars.py          v2 STUB (star removal). Raises NotImplementedError.
    └── ml_denoise.py     v2 STUB (ML denoise). Raises NotImplementedError.

frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts        Dev server proxies /process, /status, /result,
│                         /preview to :8000.
└── src/
    ├── App.tsx           Drop zone → progress → before/after slider.
    ├── index.css         Dark theme, minimal chrome.
    └── main.tsx
```

All stages are pure functions taking and returning numpy arrays, with no
module-level state. That property is what keeps the pipeline easy to
parallelize later (per-image, not per-stage) and easy to reason about in
tests.

## v2 roadmap

The v2 stubs are wired so that dropping in weights and removing a couple
of `NotImplementedError` lines is enough to enable the ML flow.

1. **Self-trained star removal** — U-Net or similar, trained on public
   star-field data, exported to ONNX under an MIT-compatible license.
   Target signature already in `stages/stars.py`:

   ```python
   def process(image, model_path, tile_size=512, overlap=32)
       -> tuple[stars_only, starless]
   ```

   Will slot into `pipeline.run()` after `stretch`, feed `starless` into
   the denoiser and sharpener, and screen-blend with `stars_only` before
   `curves`.

2. **ML denoise on starless** — fills `stages/ml_denoise.py` with the
   same tiled-inference pattern (256/512 px tiles, 32 px overlap, cosine
   window seam blend). Runs only on the starless layer to avoid
   softening star PSFs. `bm3d_denoise` remains available as a fallback.

3. **Tiled inference helper** — factor out the tiling+blending code into
   a single utility once both ML stages exist, so they share one tested
   implementation.

See [BACKLOG.md](BACKLOG.md) for the broader backlog, not just ML.

## Bundled model licenses

**v1 ships no model weights.** `backend/app/models/` is empty on
purpose, and `stages/stars.py` / `stages/ml_denoise.py` are
`NotImplementedError` stubs with stable signatures that v2 will fill.

The v1 image is therefore fully MIT-licensed end to end — every
runtime dependency (astropy, numpy, scipy, scikit-image, Pillow, pypng,
colour-demosaicing, bm3d, FastAPI, react, react-compare-slider) uses a
permissive license (MIT / BSD / Apache-2.0).

When v2 lands, any bundled weights will be listed here by name, file
path, SHA-256, and license. The [status section](#status--design-philosophy)
has the policy on which licenses are acceptable (MIT-compatible
permissive only — no CC BY-NC-SA weights, no GPL'd model checkpoints
smuggled in via dependency trees). Until then there's nothing in this
section beyond "none, by design".

## Contributing

Small patches welcome. A few guardrails:

- Don't add pipeline stages that require non-permissive model weights.
  The whole point of v1 is to keep the repo MIT-clean.
- Every stage must satisfy `(H, W, 3) float32 in [0, 1]` on input and
  output, and must be covered by at least one shape/range test plus one
  behavioral test.
- Prefer editing existing modules over adding new ones; `profiles.py` is
  the right place to expose a new parameter bundle, not a new file.
- Run `pytest` before pushing.

## References

- Malvar, He, Cutler (2004). *High-quality linear interpolation for
  demosaicing of Bayer-patterned color images.* ICASSP.
- Dabov, Foi, Katkovnik, Egiazarian (2007). *Image denoising by sparse
  3-D transform-domain collaborative filtering.* IEEE TIP. (BM3D.)
- Immerkaer (1996). *Fast noise variance estimation.* CVIU. (The MAD /
  Laplacian noise estimator used by `bm3d_denoise`.)
- Astropy Collaboration (2013, 2018, 2022). *Astropy: A community Python
  package for astronomy.*
