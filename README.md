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
- [Troubleshooting](#troubleshooting)
- [Project structure in detail](#project-structure-in-detail)
- [Writing a new stage](#writing-a-new-stage)
- [v2 roadmap](#v2-roadmap)
- [Bundled model licenses & weights policy](#bundled-model-licenses--weights-policy)
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
3. While the pipeline runs, a **stage-by-stage preview strip** below
   the progress bar fills in left-to-right with thumbnails of the
   image after each stage (load → background → color → stretch →
   denoise → curves, etc.). Useful for watching the "reveal moment"
   at the stretch step and for diagnosing which stage produces a
   surprising look.
4. When the pipeline finishes, a **before/after slider** appears over
   the full viewport. Drag the handle to compare a simple log-stretched
   preview of the raw input against the enhanced result. The
   **Show stages** button on the toolbar reopens the stage strip as
   an overlay above the slider.
5. **Download PNG** saves the 16-bit RGB output. **Process another**
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

| Order | Stage | Module | Summary | What the pixels look like after |
| --- | --- | --- | --- | --- |
| 1 | `io_fits` | `stages/io_fits.py` | Reads FITS, detects Bayer pattern, debayers, returns float32 RGB in `[0,1]`. | Full-resolution linear RGB. Looks almost black — Seestar raw has its entire signal in the bottom 2% of the value range. |
| 2 | `background` | `stages/background.py` | Removes sky gradients via a thin-plate-spline RBF fit on sigma-clipped grid samples, per channel. | Still looks almost black, but gradients and light-pollution tint are gone — the frame is now flat. |
| 3 | `color` / `spcc` | `stages/color.py` / `stages/spcc.py` | Heuristic WB (Mahalanobis-trimmed sky median + mid-tone channel match) plus sensor CCM. Opt-in `spcc` fits per-channel gains against Gaia catalog photometry. | Channel ratios are now physically meaningful. Still linear, still dim. |
| 4 | `stretch` | `stages/stretch.py` | Per-channel black subtraction + arcsinh stretch with configurable black/white percentiles. | This is the "reveal" moment — nebulosity, galaxy halos, cluster members all appear at visible brightness for the first time. |
| 5 | `stars` (opt-in) | `stages/stars.py` | Median-based split into `stars_only` + `starless` layers. Subsequent stages run on the starless layer; stars screen-blend back after `curves`. | Two layers. The starless one shows only extended structure (nebula, galaxy disk). |
| 6 | `bm3d_denoise` | `stages/bm3d_denoise.py` | Block-Matching 3D denoise plus optional edge-aware chroma blur. Sigma auto-estimated from the luminance Laplacian. | Grain gone from sky/mid-tones; chroma noise smoothed without bleeding across nebula edges. |
| 7 | `sharpen` | `stages/sharpen.py` | Luma-only unsharp mask (per-channel caused colour fringing at demosaic edges). | Star cores and filament edges crisp up; chroma untouched. |
| 8 | `clahe` (opt-in) | `stages/clahe.py` | Luma-only CLAHE with a blend-back into the original. | Dust lanes and galaxy arm contrast get lifted locally without boosting overall brightness. |
| 9 | `curves` | `stages/curves.py` | Tanh S-curve, hue-preserving HSV saturation, per-channel luma-weighted gains, star-taper. | Final tonal shape and palette. Stars keep their native colour via the taper. |
| 10 | `export` | `stages/export.py` | Writes a 16-bit RGB PNG via `pypng`. | A PNG you can open. |

Stages 5, 8, `spcc`, `cosmetic`, `deconv`, and `dark_subtract` are
**opt-in** — they run only when the active profile sets their param
block. The core path (`io_fits → background → color → stretch →
bm3d_denoise → sharpen → curves → export`) is always on.

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

`backend/app/profiles.py` exposes six named profiles:

- `default` — conservative settings that look acceptable on most targets.
- `nebula` — stronger stretch, boosted saturation, heavier denoise +
  chroma-blur for faint diffuse targets (generic nebula).
- `nebula_wide` — for wide diffuse emission that fills most of the
  frame (Rosette, Crescent, North America). Gentler stretch, lighter
  chroma blur to preserve internal dust-lane structure. Opts in to
  **SPCC** when a Gaia catalog is bundled (see below).
- `nebula_filament` — for narrow filament structures (Veil segments,
  Bubble) that cover only a small fraction of the frame. Aggressive
  sky crush and higher colour-index boost. Opts in to SPCC.
- `galaxy` — moderate stretch, more sharpening, default saturation.
- `cluster` — gentler stretch (preserves star colors), lighter denoise,
  minimal sharpening (avoids ringing on star PSFs).

The web app and CLI pick one automatically when you don't specify
`--profile`. `backend/app/stages/classify.py` computes four cheap
metrics on the debayered image — median brightness of non-star pixels,
star density per megapixel, largest connected bright-region area, and
the eccentricity of that region — and applies hand-tuned thresholds
to pick a profile. The classification and raw metrics are logged at
INFO level and surfaced in the `/status` response so the UI can show
which profile was selected.

Add your own profile by appending to `PROFILES`. Each profile is a
nested dict `{stage_name: {param: value}}`. Any stage parameter you omit
falls back to the stage's built-in default.

### Photometric Colour Calibration (SPCC)

The nebula profiles (`nebula_wide`, `nebula_filament`) opt in to a
**photometric colour calibration** stage that fits the sensor's colour
response directly against known catalog-star photometry — the same
technique as Siril's SPCC or PixInsight's SPCC. No per-sensor
hand-tuning, no hidden assumptions: the calibration is driven by the
actual pixels in the frame plus a bundled star catalog.

**What SPCC needs:**

1. **A plate-solved FITS.** Seestar firmware embeds full 2-D WCS
   (`CTYPE/CRVAL/CRPIX/CD` plus 2nd-order SIP distortion) on every
   frame, so there's nothing to do — the pipeline reads the solution
   automatically. If the Seestar fails to plate-solve at capture
   time and emits a FITS without WCS, SPCC logs a skip and the
   fallback heuristic WB takes over.
2. **The Gaia DR3 bright-star catalog** at
   `backend/app/data/gaia_bright.parquet`. Not committed to the
   repository by default; generate it with
   ```sh
   cd backend
   python scripts/fetch_gaia.py
   ```
   The script tries the ESA Gaia archive first and falls back to the
   CDS Vizier mirror. Default cutoff `G < 12` produces ~5 M rows /
   ~30 MB; override with `GAIA_G_LIMIT=13` for a more generous cut.
   Gaia DR3 photometry is CC0 — safe to bundle in a downstream
   Docker image.

**How SPCC works here:** between `background` and `color`, the
pipeline detects the brightest 200 stars in the image, projects them
to RA/Dec via the WCS, cross-matches to the Gaia catalog (2″
tolerance), measures aperture photometry per matched star, compares
against Gaia BP/G/RP-derived target fluxes, and fits a diagonal 3×3
colour-correction matrix (per-channel gain) that best aligns measured
to target. Every early-out path logs why SPCC skipped (no WCS, no
catalog, too few matches).

**What it buys you:** objectively correct per-channel gains instead
of the heuristic Mahalanobis-trimmed mid-tone match. When SPCC fires,
the profile automatically drops its static Seestar CCM and its
`channel_gains` overrides — they'd compound on top of SPCC and
over-correct. Observed on the bundled samples (NGC 6888, NGC 2244,
NGC 6960): the colour balance moves from neutral-cyan toward the
Ha-red dominant look that manual PixInsight processing produces, with
~58–191 Gaia stars matched per frame.

**Mode:** `mode="diagonal"` (default, per-channel gain only) is the
classical SPCC approach and is robust to the passband mismatch
between Gaia BP/RP and typical consumer-camera blue/red filters.
`mode="full"` fits an unconstrained 3×3 and can over-correct hue on
broadband cameras; only use it when you trust the target photometry
completely.

## CLI

```sh
cd backend
python -m app.pipeline <input.fits> <output.png> \
    [--profile NAME] [--no-auto] \
    [--override stage.param=value]... \
    [-v]

# Batch mode — multiple inputs, one .png per input.
python -m app.pipeline --batch <input.fits> ... [--output-dir DIR] [-v]
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

### `--override`

Tune a single profile parameter for one run without editing
`profiles.py`. Repeatable; each override is `stage.param=value`. The
value is coerced to `int`, `float`, `bool` (`true`/`false`), `None`,
or left as a string, and a comma-separated value becomes a tuple.

```sh
# One-off stretch tweak
python -m app.pipeline input.fit output.png \
    --override stretch.stretch=22 \
    --override curves.contrast=0.75

# Channel gains need quoting so the shell doesn't split on comma
python -m app.pipeline input.fit output.png \
    --override 'curves.channel_gains=1.4,1.1,0.6'

# Disable a stage for one run
python -m app.pipeline input.fit output.png --override cosmetic=None
```

Overrides apply to the resolved profile (auto-classified or
`--profile`) without mutating the shared registry.

### `--batch`

Process many FITS files in one invocation. Each input produces a
`.png` with the same stem next to it, or under `--output-dir`:

```sh
# Output next to each input
python -m app.pipeline --batch samples/*.fit

# All outputs to one directory
python -m app.pipeline --batch samples/*.fit --output-dir samples/outputs

# Profile + overrides apply to every input in the batch
python -m app.pipeline --batch samples/*.fit \
    --profile nebula_wide \
    --override stretch.stretch=25

# Parallel across worker processes (best throughput for large batches)
python -m app.pipeline --batch samples/*.fit --jobs 4
python -m app.pipeline --batch samples/*.fit --jobs 0   # one per CPU core
```

`--jobs N` runs N pipelines concurrently in separate processes. BLAS
thread count per worker is auto-throttled (each worker gets
`cpu_count // N` BLAS threads) so the processes don't oversubscribe
the CPU. On a 16-core box, 4 workers on 4 images was ~1.45× faster
than sequential. Each worker costs ~300 MB of RSS plus BM3D's working
set, so keep `N × 350 MB` under free memory.

Per-file failures are logged and don't abort the batch; exit code is
`0` if all succeeded, `1` if any failed.

### Output formats

Single-file mode infers the format from the explicit output path's
suffix:

```sh
python -m app.pipeline input.fit  output.png    # 16-bit PNG (default)
python -m app.pipeline input.fit  output.tif    # 16-bit TIFF for PixInsight/Siril
python -m app.pipeline input.fit  output.fits   # float32 FITS, preserves pipeline precision
```

Batch mode picks the format via `--format` (defaults to `png`):

```sh
python -m app.pipeline --batch samples/*.fit --format tiff
python -m app.pipeline --batch samples/*.fit --format fits --output-dir samples/outputs
```

| Format | Bit depth | Best for |
| --- | --- | --- |
| `png`  | 16-bit per channel | Screen viewing, sharing, the web UI default. Universally supported. |
| `tiff` | 16-bit per channel | Further editing in PixInsight, Siril, GIMP, Photoshop. Lossless, zlib-compressed. |
| `fits` | float32 per channel | Re-stretching at full pipeline precision. RGB stored as a `(3, H, W)` cube per FITS NAXIS convention. |

The web UI exposes the same choice via a dropdown next to the drop
zone — your selection is sent as `?format=...` on the upload, and
`/result/{job_id}` serves the right `Content-Type` + filename.

### Performance notes

**Where time goes:** BM3D denoise is ~75 % of per-image wall time.
The pip `bm3d` package is CPU-only, so the per-image floor is a
function of your CPU, not your GPU.

**What accelerates on Intel hardware:**

- `--jobs N` (above) for batch throughput.
- numpy/scipy already use Intel MKL or OpenBLAS with AVX-2/AVX-512,
  so stretch, curves, sharpen, background are near-peak on any
  modern Intel CPU. No knob to turn.
- When v2 ML stages ship, ONNX Runtime with the OpenVINO provider
  will run models on Intel iGPU. Planned — see [BACKLOG.md](BACKLOG.md).

**What doesn't help on Intel hardware:**

- Intel QuickSync is a video-codec block (H.264/HEVC encode/decode),
  not general-purpose compute. Not applicable to this pipeline.
- `bm3dcuda`, `cupy`, `cucim` — NVIDIA-only. The BACKLOG tracks
  these as out-of-scope for the current target hardware.

## Backend API

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/health`                          | Liveness probe + coarse load summary. Returns `{status: "ok", load: "idle"|"busy"|"backed_up", inflight, running, queued, worker_capacity, max_inflight, recent_avg_seconds}`. The SPA's drop view fetches this once on mount to surface "Pipeline busy" when uploads will queue. |
| POST | `/process?format=png|tiff|fits`    | Multipart FITS upload. Starts a background pipeline job, returns `{"job_id": "…"}`. `format` is optional (defaults to `png`); chooses the output container — 16-bit PNG (default, screen-friendly), 16-bit TIFF (lossless, opens cleanly in PixInsight/Siril/GIMP), or float32 FITS (preserves full pipeline precision; RGB stored as a `(3, H, W)` cube per FITS NAXIS convention). Unknown values 400. |
| GET  | `/status/{job_id}`                 | `{status, stage, progress 0..1, classification, error, stages_done, queue_position, queue_total, eta_seconds, worker_capacity}`. `status` is `queued`, `running`, `done`, or `error`. `queue_position` is 1-based (1..worker_capacity = currently running, beyond = waiting). `eta_seconds` comes from a rolling average of recent pipeline durations. |
| GET  | `/result/{job_id}`                 | The processed 16-bit RGB PNG. 409 if the job hasn't finished. |
| GET  | `/preview/{job_id}/before`         | A simple log-stretched 8-bit preview of the input, for the slider. |
| GET  | `/preview/{job_id}/stage/{stage}`  | A 300×300 thumbnail PNG of the in-flight image after a named stage runs. The frontend's stage-strip renders these as the pipeline progresses. 404 for unknown stage names; 409 while the named stage is still in flight. |

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

### Lint, format, and type-check

Ruff (lint + format) and mypy run on every push and pull request via
`.github/workflows/ci.yml`. Run them locally before pushing:

```sh
cd backend
ruff check .              # lint
ruff format --check .     # format (add `--fix` to auto-apply)
mypy app                  # type check (advisory for now)
```

Pre-commit hooks mirror the CI checks. Opt in once per clone:

```sh
pip install pre-commit
pre-commit install
```

After that, every `git commit` runs ruff + format + a few file-hygiene
hooks and blocks the commit on failure.

### Golden-image regression tests

`tests/test_golden.py` runs the full pipeline on each of the six
sample FITS files and compares the downscaled 400×400 output to a
committed golden PNG via SSIM ≥ 0.95. The golden bundle lives in
`backend/tests/golden/` (six PNGs, ~1.1 MB total).

These catch silent drift from profile tunes, stretch-curve changes,
or colour-stage regressions that the per-stage unit tests miss. They
take ~7 minutes end to end because BM3D is the bottleneck.

When a profile change is intentional, regenerate the goldens:

```sh
cd backend
REGEN_GOLDEN=1 pytest -k test_golden
```

and commit the updated PNGs. The test auto-skips when `samples/` is
absent, so contributors without the raw data still run the other
110 tests.

## Troubleshooting

Common issues and what to do about them.

### "Unsupported BITPIX" or wildly over/under-exposed output

Input FITS uses a `BITPIX` the loader doesn't recognise, or the data
was saved with a non-standard scaling. `io_fits.load_fits` normalises
positive integer `BITPIX` by `2^BITPIX - 1`; float inputs are clipped
to `[0, 1]` with no rescaling, which means a float FITS containing
values in `[0, 10000]` will saturate.

- Fix: re-export the file as uint16 (`BITPIX=16`) with the data
  rescaled into the dtype's full range, or pre-scale yourself and
  save as uint16.
- Workaround: subclass `io_fits.load_fits` and add rescaling for the
  specific `BITPIX` / data-range combo.

### "Bayer pattern not recognised" / odd chromatic tint

`io_fits` looks for `BAYERPAT`, then `BAYRPAT`, `COLORTYP`, `CFAPAT`.
If none are present it assumes `RGGB` (Seestar S30/S50 default).
Non-Seestar FITS with a different Bayer pattern will produce colour
channels swapped after demosaic.

- Fix: make sure your capture software writes the `BAYERPAT` header.
  All major capture suites (SharpCap, N.I.N.A., Seestar firmware)
  do this by default.
- Workaround: manually patch the header before running:
  ```python
  from astropy.io import fits
  with fits.open("capture.fits", mode="update") as f:
      f[0].header["BAYERPAT"] = "GBRG"
  ```

### "Input is already 3-channel, skipping debayer"

You fed in an already-debayered / stacked RGB FITS (shape `(3, H, W)`
or `(H, W, 3)`). This is supported — the loader skips demosaic and
passes the data through. But stacked data has usually already had
some form of background + stretch applied, so running the full
pipeline on top can double-process it.

- Fix: if you have an unprocessed stack, you typically want to skip
  `background` and use a gentler `stretch`. Either route the file
  through `profile=default` with `--profile default` and edit that
  profile's `stretch.stretch` down, or write a custom profile that
  sets `background: {}` to no-op it.

### "SPCC: skipped (no WCS in FITS header)"

SPCC needs a plate-solve. Seestar firmware usually embeds a full 2D
WCS + SIP distortion, but if the telescope fails to solve at capture
time (clouds, short exposure, no reference stars) the FITS can land
without WCS.

- Fix: this isn't a bug — SPCC skips and the heuristic WB takes
  over. If you want SPCC-level accuracy you need a solved image;
  either re-capture, or pre-solve with astrometry.net / nova.astrometry
  and inject the WCS headers.

### "SPCC: only N < min_matches stars matched"

Your image has fewer than the profile's `min_matches` (default 20)
stars cross-matching to Gaia within the 2″ tolerance. Usually means
the image has very few detectable stars (deep nebulosity, short
exposure) or the WCS is off by more than 2″.

- Fix: lower `min_matches` in the profile, or widen `tol_arcsec`
  to 4-5″. If still failing, the WCS is suspect; try re-solving.

### Golden-image tests failing after a tune

Intentional profile changes that shift the output's hue, brightness,
or structure will drop SSIM below 0.95 and fail `test_golden_matches`.

- Fix: once you're happy with the tune, regenerate the golden:
  ```sh
  cd backend
  REGEN_GOLDEN=1 pytest -k test_golden
  git add tests/golden/ && git commit
  ```
- Diagnostic: compare the actual vs golden side-by-side. If the
  diff is much bigger than you expected, you may have a real
  regression rather than a tune.

### "HTTP 429: too many jobs in flight"

The backend caps concurrent jobs at `max_workers × 3` (default 6) to
protect against a client flooding the queue. Wait for running jobs
to finish, or raise `max_workers` / the multiplier in
`backend/app/main.py` for a local deployment that can absorb more.

### Seestar FITS with unusual headers

- **No `BAYERPAT`**: assumed `RGGB` (Seestar default). Only wrong for
  non-Seestar sources.
- **Negative `BITPIX`** (float data): normalisation is skipped, data
  is clipped to `[0, 1]`. Pre-stacked float FITS with values > 1
  will saturate. Rescale before running.
- **Already-calibrated / stacked**: see "Input is already 3-channel"
  above. You may want to tune the profile rather than run the full
  v1 pipeline.

### Catalog / Parquet issues

- `Gaia catalog not found`: `backend/app/data/gaia_bright.parquet`
  didn't come through the clone. Run
  `python backend/scripts/fetch_gaia.py` to regenerate (takes ~3 min,
  needs network).
- **Catalog too big for git / push warning**: the bundled 80 MB
  full-sky catalog exceeds GitHub's 50 MB soft limit and will emit
  a `GH001` warning on push. The push still succeeds; migrating the
  file to Git LFS is [tracked in BACKLOG](BACKLOG.md).

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

## Writing a new stage

Adding a stage takes ~50 lines of code plus tests. The hard part is
deciding *where* in the pipeline it goes and what parameters it
exposes; the mechanical parts are constrained by the stage contract.

### The stage contract

```python
def process(image: np.ndarray, **params) -> np.ndarray: ...
```

- `image` is **always** shape `(H, W, 3)`, dtype `float32`, values in
  `[0, 1]`. Do not accept or return other shapes/dtypes/ranges.
- Return a new array of the same shape/dtype/range. Stages are pure —
  no module-level state, no in-place mutation of the input.
- Parameters are keyword-only and every one has a default. This lets
  `profiles.py` override any subset without specifying the full set.
- Invalid inputs raise `ValueError` with a useful message; the pipeline
  does not silently fall back.

The contract is tight on purpose. It lets the pipeline compose stages
in any order without shape/dtype adapters and makes each stage
independently testable against a single synthetic input.

### Anatomy of a minimal stage

```python
# backend/app/stages/my_stage.py
"""One-line summary of what this stage does.

Longer description if useful — WHY we need this stage, WHAT it does
to the pixels, and any constraint the caller should know about.
"""
from __future__ import annotations

import numpy as np


def process(
    image: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    out = image.astype(np.float32, copy=True)
    # ... do work ...
    return np.clip(out, 0.0, 1.0).astype(np.float32)
```

### Wire it into the pipeline

`backend/app/pipeline.py` has the stage order as a sequence of
`if params.get("my_stage") is not None: img = my_stage.process(img,
**params["my_stage"])` blocks. New stages land between existing
stages at the appropriate point in the flow — most new post-processing
goes between `bm3d_denoise` and `curves`; anything that needs linear-
space input goes before `stretch`.

### Expose it in a profile

```python
# backend/app/profiles.py
NEBULA = {
    **DEFAULT,
    # ... existing stage overrides ...
    "my_stage": {"strength": 0.6},
}
```

Any stage parameter you omit falls back to the stage's built-in
default. Setting a stage to `None` in a profile disables it outright
(useful for sub-profiles that opt out of a parent's opt-in stage).

### Test it

Two tests minimum per stage:

1. **Shape/range test** — the contract. Throw a synthetic
   `(H, W, 3) float32 in [0, 1]` at it and verify the output matches.
2. **Behavioural test** — verify the stage actually does what it
   says. For `bm3d_denoise`, that's "noisy input has lower variance
   after"; for `sharpen`, "edges have higher gradient after"; for
   `background`, "gradient is smaller after". One crisp assertion is
   enough — these are smoke tests, not full characterizations.

Put them in `backend/tests/test_stages.py` alongside the existing
ones (grouped by stage via comments), or — if the stage is
substantial — in its own `tests/test_my_stage.py`. Both conventions
exist in the tree.

If the stage has user-visible visual impact on real samples, it'll
automatically get covered by the golden-image tests once you run
`REGEN_GOLDEN=1 pytest -k test_golden`.

### Check the guardrails

Before you push:

```sh
cd backend
pytest                                   # all tests, including golden
ruff check . && ruff format --check .    # lint + format
```

CI runs both on push and pull request.

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

## Bundled model licenses & weights policy

**v1 ships no model weights.** `backend/app/models/` is empty on
purpose, and `stages/stars.py` / `stages/ml_denoise.py` are
`NotImplementedError` stubs with stable signatures that v2 will fill.

The v1 image is therefore fully MIT-licensed end to end — every
runtime dependency (astropy, numpy, scipy, scikit-image, Pillow, pypng,
colour-demosaicing, bm3d, FastAPI, react, react-compare-slider) uses a
permissive license (MIT / BSD / Apache-2.0).

### Policy

The project is MIT-licensed and intends to stay that way. Any weights
bundled into the distribution must pass **all four** of the following:

1. **Permissive license.** MIT, BSD-2/3, Apache-2.0, Unlicense, CC0,
   or a comparably permissive custom license. **Not acceptable:**
   CC BY-NC (non-commercial), CC BY-SA / CC BY-NC-SA (ShareAlike
   viral), GPL / LGPL (copyleft), source-available licenses with
   field-of-use restrictions.
2. **Attribution traceable.** The author(s), upstream URL, training
   data provenance, and exact license text must be recorded in the
   model registry manifest.
3. **SHA-256 pinned.** Weights are identified by hash, not tag. The
   manifest records the hash; the build step verifies it.
4. **Inference-only.** Weights are bundled for inference. Training
   pipelines and training data are separate concerns and are not
   required to be permissive.

Reasoning: CC BY-NC-SA kills commercial self-hosting, ShareAlike
forces every downstream app to re-license, and GPL model checkpoints
can contaminate the runtime if any linkage theory applies (the
industry consensus is unclear — we avoid the question by rejecting
GPL weights outright).

### What's been rejected and why

| Model | Reason |
| --- | --- |
| StarNet++ v2 | Weights are CC BY-NC-SA 4.0 — non-commercial + ShareAlike. Both clauses are blockers. |
| GraXpert AI denoise | Code is GPL-3.0; weights license is unspecified and the runtime downloads them from S3 (no reproducibility, no pinning). |
| StarXTerminator / NoiseXTerminator | Commercial, proprietary weights. |

Until a permissive alternative exists, v1 ships classical fallbacks
(median star split, BM3D denoise) for the same capabilities.

### Bundled weights

**None.** Once v2 lands, each bundled file will be listed below by
path, SHA-256, and license, e.g.:

```
backend/app/models/starnet_mit.onnx
  SHA-256: <hex>
  License: MIT
  Upstream: <url>
```

The `backend/app/models/` directory is reserved for this purpose and
is empty today.

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
