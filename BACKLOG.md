# Backlog

Planned work beyond v1. Items are grouped by area, not by priority within
a group. Items marked **[v2]** are the headline v2 features; the rest are
quality-of-life, infra, and polish.

Legend: `[ ]` open, `[x]` done, `[~]` in progress or partially landed.

---

## ML stages (v2)

- [x] **[v2] Star removal** — _Shipped in `574ccb5`_ as a classical
  median-filter split (no model weights required, MIT-clean). Signature
  is stable so an ONNX implementation can drop in later:
  `stars.process(image, radius) -> (stars_only, starless)`. Enabled on
  the nebula profile; explicitly not used on galaxy (median steals the
  bright galactic nucleus) or cluster (stars ARE the subject).
- [ ] **[v2] Self-trained star removal (ML)** — small U-Net / ResUNet,
  trained on public star-field data, exported to ONNX. Would replace
  the classical implementation when the galaxy-core limitation matters.
  Bundle weights under an MIT-compatible license.
  - Tiled inference with 256 or 512 px tiles and 32 px overlap; cosine
    window for seam blending.
- [ ] **[v2] ML denoise on starless layer** — self-trained or
  permissively-licensed third-party weights.
  - Fills `backend/app/stages/ml_denoise.py::process`.
  - Runs only on the `starless` layer so star PSFs aren't softened.
  - Keep `bm3d_denoise` wired as a user-selectable fallback.
- [ ] **[v2] Shared tiled-inference helper** — once ML stages exist,
  factor out `backend/app/stages/_tiled.py` with:
  - `iter_tiles(shape, tile, overlap) -> Iterable[slice pairs]`
  - `cosine_window(tile) -> np.ndarray`
  - `blend(output_accum, weight_accum, tile_out, tile_window, slices)`
- [ ] Model registry: a small JSON/TOML manifest under
  `backend/app/models/` listing each bundled model's name, SHA-256,
  input/output tensor names, tile size, and license.
- [ ] Optional CUDA provider: detect and prefer `CUDAExecutionProvider`
  in `onnxruntime` when available, fall back to CPU.

## Pipeline improvements

- [ ] Star preservation in `curves`: current saturation boost can push
  colored stars to clipping; clamp saturation near the top of the
  histogram or apply it after a star mask.
- [ ] Histogram-based auto-stretch knob that picks both black and
  stretch-factor from the image, not just the black point.
- [ ] `color`: replace mid-tone mean-matching with a more principled
  white balance (e.g., median of Mahalanobis-trimmed sky pixels).
- [ ] `background`: expose a "dark sky mask" pre-pass that can be
  supplied by the caller instead of sigma-clipping from scratch.
- [ ] Optional Lanczos downsample before BM3D for very large images;
  upsample the noise estimate back. Halves runtime with minimal quality
  loss on the Seestar sensor size.
- [ ] Star-color protection step after `color`, so white-balance gains
  don't desaturate known stellar temperatures.
- [ ] Add a `crop.py` stage (simple rectangular crop), useful before
  heavy stages to iterate faster on a region of interest.
- [x] **Luma-only sharpen** — _Shipped in `350db75`._ Per-channel unsharp
  was causing coloured fringing at demosaic edges; now the luma gets
  the high-pass and it's added equally across channels.
- [x] **Pre-stretch chroma equalization** — _Shipped in `3829aca`._ High-
  pass filter on per-channel chroma before arcsinh kills the residual
  LP / Bayer color blobs that the stretch would otherwise amplify into
  visible blue/red sky patches. Gated per-profile (galaxy=on at σ=20,
  nebula=partial at σ=80, cluster=off).
- [x] **Per-channel black subtraction in stretch** — _Shipped in
  `3829aca`._ Each channel gets its own percentile floor removed before
  the shared luma-derived normalization, equalising sky pedestals across
  R/G/B so inter-channel offsets don't blow up post-arcsinh.

## Profiles & UX

- [x] **Three content-tuned profiles** — _shipped pre-session and
  iterated on extensively through `4794864`._ `nebula`, `galaxy`,
  `cluster`, with genuinely different strategies (nebula uses v2
  star-split, galaxy does not, cluster uses gentle stretch).
- [x] **Auto-classifier** — _shipped._ `classify.classify(image)`
  picks one of the three profiles from three metrics (median brightness,
  star-density-per-mpix, largest-bright-connected-region-fraction).
- [x] **Highlight-protection via `white_percentile`** — _shipped in
  `085c568`._ All three profiles now set this high enough that cores
  (M81 nucleus, M92 dense center, bright Veil stars) keep gradient
  instead of clipping to pure white.
- [ ] Profile for wide-field Milky Way shots (Seestar mosaic mode).
- [ ] Let profiles override only a subset of a parent profile (current
  implementation uses `{**DEFAULT, "stretch": {...}}` which replaces
  the whole sub-dict — fine for now, awkward as profiles grow).
- [ ] Expose `--override stage.param=value` flags on the CLI for
  one-off experimentation without editing `profiles.py`.

## Backend API

- [x] **`POST /process`** — _shipped._ multipart FITS upload, returns
  `{"job_id": ...}`. Includes magic-byte validation and a 500 MB cap.
- [x] **`GET /status/{id}`** — _shipped._ Returns stage, progress,
  classification, error.
- [x] **`GET /result/{id}`** — _shipped._ Downloads the 16-bit PNG.
- [x] **`GET /preview/{id}/before`** — _shipped._ Simple-stretched PNG
  of the raw input for the before/after slider.
- [x] **Thread-pool worker** — _shipped._ `ThreadPoolExecutor(max=2)`;
  no Celery, no Redis.
- [x] **TTL job reaper** — _shipped in `88ab0ab`._ Sweeps terminal jobs
  + their temp dirs after 1 hour; orphan sweep on startup.
- [x] **Rate-limit and size-cap** — _shipped in `4948a9a`._ Global
  in-flight cap (max_workers × 3) returns HTTP 429 when full; magic-byte
  upload validation rejects non-FITS early.
- [x] **Non-root container** — _shipped in `4948a9a`._ `USER appuser`
  (UID 10001) in the Dockerfile.
- [x] **FITS-bomb defence** — _shipped in `4948a9a`._ Manual header
  pre-parse rejects anything declaring >2 GiB of image data before
  astropy gets a chance to allocate.
- [ ] Persistent job store (SQLite) so the server can survive restarts.

## Frontend

- [x] **Single-page upload → progress → slider → download flow** —
  _shipped._ Drop zone, full-screen progress with stage labels and
  progress bar, react-compare-slider before/after, download + process-
  another buttons. Dark astrophotography theme.
- [x] **500 ms status polling with cleanup on unmount / completion** —
  _shipped._
- [ ] Stage-by-stage preview strip: show the image at each intermediate
  stage so users can see what each step is doing.
- [ ] ~~Parameter tweak panel that mirrors `profiles.py`~~ — **explicit
  non-goal per v1 UX spec ("no settings, no sliders, no advanced panel").**
- [ ] Persist uploaded files in the browser's OPFS or IndexedDB so a
  refresh doesn't re-upload.
- [ ] Basic theming / CSS modules instead of the current inline
  `index.css`.

## Testing

- [x] **Per-stage unit tests** — _shipped._ `backend/tests/test_stages.py`
  and `test_stars.py` cover 41 passing tests.
- [x] **API end-to-end test** — _shipped._ `backend/tests/test_api.py`
  exercises the full upload → poll → result flow against the FastAPI
  TestClient.
- [x] **Pipeline progress-callback test** — _shipped._ Verifies every
  declared stage emits a progress event.
- [ ] Property-based tests with `hypothesis`: generate random float32
  RGB images and assert the `[0, 1]` invariant is preserved by every
  stage.
- [ ] Golden-image tests: store one reference PNG per profile per sample
  and diff with a perceptual metric (e.g., SSIM ≥ 0.95). Guards against
  silent regressions in numeric behavior.
- [ ] Benchmark suite: `pytest-benchmark` timings per stage at a few
  resolutions, so we can see regressions over time.
- [ ] Frontend: basic Playwright smoke test that loads the SPA and
  checks the backend health status renders.

## DevOps / CI

- [ ] GitHub Actions workflow: `pytest` + `ruff` + `mypy` on every PR.
- [ ] Pre-commit hooks: `ruff format`, `ruff check`, trailing whitespace.
- [ ] Publish a multi-arch backend Docker image (amd64 + arm64) to
  GHCR on tagged releases.
- [ ] Dependabot for `pip` and `npm`.
- [ ] Renovate or a nightly job that rebuilds the Docker images against
  latest base images and runs the suite.

## Documentation

- [ ] A short "what's happening under the hood" page with before/after
  images per stage, suitable for linking from the README.
- [ ] Stage-authoring guide: the contract (`(H,W,3) float32 in [0,1]`),
  how to add to `profiles.py`, how to test.
- [ ] License & weights policy page spelling out the v1 MIT posture and
  the bar v2 weights must meet to be accepted.
- [ ] Troubleshooting: what to do when FITS files have unusual headers
  (missing `BAYERPAT`, unusual `BITPIX`, stacked/calibrated inputs).

## Nice-to-have

- [ ] Support for other smart-telescope FITS variants (Vespera, Dwarf,
  eVscope) — mostly a matter of reading their header conventions.
- [ ] Offer a grayscale pipeline path that skips demosaic and color.
- [ ] Export to TIFF and FITS in addition to PNG.
- [ ] Batch mode on the CLI: `python -m app.pipeline --batch samples/*.fits`
  writes matching outputs under `samples/outputs/`.
- [ ] Plate solving integration (astrometry.net) so outputs are WCS-stamped.
