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

- [x] **Star preservation in `curves`** — _Shipped in `7ccb8d3`._
  `star_preserve_percentile` ramps saturation back toward 1.0 for the
  brightest ~1% of pixels so coloured stars keep their hue.
- [x] **Histogram-based auto-stretch knob** — _Shipped in `7ccb8d3`._
  `stretch.process(..., black_percentile="auto", stretch="auto")`
  picks both from the image's luma histogram (MAD-based knee for
  black, sky-to-target dynamic range for the arcsinh factor).
- [x] **Principled WB (Mahalanobis-trimmed sky median)** — _Shipped in
  `7ccb8d3`._ `mahalanobis_wb=True` in `color.process()` trims the sky
  sample cloud by Mahalanobis distance so coloured outliers (stars,
  nebula leakage) don't bias the channel balance.
- [x] **Background dark-sky mask pre-pass** — _Shipped in `7ccb8d3`._
  `background.process(..., sky_mask=bool_array)` drops grid samples
  outside the caller-supplied mask.
- [x] **Lanczos downsample before BM3D** — _Shipped in `7ccb8d3`._
  `bm3d_denoise.process(..., downsample_factor=N)` denoises at 1/N
  resolution and recovers high-frequency detail from the original;
  ~4× speed-up at factor 2 with no visible loss.
- [x] **Star-color protection in WB** — _Shipped in `7ccb8d3`._
  `star_protect_percentile` in `color.process()` tapers the WB gain
  toward 1.0 on the brightest pixels so red-M / blue-O types keep hue.
- [x] **Crop stage** — _Shipped in `7ccb8d3`._ `stages/crop.py`
  exposes `process(image, top, left, bottom, right)` with negative-
  edge convention.
- [x] **Luma-only sharpen** — _Shipped in `350db75`._ Per-channel unsharp
  was causing coloured fringing at demosaic edges; now the luma gets
  the high-pass and it's added equally across channels.
- [x] **Pre-stretch chroma equalization (high-pass + low-pass)** —
  _Shipped in `3829aca` and extended in `4c6b8af`._ High-pass kills LP
  / Bayer blobs before arcsinh; low-pass (new) averages pixel-scale
  chroma noise before arcsinh amplifies it into rainbow speckle.
- [x] **Per-channel black subtraction in stretch** — _Shipped in
  `3829aca`._ Each channel gets its own percentile floor removed before
  the shared luma-derived normalization.
- [x] **Luma-weighted channel gains in curves** — _Shipped in `4c6b8af`._
  Per-channel multiplicative boost applied post-S-curve and scaled by
  luma so dark pixels (sky) get near-identity but bright pixels
  (nebula) get the full colour shift. Used by nebula profile to
  restore the Ha-dominant R:G:B ratio.

## Reference-match improvements

Specific items needed to close the visible gap between our output and
the manually-processed references. Ordered by expected impact on the
hardest-case samples (NGC 6888 / NGC 2244 / NGC 6960).

- [ ] **Edge-preserving chroma denoise** — today's `chroma_blur` is an
  isotropic Gaussian that smears nebula colour into the sky at
  boundaries (visible as pink-tinted sky around the Rosette). Replace
  with a bilateral filter (or luma-guided joint bilateral) so chroma
  smoothing stops at luma edges. Expected: clean sky right up to the
  nebula edge, no colour bleed.
- [ ] **Hot-pixel / cosmic-ray rejection** — sigma-clipped stacking
  removes these; single-frame processing can't. Detect pixels > N×MAD
  brighter than their 3×3 neighbourhood and replace with median of
  non-rejected neighbours. Runs pre-stretch so the bright outliers
  don't get amplified into vivid rainbow dots.
- [ ] **Star PSF deconvolution** — a couple of Richardson-Lucy iterations
  with a synthetic Gaussian PSF tightens star cores before the stretch
  so they don't bloom into visible halos. Target stars: fewer, sharper,
  matching the reference's tight point-spread.
- [ ] **Local contrast enhancement (CLAHE or multi-scale unsharp)** —
  Rosette dust lanes and Veil filament edges benefit hugely from
  local contrast boosting at ~50-200 px scales. Classical CLAHE on
  luma would add this without any ML.
- [ ] **Starless-only processing branches** — extend the star-split
  pipeline so more stages run only on the starless layer: saturation
  boost, channel_gains, CLAHE, and curves should all apply to starless
  only. Stars then recombine via screen blend at their natural
  (un-saturated, un-gain-shifted) brightness.
- [ ] **Nebula profile sub-variants** — the current single nebula
  profile compromises between "wide diffuse" (Rosette) and "narrow
  filament" (Veil). Split into `nebula_wide` (less chroma_blur to keep
  dust lane contrast, less channel_gains) and `nebula_filament` (more
  chroma_blur to crush sky, more Ha gain). Classifier: filament =
  low largest_bright_fraction + narrow bright-region aspect ratio;
  wide = high largest_bright_fraction.
- [ ] **Filament/elongation metric in classifier** — currently NGC 6960
  barely squeaks past the cluster gate because its bright region is
  tiny. Measuring the aspect ratio (eccentricity) of the largest
  connected bright region would separate narrow filaments from round
  cluster cores at metric time, making classification deterministic
  for narrow-filament nebulae.
- [ ] **Sensor-specific color correction matrix** — Seestar's S50 has
  a known IR leak in R and the dual-band LP filter has specific
  transmission curves. A static 3×3 color correction matrix applied
  before WB (like dcraw's `CameraCalibration`) gets the linear
  channel balance into the right ballpark without heuristic tuning.
- [ ] **Dark-frame subtraction (when provided)** — allow the caller to
  supply a master dark (either through the CLI or a mount point on
  the container) and subtract it pre-background. Seestar app already
  does some calibration during stacking but raw dark-current patterns
  survive at the noise floor.
- [ ] **Hue-preserving saturation curve** — current saturation is a
  linear scale around per-pixel luma; on very saturated nebula pixels
  this can shift hue slightly. HSV or Lab-space saturation scaling
  preserves hue exactly and is what astro software uses.
- [ ] **Plate-solve + photometric color calibration (SPCC)** — the
  single biggest color-accuracy upgrade available, but requires a
  bundled star catalog (Gaia DR3 subset, ~200 MB) and an astrometric
  solver. Defer until either (a) the catalog can be fetched on-demand,
  or (b) we're okay breaking the "single-container offline" story.
- [ ] **ML star removal (replace classical)** — StarNet/StarXTerminator-
  quality ML star removal would eliminate the "median steals galaxy
  core" problem and give much cleaner separation on dense starfields
  (NGC 2244). Blocked on MIT-compatible weights — see ML stages
  section. Classical median is a functional fallback today.
- [ ] **ML denoise on starless** — NoiseXTerminator-class AI denoisers
  clean residual chroma/luma noise dramatically better than BM3D at
  the aggressive stretch levels emission nebulae need. Same blocker
  as above; BM3D is the fallback.

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
