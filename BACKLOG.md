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
- [ ] Optional hardware providers for `onnxruntime`: detect and prefer
  `OpenVINOExecutionProvider` on Intel CPU/iGPU (the project's target
  hardware), fall back to CPU. A `CUDAExecutionProvider` path would
  cover NVIDIA users if there are any — but see the Hardware
  acceleration section in Pipeline improvements for the scope call.

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
- [x] **Star-taper on channel_gains in curves** — _Shipped in `502d705`._
  Aggressive warm gains were tinting stars yellow-green because stars
  sit at luma≈1.0 and got the full gain applied. The curves stage
  now tapers `channel_gains` back toward identity at the top
  `star_preserve_percentile` (default 99.0), mirroring the existing
  saturation taper. Lets nebula profiles use strong Ha-red gains
  without fouling star colours.

## Reference-match improvements

Specific items needed to close the visible gap between our output and
the manually-processed references. Ordered by expected impact on the
hardest-case samples (NGC 6888 / NGC 2244 / NGC 6960).

- [x] **Edge-preserving chroma denoise** — _Shipped._ New
  `chroma_edge_aware=True` option in `bm3d_denoise.process()` uses a
  luma-gradient-weighted blend between smoothed and original chroma
  so colour stops bleeding across nebula→sky edges. Nebula profile
  uses it at `chroma_blur=120, chroma_edge_luma_sigma=0.08`.
- [x] **Hot-pixel / cosmic-ray rejection** — _Shipped as
  `stages/cosmetic.py`._ Pre-background median-filter outlier replace
  with a MAD-based rejection threshold. Opt-in per profile; nebula
  uses `neighborhood=3, sigma=6.0`.
- [x] **Star PSF deconvolution** — _Shipped as `stages/deconv.py`._
  Richardson-Lucy on luma with a synthetic Gaussian PSF. Classical,
  opt-in per profile. Nebula profile currently has it OFF by default
  — RL amplifies shot noise on faint-target stacks more than it helps;
  the stage is available for galaxy profiles or cleaner data.
- [x] **Local contrast enhancement (CLAHE)** — _Shipped as
  `stages/clahe.py`._ Luma-only CLAHE with a user-controllable blend
  into the original luma. Same rationale as deconv: on clean data
  this lifts dust-lane / filament contrast; on noisy-sky stacks it
  amplifies noise, so nebula profile has it OFF by default.
- [x] **Starless-only processing branches** — _Shipped._ Pipeline now
  runs `bm3d_denoise`, `sharpen`, `clahe`, and `curves` on the
  starless layer when `stars` is set on the profile, then screen-
  blends the unmodified stars back after `curves`. Stars never see
  the saturation / channel_gains / CLAHE adjustments so they keep
  their natural brightness and colour.
- [x] **Nebula profile sub-variants** — _Shipped._ `nebula_wide`
  (Rosette / Crescent-class: gentler stretch, lighter chroma_blur to
  preserve internal dust lanes) and `nebula_filament` (Veil-class:
  aggressive sky crush, larger chroma_blur, higher Ha gain). Classifier
  routes on `largest_bright_fraction` and the elongation metric.
- [x] **Filament/elongation metric in classifier** — _Shipped._ The
  `largest_bright_elongation` metric (second-moment eigenvalue ratio
  of the largest connected bright region) now drives the
  filament-vs-cluster decision — 0 for round blobs, → 1 for narrow
  lines. NGC 6960 (elong ≈ 0.28 because its bright region is the
  star-field itself) routes to `nebula_filament` via the density-or-
  elongation fallback; Veil cores with elongated filaments also land
  there deterministically.
- [x] **Sensor-specific color correction matrix** — _Shipped._
  `color.process(..., ccm="seestar_s50")` applies a 3×3 linear
  transform before the WB heuristics. The Seestar S50 CCM trims R
  for the broadband-filter IR leak, trims G for the RGGB double-
  green, and lifts B slightly. Accepts arbitrary user-supplied 3×3
  matrices too. All nebula profiles enable it.
- [x] **Dark-frame subtraction** — _Shipped as `stages/dark_subtract.py`
  and as a `dark=` kwarg on `pipeline.run()`._ Caller supplies a
  master-dark array; stage subtracts it pre-background with an
  optional `scale`. No-op when `dark=None`, so existing flows are
  unaffected.
- [x] **Hue-preserving saturation curve** — _Shipped._ New
  `saturation_mode="hsv"` in `curves.process()` converts to HSV,
  scales only S, converts back. Hue-preserving by construction. All
  nebula profiles use HSV mode; galaxy/cluster profiles keep the
  default linear mode.
- [x] **Plate-solve + photometric color calibration (SPCC)** —
  _Shipped across five phases on `claude/v2-start-spcc-phase-1-gaia-catalog`._
  Key insight that made this tractable: Seestar FITS files already
  embed the full astrometric solution, so we skip blind
  plate-solving entirely. Gaia DR3 photometry is CC0 so the catalog
  bundles directly. Hit-list:
    - Phase 1 (`0ad96dc`): `scripts/fetch_gaia.py`, `astroquery` +
      `pyarrow` deps, `app/data/__init__.py`, `test_catalog.py`.
    - Phase 2 (`8e80b19`): `load_fits_with_wcs`, empty SPCC stage
      wired into pipeline.
    - Phase 3 (`6823871`): private helpers —
      `_detect_bright_stars`, `_catalog_in_fov`, `_cross_match`,
      `_measure_star_rgb`. 13 synthetic tests.
    - Phase 4 (`e1968cc`): real `process()` with catalog load,
      cross-match, `_gaia_to_target_rgb`, lstsq fit, apply.
      Opt-in on `nebula_wide` + `nebula_filament`. 8 new tests.
    - Phase 5 (this commit): tuned profiles to disable static CCM
      and `channel_gains` when SPCC is active (they compound and
      swing cyan); added `mode="diagonal"` default (classical SPCC,
      robust to Gaia BP/RP vs camera RGB passband mismatch); Gaia
      Parquet bundled via region-specific cone-search (32.7k stars
      covering the six sample fields, 1 MB — full-sky can be
      regenerated any time via `fetch_gaia.py`); README documents
      the stage and data dependency.

  **Observed on the samples:** diagonal SPCC fits per-channel gains
  in the (R ~1.5, G ~0.7, B ~0.7-0.9) range across the four
  nebula frames — boosting red, cutting green, consistent with the
  Seestar S50's known sensor biases (RGGB green dominance + IR
  leak in R). Outputs now land closer to the Ha-red hue of manual
  PixInsight processing than the heuristic pipeline alone did.

  The phased plan itself stays below, left in place as a record of
  how the work was planned and where each phase landed.

  **Phased plan — five phases at ≤ 6 hours of focused work each:**

  - [x] **Phase 1 — Gaia DR3 catalog bundle (~3-4 h)**
    - Add `astroquery` + `pyarrow` to `backend/requirements.txt`.
    - Write `backend/scripts/fetch_gaia.py`: ADQL query against
      `https://gea.esac.esa.int/tap-server/tap`, filters to
      `phot_g_mean_mag < 13` (~5M rows, ~30 MB as Parquet with
      snappy compression).
    - Columns: `ra`, `dec`, `phot_g_mean_mag`, `phot_bp_mean_mag`,
      `phot_rp_mean_mag`. That's all SPCC needs.
    - Save to `backend/app/data/gaia_bright.parquet` and commit the
      file directly (< 50 MB is git-native territory).
    - Update `Dockerfile` so the `data/` directory is copied into
      the container (already in `app/` so it's likely already
      happening via `COPY backend/ ./`).
    - Write `backend/tests/test_catalog.py` that loads the Parquet
      and asserts schema, row-count floor, magnitude range.
    - Commit message: "Phase 1: bundle Gaia DR3 bright-star catalog"
    - Exit criteria: one new file, one test passing, `pip install -r
      requirements.txt` still works on a fresh container.
    - **Status:** infrastructure landed in `0ad96dc`; data file
      landed in `9888e63` after ESA recovered — see the follow-up
      item below for how the 3M-row cap got worked around.

  - [x] **Phase 1 follow-up — actually run `fetch_gaia.py` and commit
    the Parquet**. _Shipped in `9888e63`._ The ESA archive recovered
    (small probes return in ~1 s with an "archive unstable" banner
    but functional responses). Gotcha discovered: ESA async TAP caps
    results at 3,000,000 rows server-side — a single G<12 full-sky
    query silently truncates. Fix was to split the ADQL into four
    90° RA bands and vstack to 3.08M rows. Catalog is 84 MB
    (brotli + float32 ra/dec). All 288 sky tiles populated, up from
    ~50 in the regional workaround. Historic context below:
    - Try `python backend/scripts/fetch_gaia.py` (defaults to G < 12,
      auto-fallback ESA → Vizier). Expect ~30 MB output.
    - If ESA is still 500-ing, set `GAIA_SOURCE=vizier` and wait
      longer (Vizier is slow but doesn't 500).
    - Commit the resulting `backend/app/data/gaia_bright.parquet`
      with message "Phase 1 data: bundle Gaia DR3 bright-star
      Parquet" and push. The 5 tests in `test_catalog.py` stop
      being skipped the moment the file lands.
    - Blocker when Phase 1 closed: both ESA Gaia TAP
      (`gea.esac.esa.int/tap-server/tap`) and CDS Vizier
      (`tapvizier.u-strasbg.fr/TAPVizieR/tap`) were returning
      HTTP 500 / timing out during the session. The archive
      welcome page explicitly warned it was unstable.

  - [x] **Phase 2 — WCS plumbing through the pipeline (~2-3 h)**
    - Modify `stages/io_fits.py::load_fits()` to optionally return a
      `WCS` object (add a sibling `load_fits_with_wcs()` that returns
      `(image, wcs_or_none)`; keep the existing function as a
      backwards-compat wrapper calling the new one and dropping the
      WCS).
    - Modify `pipeline.run()` to call the new variant and thread the
      WCS into a new optional `spcc` stage that's wired but empty.
    - Add a WCS-reading test against one of the Seestar samples
      (verify a known RA/Dec round-trips through pixel coords).
    - Commit message: "Phase 2: load and thread WCS through pipeline"
    - Exit criteria: all existing tests still pass (77 + 1 new test
      for WCS). No behavioural change to output.

  - [x] **Phase 3 — SPCC helper functions (~4-5 h)**
    - New module `stages/spcc.py` with private helpers:
      - `_detect_bright_stars(image, n=100) → ndarray[(N,2) int]` —
        returns (y, x) of the N brightest local maxima above a MAD-
        threshold, 7-px suppression window. (Adapt the code already
        in `classify._metrics` — extract once, reuse.)
      - `_catalog_in_fov(cat_df, wcs, margin_deg=0.2) → cat_df` —
        filters the catalog to stars whose RA/Dec projects inside
        the image (+margin). Uses `wcs.all_world2pix`.
      - `_cross_match(img_stars_radec, cat_stars_df, tol_arcsec=2.0)
        → (img_idx, cat_idx)` — nearest-neighbor in Dec-scaled flat
        RA/Dec space via `scipy.spatial.cKDTree`. Rejects unmatched
        stars.
      - `_measure_star_rgb(image, stars, aperture_r=3) → ndarray[(M,3)]`
        — aperture photometry on each detected star, median of pixels
        inside `r` minus median of annulus outside.
    - Unit tests for each helper with synthetic inputs (a fake 200x200
      image, three hand-placed stars, a mock catalog DataFrame).
    - Commit message: "Phase 3: SPCC star detection + cross-match"
    - Exit criteria: tests pass, no pipeline wiring yet (helpers are
      private, so they can change signature in phase 4).

  - [x] **Phase 4 — SPCC CCM fit and stage integration (~4-5 h)**
    - In `stages/spcc.py`, add the public `process()`:
      1. Read catalog from `app/data/gaia_bright.parquet`.
      2. Run the helpers from phase 3.
      3. For each matched pair, collect image-RGB and the Gaia
         synthetic sRGB (computed from BP, RP, G via the Gaia/sRGB
         transformation from DR3 documentation).
      4. Solve `image_rgb @ M = target_rgb` via `np.linalg.lstsq`,
         where `M` is the 3x3 CCM that best maps our sensor to sRGB.
      5. Apply `image @ M.T`.
      6. If fewer than `min_matches` pairs (default 20), log a warning
         and return the image unchanged.
    - Wire into `pipeline.run()` as an opt-in stage: profile sets
      `"spcc": {...}` and the stage runs between background and the
      heuristic WB in `color`. The static `SEESTAR_S50_CCM` in color
      stays as the fallback when SPCC is disabled or fails.
    - Profile update: add `"spcc": {"min_matches": 20}` to NEBULA_WIDE
      and NEBULA_FILAMENT (the two most sensitive to colour accuracy).
      Galaxy / cluster can enable in a later tuning pass.
    - Synthetic end-to-end test: construct a fake image + WCS + CSV
      catalog where the "true" CCM is identity; verify SPCC fits
      identity within 1 %.
    - Commit message: "Phase 4: SPCC CCM fit + pipeline wiring"
    - Exit criteria: all tests pass, real sample run completes
      without errors, SPCC logs the number of matched stars.

  - [x] **Phase 5 — Tuning + docs + close-out (~3-4 h)**
    - Run all 6 samples end-to-end with SPCC on; compare to the
      references the user supplied. Tune the nebula profiles:
      typically the static CCM and `channel_gains` can be relaxed
      once SPCC is doing the calibration work.
    - If SPCC ever fails on the samples (low match count), investigate
      and either adjust thresholds or gracefully fall back.
    - Update `README.md` to document the SPCC opt-in and its data
      dependency.
    - Mark item 11 as shipped in this file with the commit hash.
    - Commit message: "Phase 5: tune profiles with SPCC, close
      reference-match section"
    - Exit criteria: item 11 marked done, all tests pass, the "what
      it buys us" items from the proposal are visible on at least
      one of the samples.

  **Total effort:** ~17-22 hours across the five phases, each sized
  to comfortably fit a 6-hour session. Phases can be run on
  consecutive sessions or stretched across days.

  **Rollback:** each phase is independently revertable — phase 1
  only adds a data file and a script, phase 2 only adds an optional
  return value, phase 3 adds private helpers, phase 4 wires an
  opt-in stage, phase 5 is profile tuning. Any phase can be held at
  its commit boundary while the rest of the pipeline continues to
  work.
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
- [x] **Nested profile merging** — _Shipped._ `profiles.merge(parent,
  **overrides)` recursively merges override dicts onto a deep copy of
  the parent. Sub-profiles only specify keys they're actually changing;
  sibling keys at every nesting level are inherited automatically.
  Replaces the `{**PARENT, "color": {**PARENT["color"], "ccm": None}}`
  pattern that silently wiped sibling sub-keys when the spread was
  forgotten. NEBULA / NEBULA_WIDE / NEBULA_FILAMENT / NEBULA_DOMINANT
  / GALAXY / CLUSTER all ported to the new shape; output unchanged
  (golden tests pass). 11 new tests in `test_profiles.py`.
- [x] **`--override stage.param=value` flag on the CLI** — _Shipped._
  `python -m app.pipeline ... --override stretch.stretch=22 --override
  'curves.channel_gains=1.4,1.1,0.6'` patches the resolved profile
  for one run without mutating the shared registry. Values coerce to
  int / float / bool / None / tuple automatically.

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
- [x] **FITS-bomb defence** — _shipped in `4948a9a`, hardened later._
  Manual header pre-parse rejects anything declaring >2 GiB of image
  data before astropy gets a chance to allocate. Hardening pass
  extended the validator to walk extension HDUs and inspect
  `ZIMAGE`/`ZNAXIS*` keywords so a tile-compressed BINTABLE that
  declares a multi-GB decompressed array is rejected at the gate.
- [x] **CORS allowlist + security headers** — _shipped._ Default
  `allow_origins` restricted to localhost (`8000` + `5173`); operators
  fronting the API with a known SPA host set the `ALLOWED_ORIGINS`
  comma-separated env var. New `_SecurityHeadersMiddleware` adds
  `Content-Security-Policy`, `X-Content-Type-Options`,
  `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer` to
  every response. `_friendly_error` strips the per-job temp path
  out of exception messages so internal layout doesn't leak.
- [ ] Persistent job store (SQLite) so the server can survive restarts.

## Frontend

- [x] **Single-page upload → progress → slider → download flow** —
  _shipped._ Drop zone, full-screen progress with stage labels and
  progress bar, react-compare-slider before/after, download + process-
  another buttons. Dark astrophotography theme.
- [x] **500 ms status polling with cleanup on unmount / completion** —
  _shipped._
- [x] **Stage-by-stage preview strip** — _Shipped._ `pipeline.run()`
  takes an opt-in `stage_preview_dir` parameter that writes a 300×300
  thumbnail PNG named `<stage>.png` after every stage runs (auto-
  applies a log-stretch preview when the stage output is still
  linear, so pre-stretch thumbs aren't black). The web UI's job
  status surfaces a `stages_done` list, and a new
  `/preview/{job_id}/stage/{name}` endpoint serves each thumbnail.
  The SPA renders the strip below the progress bar during processing
  and as an opt-in overlay on the done view (Show stages button).
  ~1.1 MB of thumbnails per job, cleaned up by the existing TTL
  reaper.
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
- [x] **Property-based tests with `hypothesis`** — _Shipped._
  `tests/test_stage_invariants.py` runs 25 hypothesis-driven examples
  against each stage (background / color / stretch / bm3d_denoise /
  sharpen / curves / cosmetic / dark_subtract / crop / classify) plus
  15 explicit constant-frame edge cases. Verifies the `(H, W, 3)
  float32 in [0, 1]` contract end-to-end. The first run found a real
  bug: BM3D's upstream library divides by zero on constant or
  axis-constant inputs and returns NaN, which would NaN the entire
  pipeline on a corrupt FITS. Fixed in `bm3d_denoise.process()` with
  a zero-variance early-out and a NaN-pixel fallback to original.
- [x] **Golden-image tests** — _Shipped in `e54ac8c`._
  `backend/tests/test_golden.py` runs the full pipeline on each of the
  six sample FITS and compares the 400×400 downscaled output to a
  committed golden PNG via SSIM ≥ 0.95. Six goldens total, ~1.1 MB
  bundle under `backend/tests/golden/`. `REGEN_GOLDEN=1 pytest -k
  test_golden` refreshes them when a profile tune is intentional.
  Auto-skips when `samples/` is absent so contributors without the
  raw data still run the other 110 tests.
- [ ] Benchmark suite: `pytest-benchmark` timings per stage at a few
  resolutions, so we can see regressions over time.
- [ ] Frontend: basic Playwright smoke test that loads the SPA and
  checks the backend health status renders.

## DevOps / CI

- [x] **GitHub Actions workflow: `pytest` + `ruff` + `mypy` on every
  PR.** — _Shipped._ `pytest`, `ruff check`, and `ruff format --check`
  hard-fail the job; `mypy app` runs but is `continue-on-error` until
  type annotations land. Workflow at `.github/workflows/ci.yml`.
- [x] **Ruff baseline cleanup** — _Shipped._ 74 findings auto-fixed
  or hand-cleaned down to zero. `continue-on-error: true` removed from
  the ruff CI steps, so new lint debt can't land.
- [x] **Pre-commit hooks** — _Shipped._ `.pre-commit-config.yaml`
  runs `ruff check --fix`, `ruff format`, trailing-whitespace,
  end-of-file-fixer, YAML/TOML validity, and a 2 MB file-size guard
  (exception documented for the Gaia parquet). Opt-in per clone
  with `pre-commit install`.
- [ ] **Git LFS for `gaia_bright.parquet`** — the full-sky catalog
  is 80 MB, above GitHub's 50 MB soft limit. The file pushes fine
  today but GitHub warns on every push. Migrating the single file
  to LFS removes the warning and keeps clone size sane if the
  catalog grows (e.g. deeper magnitude limit, multi-band extensions).
- [ ] Publish a multi-arch backend Docker image (amd64 + arm64) to
  GHCR on tagged releases.
- [ ] Dependabot for `pip` and `npm`.
- [ ] Renovate or a nightly job that rebuilds the Docker images against
  latest base images and runs the suite.

## Documentation

- [x] **"Under the hood" coverage in the README** — _Shipped in the
  README doc pass._ The Stage reference table now includes a
  "What the pixels look like after" column describing the visual
  effect of each stage. Full before/after image gallery per stage
  still TODO — would need intermediate PNGs committed and is a
  larger scope.
- [x] **Stage-authoring guide** — _Shipped in the README as the
  "Writing a new stage" section._ Covers the contract, anatomy of
  a minimal stage, pipeline wiring, profile exposure, tests, and
  the CI guardrails.
- [x] **License & weights policy page** — _Shipped in the README as
  the "Bundled model licenses & weights policy" section._ The
  four-point acceptance bar (permissive license / attribution
  traceable / SHA-256 pinned / inference-only) is now explicit,
  with a rejected-list covering StarNet++ v2, GraXpert AI denoise,
  and StarXTerminator.
- [x] **Troubleshooting** — _Shipped in the README as the
  "Troubleshooting" section._ Covers `BITPIX` / Bayer pattern /
  already-debayered FITS / SPCC-no-WCS / SPCC-low-match-count /
  golden-test failures / 429 / stacked inputs / catalog fetch
  issues.

## Nice-to-have

- [ ] Support for other smart-telescope FITS variants (Vespera, Dwarf,
  eVscope) — mostly a matter of reading their header conventions.
- [ ] Offer a grayscale pipeline path that skips demosaic and color.
- [ ] Export to TIFF and FITS in addition to PNG.
- [x] **Batch mode on the CLI** — _Shipped._ `python -m app.pipeline
  --batch samples/*.fit` writes a matching `.png` next to each input,
  or under `--output-dir` when supplied. `--jobs N` runs N pipelines
  in parallel worker processes (default 1; `--jobs 0` means one-per-
  core). BLAS thread count per worker is auto-throttled so N workers
  × all-cores-each doesn't oversubscribe. Measured ~1.45× throughput
  on 4 images with 4 workers on a 16-core box. Per-file failures are
  logged but don't abort the batch.

## Hardware acceleration

Where the wall time actually goes and what can be done about it, with
a specific note on what's Intel-iGPU-compatible (the target hardware)
vs NVIDIA-only (explicitly out of scope).

Bottleneck: `bm3d_denoise` is ~75% of per-image pipeline time (BM3D
block matching + 3D DCT). Everything else is numpy/scipy with BLAS
auto-threading, already close to CPU-peak.

- [x] **Parallel batch (CPU cores)** — _Shipped above._ `--jobs N`
  runs multiple pipelines concurrently. Best throughput win for
  batch scenarios. Intel CPUs only need `cpu_count >= 2`, which
  covers every current chip.
- [ ] **`downsample_factor=2` on BM3D as a user-selectable speed
  preset** — the stage already supports it. Measured 2.7× speedup
  on NGC 6888 but SSIM 0.80 vs full-res (visibly softer arc,
  chroma blobs in sky). Not enabled by default because it costs
  quality, but could ship as `--fast` flag that flips it per run.
- [ ] **ONNX Runtime with OpenVINO provider for v2 ML stages** —
  Intel's inference toolkit runs ONNX models on CPU/iGPU/VPU with
  a single provider string. When v2 ML star removal + denoise ship,
  the pipeline should detect OpenVINO availability and prefer the
  iGPU provider (`OpenVINOExecutionProvider`) over CPU on Intel
  hardware. Pattern is identical to CUDA detection — one provider
  string and a fallback chain. This is the main "Intel iGPU
  acceleration" path for this project.
- [ ] **OpenCV UMat (OpenCL) for classical stages** — `cv2.UMat`
  dispatches Gaussian blur / resize / CLAHE to OpenCL, which Intel
  iGPU supports. Would accelerate `sharpen`, `clahe`, and the
  BM3D downsample/upsample but not BM3D itself. Adds a ~40 MB
  OpenCV dependency for single-digit-percent total speedup given
  BM3D dominance. Borderline-worthwhile; revisit if BM3D ever gets
  a non-CPU backend.

### Out of scope (NVIDIA-only, documented for completeness)

User confirmed no NVIDIA GPU plan. The following are faster than
anything on Intel iGPU but would add a GPU dependency we can't use:

- `bm3dcuda` (pip) — 10-30× faster BM3D, NVIDIA-only. The one path
  that would actually make the dominant stage fast.
- `cupy` / `cupyx.scipy.ndimage` / `cucim` — drop-in GPU scipy /
  skimage. NVIDIA-only.
- `CUDAExecutionProvider` for ONNX Runtime — NVIDIA-only analogue
  of the OpenVINO provider above.

If the user hardware plan ever changes, these are well-understood
drop-ins, not research problems.

### Already-used, no action needed

- **Intel MKL via numpy/scipy wheels** — numpy ships with MKL or
  OpenBLAS, both of which use AVX-2/AVX-512 on Intel CPUs. Every
  dot product / FFT / convolution in scipy is already SIMD-
  accelerated. No knob to turn.
- [ ] Plate solving integration (astrometry.net) so outputs are WCS-stamped.

## Latent landmines

Code paths that work today but would race or corrupt under a future
change. Documented so the trap is visible when the trigger lands.

- [ ] **`pipeline._run_one` mutates the global profile registry**
  (`profiles.PROFILES[profile] = patched`) inside a try/finally. Safe
  in single-process CLI today. If `_run_one` ever gets called from a
  thread (e.g. the FastAPI worker pool), two concurrent overrides
  would clobber each other. Fix when triggered: thread the patched
  profile through `pipeline.run()` directly instead of monkey-
  patching the registry.
- [ ] **`pipeline._run_batch_parallel` mutates `os.environ`** to
  throttle BLAS threads, then restores in a finally. Other threads
  in the same process see the throttled values during the batch.
  Currently CLI-only so no trigger. Fix when triggered: use
  `subprocess.Popen` with the throttle in `env=` rather than
  mutating the parent process env, OR serialise concurrent batch
  invocations with a module-level lock.
