# Backlog

Planned work beyond v1. Items are grouped by area, not by priority within
a group. Items marked **[v2]** are the headline v2 features; the rest are
quality-of-life, infra, and polish.

Legend: `[ ]` open, `[x]` done, `[~]` in progress or partially landed.

---

## ML stages (v2)

- [ ] **[v2] Self-trained star removal** — small U-Net / ResUNet, trained
  on public star-field data, exported to ONNX. Bundle weights in
  `backend/app/models/` under an MIT-compatible license.
  - Fills `backend/app/stages/stars.py::process`.
  - Stable return signature: `(stars_only, starless)`, both float32 RGB in
    `[0, 1]`.
  - Tiled inference with 256 or 512 px tiles and 32 px overlap; cosine
    window for seam blending.
  - Slot into `pipeline.run()` after `stretch`; screen-blend stars back
    in after `sharpen` (which will run on starless only).
- [ ] **[v2] ML denoise on starless layer** — self-trained or
  permissively-licensed third-party weights.
  - Fills `backend/app/stages/ml_denoise.py::process`.
  - Runs only on the `starless` layer so star PSFs aren't softened.
  - Keep `bm3d_denoise` wired as a user-selectable fallback.
- [ ] **[v2] Shared tiled-inference helper** — once both ML stages exist,
  factor out `backend/app/stages/_tiled.py` with:
  - `iter_tiles(shape, tile, overlap) -> Iterable[slice pairs]`
  - `cosine_window(tile) -> np.ndarray`
  - `blend(output_accum, weight_accum, tile_out, tile_window, slices)`
  - One unit test proving full-image reconstruction on random input.
- [ ] Model registry: a small JSON/TOML manifest under
  `backend/app/models/` listing each bundled model's name, SHA-256,
  input/output tensor names, tile size, and license. Pipeline checks the
  manifest at startup.
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

## Profiles & UX

- [ ] Profile for wide-field Milky Way shots (Seestar mosaic mode).
- [ ] `auto` profile that picks one of `nebula / galaxy / cluster` from
  a cheap classifier over the image histogram.
- [ ] Let profiles override only a subset of a parent profile (current
  implementation uses `{**DEFAULT, "stretch": {...}}` which replaces
  the whole sub-dict — fine for now, awkward as profiles grow).
- [ ] Expose `--override stage.param=value` flags on the CLI for
  one-off experimentation without editing `profiles.py`.

## Backend API

- [ ] `POST /jobs` — upload a FITS file (multipart) and a profile name,
  return a job ID.
- [ ] `GET /jobs/{id}` — poll job state (`queued`, `running`, `done`,
  `error`) with per-stage progress.
- [ ] `GET /jobs/{id}/result` — download the PNG (and maybe an
  intermediate preview-strip sheet showing each stage's output).
- [ ] Background worker: either a thread pool in-process or a separate
  container running `rq`/`arq` with Redis. Probably in-process is fine
  for v1.5.
- [ ] Persistent job store (SQLite) so the server can survive restarts.
- [ ] Rate-limit and size-cap uploads; FITS files can be hundreds of MB.

## Frontend

- [ ] Replace placeholder `App.tsx` with a minimal upload-preview-tune
  flow: file picker, profile dropdown, "process" button, result image.
- [ ] Stage-by-stage preview strip: show the image at each intermediate
  stage so users can see what each step is doing.
- [ ] Parameter tweak panel that mirrors `profiles.py` (dial in custom
  params without editing Python).
- [ ] Before/after slider on the result image.
- [ ] Persist uploaded files in the browser's OPFS or IndexedDB so a
  refresh doesn't re-upload.
- [ ] Basic theming / CSS modules instead of the current inline
  `index.css`.

## Testing

- [ ] Property-based tests with `hypothesis`: generate random float32
  RGB images and assert the `[0, 1]` invariant is preserved by every
  stage.
- [ ] Golden-image tests: store one reference PNG per profile per sample
  and diff with a perceptual metric (e.g., SSIM ≥ 0.95). Guards against
  silent regressions in numeric behavior.
- [ ] Pipeline integration test that runs on a real Seestar FITS
  (committed as test data or fetched via `pytest --runslow`).
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
