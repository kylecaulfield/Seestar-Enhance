"""Golden-image regression tests.

Runs the full pipeline on each sample FITS in `samples/` and compares
the output to a committed golden PNG using SSIM. A failure means the
pipeline output drifted measurably — either an intended profile tune
(refresh the golden with `REGEN_GOLDEN=1 pytest -k test_golden`) or a
silent regression (investigate before regenerating).

The test auto-skips when the samples directory isn't available (e.g.
on a shallow checkout without LFS, or in a contributor's sandbox
without the raw data). The 87 non-golden tests still run.

Golden PNGs are intentionally downscaled to 400×400. Small enough
to bundle (~150 KB each, ~1 MB total for six samples) and still
sensitive to hue, brightness, and structural drift. Full-size
regressions too subtle to show at that resolution are below the
noise floor of the pipeline anyway (BM3D has its own stochasticity
even with seeds).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from app import pipeline
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"
GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_SIZE = (400, 400)
# SSIM floor. Pipeline output is deterministic for a given input + code
# state, so a real regression usually drops SSIM well below 0.95; tiny
# platform differences (libjpeg/libpng/libc versions) can still push a
# clean run a few thousandths below 1.0.
SSIM_THRESHOLD = 0.95


# (filename under samples/, short tag used for golden PNG)
SAMPLES = [
    ("NGC 6888.fit", "ngc6888"),
    ("NGC 2244.fit", "ngc2244"),
    ("6960.fit", "ngc6960"),
    ("M81_galaxy_Seestar-S50.fit", "m81"),
    ("M92_globular-cluster_Seestar-S50.fit", "m92"),
    ("NGC6992_Veil-nebula_Seestar-S50.fit", "ngc6992"),
]


pytestmark = pytest.mark.skipif(
    not SAMPLES_DIR.is_dir(),
    reason=(
        f"samples directory not found at {SAMPLES_DIR}. Golden-image "
        "tests require the bundled sample FITS files."
    ),
)


def _thumb(png_path: Path) -> np.ndarray:
    """Load a PNG and downscale to GOLDEN_SIZE for comparison."""
    img = Image.open(png_path).convert("RGB")
    img = img.resize(GOLDEN_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on RGB arrays (channel_axis=-1)."""
    from skimage.metrics import structural_similarity as ssim

    return float(ssim(a, b, channel_axis=-1, data_range=255))


def _run_and_thumb(sample_path: Path, tmp_path: Path) -> np.ndarray:
    out = tmp_path / "out.png"
    pipeline.run(str(sample_path), str(out), verbose=False)
    return _thumb(out)


@pytest.mark.parametrize("fits_name,tag", SAMPLES, ids=[t for _, t in SAMPLES])
def test_golden_matches(
    fits_name: str,
    tag: str,
    tmp_path: Path,
) -> None:
    fits_path = SAMPLES_DIR / fits_name
    if not fits_path.exists():
        pytest.skip(f"sample {fits_path} not present")

    actual = _run_and_thumb(fits_path, tmp_path)

    golden_path = GOLDEN_DIR / f"{tag}.png"

    # REGEN_GOLDEN=1 pytest -k test_golden regenerates the committed
    # golden PNGs. Use this when a profile tune is intentional.
    if os.environ.get("REGEN_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        Image.fromarray(actual).save(golden_path, optimize=True)
        pytest.skip(f"regenerated {golden_path}")

    if not golden_path.exists():
        pytest.skip(f"golden {golden_path} missing. Set REGEN_GOLDEN=1 to create it.")

    expected = np.asarray(Image.open(golden_path).convert("RGB"), dtype=np.uint8)

    # Shape guard — if the pipeline somehow changes output aspect ratio,
    # SSIM doesn't apply.
    assert actual.shape == expected.shape, f"{tag}: shape {actual.shape} vs golden {expected.shape}"

    score = _ssim(actual, expected)
    assert score >= SSIM_THRESHOLD, (
        f"{tag}: SSIM {score:.4f} below {SSIM_THRESHOLD}. "
        "Regression or intended tune? Rerun with REGEN_GOLDEN=1 if intended."
    )
