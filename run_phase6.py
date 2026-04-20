"""Run 5 parameter variations per sample, save to samples/outputs/phase6/."""
import sys
import os
sys.path.insert(0, "/home/user/Seestar-Enhance/backend")

import numpy as np
from PIL import Image
from app import pipeline, profiles
from app.stages import io_fits, classify, background, color, stretch as stretch_stage, bm3d_denoise, sharpen, curves

OUT = "/home/user/Seestar-Enhance/samples/outputs/phase6"
SAMPLES = {
    "galaxy":  "/home/user/Seestar-Enhance/samples/M81_galaxy_Seestar-S50.fit",
    "cluster": "/home/user/Seestar-Enhance/samples/M92_globular-cluster_Seestar-S50.fit",
    "nebula":  "/home/user/Seestar-Enhance/samples/NGC6992_Veil-nebula_Seestar-S50.fit",
}

# Variations: list of (label, override_dict)
# Each override_dict maps stage_name -> dict of kwargs to MERGE into the profile
VARIATIONS = {
    "galaxy": [
        ("v1_baseline",   {}),
        ("v2_more_stretch",  {"stretch": {"stretch": 24.0, "black_percentile": 0.1}}),
        ("v3_more_sharpen",  {"sharpen": {"radius": 1.5, "amount": 0.45}}),
        ("v4_more_sat",      {"curves": {"contrast": 0.55, "saturation": 1.35}}),
        ("v5_less_denoise",  {"bm3d_denoise": {"sigma": None, "strength": 1.2, "chroma_blur": 2.0}}),
    ],
    "cluster": [
        ("v1_baseline",   {}),
        ("v2_less_contrast",  {"curves": {"contrast": 0.15, "saturation": 1.08}}),
        ("v3_more_sharpen",   {"sharpen": {"radius": 1.0, "amount": 0.22}}),
        ("v4_less_stretch",   {"stretch": {"black_percentile": 0.2, "stretch": 16.0}}),
        ("v5_more_denoise",   {"bm3d_denoise": {"sigma": None, "strength": 1.5, "chroma_blur": 1.5}}),
    ],
    "nebula": [
        ("v1_baseline",   {}),
        ("v2_lower_scnr",     {"color": {"dark_percentile": 25.0, "mid_low": 40.0, "mid_high": 80.0,
                                          "wb_strength": 1.0, "green_clip": 0.7}}),
        ("v3_more_sat",       {"curves": {"contrast": 0.6, "saturation": 1.25}}),
        ("v4_less_chroma",    {"bm3d_denoise": {"sigma": 0.15, "strength": 1.0, "chroma_blur": 5.0}}),
        ("v5_more_contrast",  {"curves": {"contrast": 0.70, "saturation": 1.15}}),
    ],
}


def run_with_params(img_raw, base_profile, overrides, out_path):
    """Run the full v1 pipeline with merged overrides."""
    params = {k: dict(v) for k, v in base_profile.items()}
    for stage, kw in overrides.items():
        if stage in params:
            params[stage].update(kw)
        else:
            params[stage] = kw

    img = background.process(img_raw, **params["background"])
    img = color.process(img, **params["color"])
    img = stretch_stage.process(img, **params["stretch"])
    img = bm3d_denoise.process(img, **params["bm3d_denoise"])
    img = sharpen.process(img, **params["sharpen"])
    img = curves.process(img, **params["curves"])

    out8 = (np.clip(img, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(out8).save(out_path, optimize=True)
    # thumbnail
    pil = Image.fromarray(out8)
    pil.thumbnail((600, 600))
    thumb_path = out_path.replace(".png", ".thumb.png")
    pil.save(thumb_path)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)

    for target, fits_path in SAMPLES.items():
        print(f"\n=== {target.upper()} ===")
        img_raw = io_fits.load_fits(fits_path)
        print(f"  Raw shape: {img_raw.shape}, range [{img_raw.min():.5f}, {img_raw.max():.5f}]")

        # classify to confirm
        cls = classify.classify(img_raw)
        print(f"  Auto-classify: {cls}")

        base = profiles.get(cls)
        for label, overrides in VARIATIONS[target]:
            out_path = os.path.join(OUT, f"{target}_{label}.png")
            print(f"  Running {label} ...", end=" ", flush=True)
            run_with_params(img_raw, base, overrides, out_path)
            print(f"-> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
