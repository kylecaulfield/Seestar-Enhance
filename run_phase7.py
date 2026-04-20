"""Re-run with updated profiles, save to samples/outputs/phase7/."""
import sys, os
sys.path.insert(0, "/home/user/Seestar-Enhance/backend")

import numpy as np
from PIL import Image
from app import pipeline, profiles
from app.stages import io_fits, classify

OUT = "/home/user/Seestar-Enhance/samples/outputs/phase7"
os.makedirs(OUT, exist_ok=True)

SAMPLES = {
    "galaxy":  "/home/user/Seestar-Enhance/samples/M81_galaxy_Seestar-S50.fit",
    "cluster": "/home/user/Seestar-Enhance/samples/M92_globular-cluster_Seestar-S50.fit",
    "nebula":  "/home/user/Seestar-Enhance/samples/NGC6992_Veil-nebula_Seestar-S50.fit",
}

for label, path in SAMPLES.items():
    out_png = os.path.join(OUT, f"{label}_final.png")
    print(f"{label}: {path} -> {out_png}")
    pipeline.run(path, out_png, verbose=False)
    img = np.array(Image.open(out_png))
    pil = Image.fromarray(img); pil.thumbnail((600, 600))
    pil.save(out_png.replace(".png", ".thumb.png"))
    print(f"  done. shape={img.shape}")

print("done")
