"""Iteration runner. Re-runs pipeline on all three samples and saves tagged output."""
import sys, os, pathlib
sys.path.insert(0, "/home/user/Seestar-Enhance/backend")

from PIL import Image
from app import pipeline

SAMPLES = {
    "galaxy":  "/home/user/Seestar-Enhance/samples/M81_galaxy_Seestar-S50.fit",
    "cluster": "/home/user/Seestar-Enhance/samples/M92_globular-cluster_Seestar-S50.fit",
    "nebula":  "/home/user/Seestar-Enhance/samples/NGC6992_Veil-nebula_Seestar-S50.fit",
}

def run_iter(tag: str) -> None:
    out = pathlib.Path(f"/home/user/Seestar-Enhance/samples/outputs/iter_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    for label, path in SAMPLES.items():
        target = out / f"{label}.png"
        print(f"  {label}...", end=" ", flush=True)
        pipeline.run(path, str(target), verbose=False)
        img = Image.open(target)
        img.thumbnail((600, 600))
        img.save(out / f"{label}.thumb.png")
        print("done")
    print(f"  -> {out}")

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"=== iter {tag} ===")
    run_iter(tag)
