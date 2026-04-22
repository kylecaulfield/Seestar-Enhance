"""Run our pipeline on the 3 new samples (forced to nebula profile)."""
import sys, pathlib
sys.path.insert(0, "/home/user/Seestar-Enhance/backend")

from PIL import Image
from app import pipeline

SAMPLES = {
    "ngc6888": "/home/user/Seestar-Enhance/samples/NGC 6888.fit",
    "ngc2244": "/home/user/Seestar-Enhance/samples/NGC 2244.fit",
    "ngc6960": "/home/user/Seestar-Enhance/samples/6960.fit",
}

def run_iter(tag: str) -> None:
    out = pathlib.Path(f"/home/user/Seestar-Enhance/samples/outputs/new_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    for label, path in SAMPLES.items():
        target = out / f"{label}.png"
        print(f"  {label}...", end=" ", flush=True)
        pipeline.run(path, str(target), profile="nebula", verbose=False)
        img = Image.open(target)
        img.thumbnail((700, 700))
        img.save(out / f"{label}.thumb.png")
        print("done")
    print(f"  -> {out}")

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"=== {tag} ===")
    run_iter(tag)
