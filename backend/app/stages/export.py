"""Export to 16-bit PNG.

Uses pypng for the PNG writer because Pillow's high-level API does not
support 16-bit-per-channel RGB output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import png

PathLike = Union[str, Path]


def process(image: np.ndarray, path: PathLike) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    as_u16 = np.round(clipped * 65535.0).astype(np.uint16)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w, _ = as_u16.shape
    rows = as_u16.reshape(h, w * 3)
    writer = png.Writer(width=w, height=h, bitdepth=16, greyscale=False)
    with out.open("wb") as f:
        writer.write(f, rows.tolist())

    return clipped
