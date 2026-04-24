"""Rectangular crop.

Useful as a lightweight first stage when iterating on profiles against
a specific region of interest (a galaxy, a nebula knot) — chopping 70%
of the pixels off up front makes every downstream stage much faster.
Also lets us trim Seestar field-rotation artifacts at frame edges when
they're present.

Convention: the crop is specified as a `(top, left, bottom, right)`
box in pixel coordinates, matching numpy/PIL. Bottom / right may be
negative, meaning "N pixels from the far edge".
"""

from __future__ import annotations

import numpy as np


def process(
    image: np.ndarray,
    top: int = 0,
    left: int = 0,
    bottom: int | None = None,
    right: int | None = None,
) -> np.ndarray:
    """Return a rectangular sub-image.

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB image of shape (H, W, 3).
    top, left : int
        Inclusive upper-left corner (pixels).
    bottom, right : int | None
        Exclusive lower-right corner. ``None`` means "to the edge";
        negative values are measured from the far edge.

    Returns
    -------
    np.ndarray
        Cropped view (contiguous copy) with the same dtype.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {image.shape}")

    h, w = image.shape[:2]
    b = h if bottom is None else (bottom if bottom >= 0 else h + bottom)
    r = w if right is None else (right if right >= 0 else w + right)

    if not (0 <= top < b <= h) or not (0 <= left < r <= w):
        raise ValueError(
            f"invalid crop box: top={top}, left={left}, bottom={b}, right={r} "
            f"on image of shape {image.shape}"
        )

    return np.ascontiguousarray(image[top:b, left:r, :])
