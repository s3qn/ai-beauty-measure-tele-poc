"""Image quality gate (CLAUDE.md §4 step 5, §7).

Cheap, pure-numpy checks run *before* detection so obviously bad input is rejected
without paying for landmark inference. Face-size/centering needs a detected box, so
it's a separate post-detection check (``check_face_geometry``).

No OpenCV: brightness is mean luma, sharpness is the variance of a discrete
Laplacian computed with numpy slicing (equivalent to "variance of Laplacian").
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config
from .schema import ErrorCode

# Rec. 601 luma weights.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> float luma (H, W)."""
    return img_rgb[..., :3].astype(np.float64) @ _LUMA


def brightness(img_rgb: np.ndarray) -> float:
    """Mean luma on a 0-255 scale. Higher is brighter."""
    return float(to_gray(img_rgb).mean())


def blur_score(img_rgb: np.ndarray) -> float:
    """Variance of the discrete Laplacian — higher means sharper.

    Uses the 4-neighbour Laplacian (4*c - up - down - left - right) on interior
    pixels. A flat or out-of-focus image yields low variance.
    """
    g = to_gray(img_rgb)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        4.0 * g[1:-1, 1:-1]
        - g[:-2, 1:-1]
        - g[2:, 1:-1]
        - g[1:-1, :-2]
        - g[1:-1, 2:]
    )
    return float(lap.var())


def check_prequality(img_rgb: np.ndarray, cfg: Config) -> Optional[ErrorCode]:
    """Cheap pre-detection checks. Returns the first failing code, or None if OK.

    Order is cheapest/most-decisive first: resolution, brightness, blur.
    """
    h, w = img_rgb.shape[:2]
    if min(h, w) < cfg.min_resolution:
        return ErrorCode.FACE_TOO_SMALL
    if brightness(img_rgb) < cfg.brightness_min:
        return ErrorCode.TOO_DARK
    if blur_score(img_rgb) < cfg.blur_threshold:
        return ErrorCode.TOO_BLURRY
    return None


def check_face_geometry(
    face_box: tuple[float, float, float, float],
    img_shape: tuple[int, int],
    cfg: Config,
) -> Optional[ErrorCode]:
    """Post-detection size/centering check.

    ``face_box`` is (x_min, y_min, x_max, y_max) in pixel coords; ``img_shape`` is
    (height, width). Returns FACE_TOO_SMALL if the face is too small *or* badly
    off-center (both map to the "move closer / center" guidance, §6/§7).
    """
    h, w = img_shape
    x0, y0, x1, y1 = face_box
    face_w = max(0.0, x1 - x0)
    if w == 0 or (face_w / w) < cfg.min_face_ratio:
        return ErrorCode.FACE_TOO_SMALL

    # Centering: face center must sit within the central 60% of each axis.
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if not (0.20 * w <= cx <= 0.80 * w) or not (0.20 * h <= cy <= 0.80 * h):
        return ErrorCode.FACE_TOO_SMALL
    return None


__all__ = [
    "to_gray",
    "brightness",
    "blur_score",
    "check_prequality",
    "check_face_geometry",
]
