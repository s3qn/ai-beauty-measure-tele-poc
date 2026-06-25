"""Landmark -> versioned measurement schema (CLAUDE.md §5 — the versioned core).

All distances are computed in **pixel space** (normalized landmarks scaled by image
width/height so x and y share one metric) and then **normalized by interpupillary
distance (IPD)** unless noted, making them scale-invariant.

LANDMARK INDICES BELOW ARE PROVISIONAL (mediapipe 478-point canonical mesh). The
*math* is unit-tested; the *index choices* (which point counts as "nose width", the
jaw vertex, etc.) need sign-off from the ML/UX leads before this leaves POC. Do not
treat them as final — see CLAUDE.md §12.
"""

from __future__ import annotations

import numpy as np

from .schema import Measurement

# --- Provisional landmark indices (canonical mediapipe FaceLandmarker, 478 pts) ---
R_IRIS = 468        # right iris center (iris landmarks present in the 478-pt model)
L_IRIS = 473        # left iris center
FACE_RIGHT = 234    # right edge of the face oval (subject's right)
FACE_LEFT = 454     # left edge of the face oval
FOREHEAD_TOP = 10   # top-center of forehead (on the midline)
CHIN_BOTTOM = 152   # bottom-center of chin (on the midline)
NOSE_RIGHT = 129    # right alar (nostril wing)
NOSE_LEFT = 358     # left alar
MOUTH_RIGHT = 61    # right mouth corner
MOUTH_LEFT = 291    # left mouth corner
JAW_RIGHT = 172     # right jaw / near gonion
JAW_LEFT = 397      # left jaw / near gonion

# Symmetric landmark pairs (right, left) used by the symmetry score.
_SYMMETRY_PAIRS = [
    (R_IRIS, L_IRIS),
    (NOSE_RIGHT, NOSE_LEFT),
    (MOUTH_RIGHT, MOUTH_LEFT),
    (FACE_RIGHT, FACE_LEFT),
    (JAW_RIGHT, JAW_LEFT),
]

_EPS = 1e-9


def to_pixels(landmarks_norm: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Normalized (N, 2+) landmarks in [0, 1] -> (N, 2) pixel coords.

    Normalized x is relative to width and y to height, so they must be scaled by
    their own dimension before any distance/angle is meaningful.
    """
    h, w = img_shape
    pts = np.asarray(landmarks_norm, dtype=np.float64)[:, :2].copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def _dist(pts: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pts[i] - pts[j]))


def _reflect_across_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reflect point ``p`` across the line through ``a`` and ``b``."""
    d = b - a
    n = float(np.dot(d, d))
    if n < _EPS:
        return p.copy()
    v = p - a
    proj = (np.dot(v, d) / n) * d
    perp = v - proj
    return p - 2.0 * perp


def _angle_deg(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    u = p1 - vertex
    v = p2 - vertex
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < _EPS or nv < _EPS:
        return 0.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _symmetry_score(pts: np.ndarray, ipd: float) -> float:
    """Bilateral symmetry in [0, 1]. 1.0 = perfectly symmetric about the facial axis.

    The axis is the line through forehead-top and chin-bottom (both on the midline).
    Each right-side point is reflected across the axis; its distance to the matching
    left-side point is the residual. Mean residual / IPD is the asymmetry; score is
    ``1 - asymmetry`` clamped to [0, 1].
    """
    a = pts[FOREHEAD_TOP]
    b = pts[CHIN_BOTTOM]
    residuals = [
        np.linalg.norm(_reflect_across_line(pts[r], a, b) - pts[l])
        for r, l in _SYMMETRY_PAIRS
    ]
    asym = float(np.mean(residuals)) / max(ipd, _EPS)
    return float(np.clip(1.0 - asym, 0.0, 1.0))


def compute_measurements(pts: np.ndarray) -> dict[str, Measurement]:
    """Compute the versioned measurement set from (N, 2) pixel-space landmarks.

    Raises ValueError if IPD is degenerate (collapsed/invalid landmarks).
    """
    ipd = _dist(pts, R_IRIS, L_IRIS)
    if ipd < _EPS:
        raise ValueError("degenerate interpupillary distance; cannot normalize")

    face_width = _dist(pts, FACE_RIGHT, FACE_LEFT)
    face_height = _dist(pts, FOREHEAD_TOP, CHIN_BOTTOM)
    nose_width = _dist(pts, NOSE_RIGHT, NOSE_LEFT)
    mouth_width = _dist(pts, MOUTH_RIGHT, MOUTH_LEFT)
    jaw_angle = _angle_deg(pts[CHIN_BOTTOM], pts[JAW_RIGHT], pts[JAW_LEFT])

    return {
        "interpupillary_distance_norm": Measurement(
            value=1.0,
            unit="ratio_self",
            description="IPD normalized by itself; the normalization basis marker.",
        ),
        "face_width_to_height_ratio": Measurement(
            value=round(face_width / max(face_height, _EPS), 4),
            unit="ratio",
            description="Face-oval width (lm 234<->454) / height (lm 10<->152).",
        ),
        "nose_width_norm": Measurement(
            value=round(nose_width / ipd, 4),
            unit="ratio_ipd",
            description="Alar width (lm 129<->358) divided by IPD.",
        ),
        "mouth_width_norm": Measurement(
            value=round(mouth_width / ipd, 4),
            unit="ratio_ipd",
            description="Mouth-corner width (lm 61<->291) divided by IPD.",
        ),
        "face_symmetry_score": Measurement(
            value=round(_symmetry_score(pts, ipd), 4),
            unit="ratio_0to1",
            description="Bilateral symmetry about the forehead-chin axis; 1.0 = symmetric.",
        ),
        "jaw_angle_deg": Measurement(
            value=round(jaw_angle, 2),
            unit="degrees",
            description="Interior angle at the chin (lm 152) between jaw points 172 and 397.",
        ),
    }


def measure(landmarks_norm: np.ndarray, img_shape: tuple[int, int]) -> dict[str, Measurement]:
    """Convenience: scale normalized landmarks to pixels, then compute measurements."""
    return compute_measurements(to_pixels(landmarks_norm, img_shape))


__all__ = ["compute_measurements", "measure", "to_pixels"]
