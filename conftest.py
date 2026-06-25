"""Shared test fixtures/builders.

Per CLAUDE.md §9/§10 we use NO real face images. Measurement math is tested against
a hand-constructed, symmetric landmark array; quality checks against synthetic
generated images. This keeps tests deterministic and free of biometric data.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

# Ensure the repo root (this file's dir) is importable as `app`.
sys.path.insert(0, os.path.dirname(__file__))

from app import measurements as M  # noqa: E402
from app.config import Config  # noqa: E402

N_LANDMARKS = 478


def synthetic_landmarks() -> np.ndarray:
    """A perfectly bilaterally-symmetric 478x3 normalized landmark array.

    Only the indices the measurement code reads are meaningful; the rest sit on the
    midline. Geometry is chosen to give clean expected ratios (see test_measurements).
    """
    pts = np.full((N_LANDMARKS, 3), 0.5, dtype=np.float64)
    pts[:, 2] = 0.0
    coords = {
        M.FOREHEAD_TOP: (0.50, 0.10),
        M.CHIN_BOTTOM: (0.50, 0.90),
        M.R_IRIS: (0.42, 0.40),
        M.L_IRIS: (0.58, 0.40),
        M.NOSE_RIGHT: (0.45, 0.55),
        M.NOSE_LEFT: (0.55, 0.55),
        M.MOUTH_RIGHT: (0.40, 0.70),
        M.MOUTH_LEFT: (0.60, 0.70),
        M.FACE_RIGHT: (0.25, 0.50),
        M.FACE_LEFT: (0.75, 0.50),
        M.JAW_RIGHT: (0.35, 0.80),
        M.JAW_LEFT: (0.65, 0.80),
    }
    for idx, (x, y) in coords.items():
        pts[idx, 0] = x
        pts[idx, 1] = y
    return pts


def checkerboard(size: int = 300, cell: int = 10) -> np.ndarray:
    """A sharp, mid-brightness RGB image that passes the quality gate."""
    ys, xs = np.indices((size, size))
    board = (((xs // cell) + (ys // cell)) % 2) * 255
    return np.stack([board] * 3, axis=-1).astype(np.uint8)


def png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A token-less Config writing telemetry/temp under a tmp dir."""
    return Config(
        telegram_bot_token="test-token",
        telemetry_path=str(tmp_path / "telemetry.jsonl"),
        temp_dir=str(tmp_path / "temp"),
    )
