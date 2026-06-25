"""Quality-gate thresholds map to the right error codes (CLAUDE.md §7/§10)."""

import numpy as np

from app import quality
from app.schema import ErrorCode
from conftest import checkerboard


def test_good_image_passes(cfg):
    assert quality.check_prequality(checkerboard(300), cfg) is None


def test_dark_image_too_dark(cfg):
    dark = np.full((300, 300, 3), 5, dtype=np.uint8)
    assert quality.check_prequality(dark, cfg) is ErrorCode.TOO_DARK


def test_flat_image_too_blurry(cfg):
    # Bright but no high-frequency detail => zero Laplacian variance.
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    assert quality.check_prequality(flat, cfg) is ErrorCode.TOO_BLURRY


def test_tiny_image_too_small(cfg):
    tiny = checkerboard(100)
    assert quality.check_prequality(tiny, cfg) is ErrorCode.FACE_TOO_SMALL


def test_brightness_and_blur_metrics_directional():
    assert quality.brightness(np.full((10, 10, 3), 200, np.uint8)) > quality.brightness(
        np.full((10, 10, 3), 20, np.uint8)
    )
    assert quality.blur_score(checkerboard(100)) > quality.blur_score(
        np.full((100, 100, 3), 128, np.uint8)
    )


def test_face_geometry_centered_ok(cfg):
    # Big, centered face box in a 300x300 image.
    assert quality.check_face_geometry((75, 30, 225, 270), (300, 300), cfg) is None


def test_face_geometry_too_small(cfg):
    # 30px wide / 300 => 0.1 ratio, below MIN_FACE_RATIO.
    assert quality.check_face_geometry((135, 135, 165, 165), (300, 300), cfg) is ErrorCode.FACE_TOO_SMALL


def test_face_geometry_off_center(cfg):
    # Wide enough but pushed into the corner.
    assert quality.check_face_geometry((0, 0, 100, 100), (300, 300), cfg) is ErrorCode.FACE_TOO_SMALL
