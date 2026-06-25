"""Measurement math against a known synthetic landmark fixture (CLAUDE.md §5/§10)."""

import math

import numpy as np
import pytest

from app import measurements as M
from app.schema import SCHEMA_VERSION
from conftest import synthetic_landmarks

IMG = (1000, 1000)  # square so x/y scale equally


@pytest.fixture
def result():
    return M.measure(synthetic_landmarks(), IMG)


def test_all_expected_fields_present(result):
    assert set(result) == {
        "interpupillary_distance_norm",
        "face_width_to_height_ratio",
        "nose_width_norm",
        "mouth_width_norm",
        "face_symmetry_score",
        "jaw_angle_deg",
    }


def test_units_are_documented(result):
    units = {k: m.unit for k, m in result.items()}
    assert units == {
        "interpupillary_distance_norm": "ratio_self",
        "face_width_to_height_ratio": "ratio",
        "nose_width_norm": "ratio_ipd",
        "mouth_width_norm": "ratio_ipd",
        "face_symmetry_score": "ratio_0to1",
        "jaw_angle_deg": "degrees",
    }
    assert all(m.description for m in result.values())


def test_known_values(result):
    # IPD = |0.58-0.42| * 1000 = 160 px.
    assert result["interpupillary_distance_norm"].value == 1.0
    # face width 500 / height 800.
    assert result["face_width_to_height_ratio"].value == pytest.approx(0.625, abs=1e-3)
    # nose 100 / 160 ; mouth 200 / 160.
    assert result["nose_width_norm"].value == pytest.approx(0.625, abs=1e-3)
    assert result["mouth_width_norm"].value == pytest.approx(1.25, abs=1e-3)
    # perfectly symmetric layout.
    assert result["face_symmetry_score"].value == pytest.approx(1.0, abs=1e-6)
    # angle at chin between the two jaw points.
    assert result["jaw_angle_deg"].value == pytest.approx(112.62, abs=0.1)


def test_scale_invariance():
    """Ratios must be identical regardless of image resolution."""
    lm = synthetic_landmarks()
    a = M.measure(lm, (1000, 1000))
    b = M.measure(lm, (480, 480))
    for k in a:
        assert a[k].value == pytest.approx(b[k].value, abs=1e-6)


def test_asymmetry_lowers_score():
    lm = synthetic_landmarks()
    lm[M.MOUTH_LEFT, 0] += 0.1  # push one mouth corner off the mirror
    score = M.measure(lm, IMG)["face_symmetry_score"].value
    assert 0.0 <= score < 1.0


def test_degenerate_ipd_raises():
    lm = synthetic_landmarks()
    lm[M.L_IRIS] = lm[M.R_IRIS]  # collapse IPD
    with pytest.raises(ValueError):
        M.measure(lm, IMG)


def test_schema_version_marker():
    assert SCHEMA_VERSION == "1.0.0"
