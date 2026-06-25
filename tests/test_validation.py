"""Pipeline error routing + temp-file cleanup (CLAUDE.md §4/§6/§9/§10).

Uses a FakeDetector so no model or biometric image is needed; detection *counts* are
controlled directly to exercise NO_FACE / MULTIPLE_FACES / success.
"""

import os

import pytest

from app.detection import DetectionResult
from app.pipeline import Pipeline
from app.schema import ErrorCode, MeasurementResult, PipelineError
from conftest import checkerboard, png_bytes, synthetic_landmarks

GOOD_BOX = (75.0, 30.0, 225.0, 270.0)  # passes geometry on a 300x300 image


class FakeDetector:
    """Returns a preset list of faces; raises if a test expected the gate to skip it."""

    def __init__(self, faces):
        self.faces = faces
        self.called = False

    def detect(self, img_rgb):
        self.called = True
        return self.faces


def face(box=GOOD_BOX):
    return DetectionResult(landmarks=synthetic_landmarks(), face_box=box)


def make_pipeline(cfg, faces):
    return Pipeline(cfg, detector=FakeDetector(faces))


def good_png():
    return png_bytes(checkerboard(300))


# --- input/decoding ---------------------------------------------------------

def test_empty_bytes_not_an_image(cfg):
    out = make_pipeline(cfg, []).process(b"")
    assert isinstance(out, PipelineError) and out.code is ErrorCode.NOT_AN_IMAGE


def test_garbage_decode_failed(cfg):
    out = make_pipeline(cfg, []).process(b"this is not an image")
    assert isinstance(out, PipelineError) and out.code is ErrorCode.DECODE_FAILED


# --- quality gate short-circuits before detection ---------------------------

def test_dark_image_rejected_before_detection(cfg):
    import numpy as np
    dark = png_bytes(np.full((300, 300, 3), 5, dtype=np.uint8))
    det = FakeDetector([face()])
    out = Pipeline(cfg, detector=det).process(dark)
    assert isinstance(out, PipelineError) and out.code is ErrorCode.TOO_DARK
    assert det.called is False  # detection must NOT run on bad input


# --- detection / single-face enforcement ------------------------------------

def test_no_face(cfg):
    out = make_pipeline(cfg, []).process(good_png())
    assert isinstance(out, PipelineError) and out.code is ErrorCode.NO_FACE


def test_multiple_faces(cfg):
    out = make_pipeline(cfg, [face(), face()]).process(good_png())
    assert isinstance(out, PipelineError) and out.code is ErrorCode.MULTIPLE_FACES


def test_face_too_small(cfg):
    out = make_pipeline(cfg, [face((135, 135, 165, 165))]).process(good_png())
    assert isinstance(out, PipelineError) and out.code is ErrorCode.FACE_TOO_SMALL


def test_success(cfg):
    out = make_pipeline(cfg, [face()]).process(good_png())
    assert isinstance(out, MeasurementResult)
    assert out.schema_version == "1.0.0"
    assert out.correlation_id
    assert out.processing_ms is not None and out.processing_ms >= 0
    assert "face_symmetry_score" in out.measurements


# --- correlation id + telemetry ---------------------------------------------

def test_every_outcome_has_correlation_id_and_telemetry(cfg):
    import json
    pipe = make_pipeline(cfg, [face()])
    ok = pipe.process(good_png())
    bad = make_pipeline(cfg, []).process(good_png())
    assert ok.correlation_id and bad.correlation_id

    lines = [json.loads(l) for l in open(cfg.telemetry_path)]
    assert len(lines) == 2
    for rec in lines:
        assert rec["correlation_id"]
        assert rec["processing_ms"] >= 0
        assert rec["schema_version"] == "1.0.0"
    assert {r["outcome"] for r in lines} == {"success", "rejected"}


def test_telemetry_has_no_pii(cfg):
    import json
    make_pipeline(cfg, [face()]).process(good_png(), chat_id=123456)
    rec = json.loads(open(cfg.telemetry_path).readline())
    # No salt configured => no chat id hash, and never a raw chat id.
    assert "chat_id" not in rec
    assert "chat_id_hash" not in rec


# --- temp-file cleanup (US1.4) ----------------------------------------------

@pytest.mark.parametrize("faces", [[face()], []], ids=["success", "failure"])
def test_no_temp_file_remains(cfg, faces):
    os.makedirs(cfg.temp_dir, exist_ok=True)
    stray = os.path.join(cfg.temp_dir, "leftover.jpg")
    with open(stray, "wb") as fh:
        fh.write(b"x")
    make_pipeline(cfg, faces).process(good_png())
    assert os.listdir(cfg.temp_dir) == []  # swept clean on success and failure
