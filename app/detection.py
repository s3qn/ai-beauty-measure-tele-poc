"""Face + landmark detection with single-face enforcement (CLAUDE.md §4 step 6).

Backend: mediapipe Tasks ``FaceLandmarker`` (478 landmarks incl. iris). The legacy
``mediapipe.solutions.face_mesh`` API is not shipped in 0.10.35, so we use the Tasks
API. The model bundle is downloaded + cached on first use.

The single-face *count* logic (``enforce_single_face``) is pure and unit-tested
separately from the heavy model so we don't need a biometric fixture to test it.
"""

from __future__ import annotations

import atexit
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .schema import ErrorCode

# Number of faces the detector is allowed to find. >1 so we can *detect* and reject
# the multi-face case rather than silently taking the first.
_MAX_FACES = 5


@dataclass
class DetectionResult:
    """One detected face. ``landmarks`` is (478, 3) normalized (x, y, z) in [0, 1]."""

    landmarks: np.ndarray
    face_box: tuple[float, float, float, float]  # pixel coords (x0, y0, x1, y1)


def enforce_single_face(num_faces: int) -> Optional[ErrorCode]:
    """Map a detected-face count to an error code, or None for exactly one.

    Pure function — the heart of US1.2's single-face rule, testable without a model.
    """
    if num_faces == 0:
        return ErrorCode.NO_FACE
    if num_faces > 1:
        return ErrorCode.MULTIPLE_FACES
    return None


def ensure_model(cfg: Config) -> str:
    """Return a path to the FaceLandmarker model, downloading + caching if absent."""
    path = cfg.model_path
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Download atomically so a partial file is never mistaken for a valid model.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(cfg.model_url, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _landmarks_to_box(
    landmarks: np.ndarray, img_shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    h, w = img_shape
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class FaceDetector:
    """Wraps a mediapipe FaceLandmarker. Created lazily and reused (the model load
    is expensive). Not thread-safe — fine for the single-threaded polling POC."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._landmarker = None  # built on first detect()

    def _ensure_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        # Imported lazily so importing this module (e.g. for enforce_single_face in
        # tests) doesn't require the native mediapipe runtime.
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        model_path = ensure_model(self._cfg)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=_MAX_FACES,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        # We close() deterministically at exit (below). mediapipe's own __del__ then
        # closes a *second* time at interpreter teardown, after module globals are
        # gone, raising a harmless TypeError to stderr. Neutralize that finalizer so
        # only our explicit close() runs. (__del__ is resolved on the type.)
        try:
            type(landmarker).__del__ = lambda self: None
        except Exception:
            pass
        self._landmarker = landmarker
        atexit.register(self.close)
        return self._landmarker

    def close(self) -> None:
        """Release the native landmarker. Safe to call more than once."""
        landmarker = self._landmarker
        self._landmarker = None
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

    def detect(self, img_rgb: np.ndarray) -> list[DetectionResult]:
        """Return all detected faces (up to ``_MAX_FACES``). Empty if none."""
        import mediapipe as mp

        landmarker = self._ensure_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
        result = landmarker.detect(mp_image)

        h, w = img_rgb.shape[:2]
        faces: list[DetectionResult] = []
        for face in result.face_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float64)
            faces.append(DetectionResult(landmarks=pts, face_box=_landmarks_to_box(pts, (h, w))))
        return faces


__all__ = ["DetectionResult", "FaceDetector", "enforce_single_face", "ensure_model"]
