"""Request orchestration: validate -> decode -> quality -> detect -> measure -> respond.

Implements the flow in CLAUDE.md §4. Processes images **in memory** (no raw image
persisted). Every path produces a correlation_id and exactly one telemetry line, and
cleans up temp artifacts in ``finally``.
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from typing import Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import quality, telemetry
from .config import Config
from .detection import FaceDetector, enforce_single_face
from .measurements import compute_measurements, to_pixels
from .schema import ErrorCode, MeasurementResult, PipelineError

logger = logging.getLogger("beauty_bot.pipeline")

Outcome = Union[MeasurementResult, PipelineError]


def sweep_temp_dir(cfg: Config) -> None:
    """Clear TEMP_DIR on boot as a backstop against leftover temp images (§9)."""
    d = cfg.temp_dir
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        return
    for name in os.listdir(d):
        path = os.path.join(d, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("temp sweep could not remove %s: %s", path, exc)


class Pipeline:
    """Owns config + detector and runs one request end to end.

    The detector is injectable so tests can supply a fake (no model needed) or skip
    detection entirely.
    """

    def __init__(self, cfg: Config, detector: Optional[FaceDetector] = None):
        self.cfg = cfg
        self.detector = detector if detector is not None else FaceDetector(cfg)

    def process(self, image_bytes: bytes, chat_id: Optional[int] = None) -> Outcome:
        correlation_id = uuid.uuid4().hex
        started = time.perf_counter()
        outcome_kind = "error"
        error_code: Optional[ErrorCode] = None
        result: Outcome

        try:
            result = self._run(image_bytes, correlation_id)
            if isinstance(result, PipelineError):
                error_code = result.code
                outcome_kind = "error" if result.code == ErrorCode.INTERNAL_ERROR else "rejected"
            else:
                outcome_kind = "success"
            return result
        except Exception as exc:  # never leak internals to the user (§6)
            logger.exception("unhandled pipeline error (%s)", correlation_id)
            error_code = ErrorCode.INTERNAL_ERROR
            result = PipelineError(correlation_id, ErrorCode.INTERNAL_ERROR, detail=str(exc))
            return result
        finally:
            processing_ms = int((time.perf_counter() - started) * 1000)
            # Stamp processing time onto whatever we return.
            try:
                result.processing_ms = processing_ms  # type: ignore[has-type]
            except Exception:
                pass
            self._cleanup_temp()
            telemetry.log_outcome(
                correlation_id=correlation_id,
                outcome=outcome_kind,
                processing_ms=processing_ms,
                error_code=error_code,
                chat_id_hash=telemetry.hash_chat_id(chat_id, self.cfg.chat_id_salt)
                if chat_id is not None
                else None,
                telemetry_path=self.cfg.telemetry_path,
            )
            if processing_ms > self.cfg.target_latency_ms:
                logger.warning(
                    "processing_ms=%d exceeded target %d (%s)",
                    processing_ms,
                    self.cfg.target_latency_ms,
                    correlation_id,
                )

    # --- internal steps -------------------------------------------------------

    def _run(self, image_bytes: bytes, correlation_id: str) -> Outcome:
        if not image_bytes:
            return PipelineError(correlation_id, ErrorCode.NOT_AN_IMAGE)

        img = self._decode(image_bytes)
        if img is None:
            return PipelineError(correlation_id, ErrorCode.DECODE_FAILED)

        # Cheap quality gate before paying for detection (§4 step 5).
        bad = quality.check_prequality(img, self.cfg)
        if bad is not None:
            return PipelineError(correlation_id, bad)

        faces = self.detector.detect(img)
        single = enforce_single_face(len(faces))
        if single is not None:
            return PipelineError(correlation_id, single)

        face = faces[0]
        geo = quality.check_face_geometry(face.face_box, img.shape[:2], self.cfg)
        if geo is not None:
            return PipelineError(correlation_id, geo)

        pts = to_pixels(face.landmarks, img.shape[:2])
        measurements = compute_measurements(pts)
        return MeasurementResult(correlation_id=correlation_id, measurements=measurements)

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode bytes to an RGB uint8 array in memory. None on failure."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _cleanup_temp(self) -> None:
        """Defensive: remove any stray files in TEMP_DIR on this request path."""
        d = self.cfg.temp_dir
        if not os.path.isdir(d):
            return
        for name in os.listdir(d):
            path = os.path.join(d, name)
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass


__all__ = ["Pipeline", "sweep_temp_dir", "Outcome"]
