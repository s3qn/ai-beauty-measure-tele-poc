"""Versioned measurement schema, error codes, and user-facing copy.

This module is the *contract* every other module depends on (CLAUDE.md §5/§6/§7).
Keep it free of heavy imports (no mediapipe/Pillow) so it's cheap to import in tests.

Versioning (semver on ``SCHEMA_VERSION``):
  - patch: doc-only clarification
  - minor: added field
  - major: renamed/removed field, or changed unit/normalization basis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# Bump per the rules above. Mirrors the SCHEMA_VERSION env default in §8.
SCHEMA_VERSION = "1.0.0"

# Identifies the landmark backend so downstream consumers can interpret indices.
# mediapipe 0.10.x ships the Tasks FaceLandmarker (478 landmarks incl. iris).
LANDMARK_MODEL = "mediapipe_face_landmarker_v2_478pt"

# All distance measurements are normalized by this basis (CLAUDE.md §5).
NORMALIZATION_BASIS = "interpupillary_distance"


class ErrorCode(str, Enum):
    """Structured outcome codes (CLAUDE.md §6). Logs use these; users see copy below.

    Inherits ``str`` so the value serializes directly to JSON and compares to plain
    strings in tests.
    """

    NOT_AN_IMAGE = "NOT_AN_IMAGE"
    DECODE_FAILED = "DECODE_FAILED"
    TOO_DARK = "TOO_DARK"
    TOO_BLURRY = "TOO_BLURRY"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Centralized user-facing guidance (CLAUDE.md §7). Capture-only; no diagnosis,
# no cosmetic/medical language. Keep literals here — never scatter them.
GUIDANCE: dict[ErrorCode, str] = {
    ErrorCode.NOT_AN_IMAGE: "Please send a photo (JPG or PNG) to get started.",
    ErrorCode.DECODE_FAILED: "I couldn't read that image. Please resend a standard JPG or PNG.",
    ErrorCode.TOO_DARK: "It's a bit dark. Move to brighter, even lighting and try again.",
    ErrorCode.TOO_BLURRY: "That came out blurry. Hold steady and retake.",
    ErrorCode.FACE_TOO_SMALL: "Move a little closer and center your face in the frame.",
    ErrorCode.NO_FACE: "I couldn't find a face. Send a clear, front-facing selfie.",
    ErrorCode.MULTIPLE_FACES: "I see more than one face. Send a photo with just you in it.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on my side. Please try again.",
}


def guidance_for(code: ErrorCode) -> str:
    """User-facing capture guidance for an error code. Falls back to a generic line."""
    return GUIDANCE.get(code, GUIDANCE[ErrorCode.INTERNAL_ERROR])


@dataclass(frozen=True)
class Measurement:
    """A single measured value with its unit and a human-readable description.

    ``unit`` vocabulary (documented, not free-text):
      - ``ratio``        : unitless ratio of two pixel distances in the same image
      - ``ratio_self``   : normalized against itself (always 1.0; the basis marker)
      - ``ratio_ipd``    : distance divided by interpupillary distance
      - ``ratio_0to1``   : bounded score in [0, 1]
      - ``degrees``      : angle in degrees
    """

    value: float
    unit: str
    description: str = ""

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "description": self.description}


@dataclass
class MeasurementResult:
    """Successful pipeline output — the versioned envelope (CLAUDE.md §5)."""

    correlation_id: str
    measurements: dict[str, Measurement]
    processing_ms: Optional[int] = None
    schema_version: str = SCHEMA_VERSION
    landmark_model: str = LANDMARK_MODEL
    normalization_basis: str = NORMALIZATION_BASIS

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "correlation_id": self.correlation_id,
            "landmark_model": self.landmark_model,
            "normalization_basis": self.normalization_basis,
            "processing_ms": self.processing_ms,
            "measurements": {k: m.to_dict() for k, m in self.measurements.items()},
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PipelineError:
    """Failure outcome — carries a structured code plus the correlation id.

    Never carries internals destined for the user. ``detail`` is for logs only and
    must not be surfaced verbatim to a person (see INTERNAL_ERROR handling).
    """

    correlation_id: str
    code: ErrorCode
    processing_ms: Optional[int] = None
    detail: str = ""  # log-only; never shown to the user

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "correlation_id": self.correlation_id,
            "error_code": self.code.value,
            "processing_ms": self.processing_ms,
        }


__all__ = [
    "SCHEMA_VERSION",
    "LANDMARK_MODEL",
    "NORMALIZATION_BASIS",
    "ErrorCode",
    "GUIDANCE",
    "guidance_for",
    "Measurement",
    "MeasurementResult",
    "PipelineError",
]
