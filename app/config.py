"""Configuration via environment variables only (CLAUDE.md §8).

Never hardcode the bot token. ``Config.load()`` fails fast if the token is missing
(unless ``require_token=False``, used by tests that exercise non-bot logic).

Thresholds are POC defaults — tune against fixtures (§8/§10), don't treat as final.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Default model bundle for the mediapipe Tasks FaceLandmarker (478 landmarks).
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "assets" / "face_landmarker.task")
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    log_level: str = "INFO"
    target_latency_ms: int = 5000

    # Quality-gate thresholds (see quality.py for how each is applied).
    brightness_min: float = 50.0      # mean luma on 0-255; below => TOO_DARK
    blur_threshold: float = 100.0     # variance of Laplacian; below => TOO_BLURRY
    min_face_ratio: float = 0.15      # face-box width / image width; below => FACE_TOO_SMALL
    min_resolution: int = 200         # min(width, height) px; below => FACE_TOO_SMALL

    # Detection model.
    model_path: str = DEFAULT_MODEL_PATH
    model_url: str = DEFAULT_MODEL_URL

    # Data handling.
    temp_dir: str = "/tmp/beauty-bot"
    telemetry_path: str = "telemetry.jsonl"
    chat_id_salt: str = ""            # salt for hashing chat ids; empty => no per-user counter

    @classmethod
    def load(cls, *, require_token: bool = True) -> "Config":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if require_token and not token:
            raise SystemExit(
                "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and add your token."
            )
        return cls(
            telegram_bot_token=token,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            target_latency_ms=_get_int("TARGET_LATENCY_MS", 5000),
            brightness_min=_get_float("BRIGHTNESS_MIN", 50.0),
            blur_threshold=_get_float("BLUR_THRESHOLD", 100.0),
            min_face_ratio=_get_float("MIN_FACE_RATIO", 0.15),
            min_resolution=_get_int("MIN_RESOLUTION", 200),
            model_path=os.getenv("FACE_LANDMARKER_MODEL", DEFAULT_MODEL_PATH),
            model_url=os.getenv("FACE_LANDMARKER_MODEL_URL", DEFAULT_MODEL_URL),
            temp_dir=os.getenv("TEMP_DIR", "/tmp/beauty-bot"),
            telemetry_path=os.getenv("TELEMETRY_PATH", "telemetry.jsonl"),
            chat_id_salt=os.getenv("CHAT_ID_SALT", ""),
        )


__all__ = ["Config", "DEFAULT_MODEL_PATH", "DEFAULT_MODEL_URL"]
