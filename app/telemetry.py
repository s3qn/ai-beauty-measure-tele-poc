"""Structured, privacy-preserving telemetry (CLAUDE.md §9).

One JSON line per request to an exportable ``.jsonl`` file. NEVER logs image bytes,
file paths to user images, usernames, or raw chat ids. If a per-user counter is
needed, the chat id is salted-hashed (``hash_chat_id``) and only the hash is stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

from .schema import LANDMARK_MODEL, SCHEMA_VERSION, ErrorCode

logger = logging.getLogger("beauty_bot.telemetry")


def hash_chat_id(chat_id: int | str, salt: str) -> Optional[str]:
    """Salted SHA-256 of a chat id (first 16 hex chars), or None if no salt set.

    Without a salt we store nothing identifying — per §9, the per-user counter is
    opt-in via CHAT_ID_SALT.
    """
    if not salt:
        return None
    digest = hashlib.sha256(f"{salt}:{chat_id}".encode("utf-8")).hexdigest()
    return digest[:16]


def log_outcome(
    *,
    correlation_id: str,
    outcome: str,
    processing_ms: int,
    error_code: Optional[ErrorCode] = None,
    chat_id_hash: Optional[str] = None,
    telemetry_path: str = "telemetry.jsonl",
    now: Optional[float] = None,
) -> dict:
    """Append one telemetry record and return it. ``outcome`` is "success"|"rejected"|"error".

    The record schema is fixed (§9): ts, correlation_id, outcome, error_code|null,
    processing_ms, landmark_model, schema_version. ``chat_id_hash`` is included only
    when a salt produced one.
    """
    record = {
        "ts": now if now is not None else time.time(),
        "correlation_id": correlation_id,
        "outcome": outcome,
        "error_code": error_code.value if error_code else None,
        "processing_ms": processing_ms,
        "landmark_model": LANDMARK_MODEL,
        "schema_version": SCHEMA_VERSION,
    }
    if chat_id_hash:
        record["chat_id_hash"] = chat_id_hash

    try:
        with open(telemetry_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # never let telemetry failure break the request
        logger.warning("telemetry write failed (%s): %s", correlation_id, exc)

    logger.info(
        "outcome=%s code=%s processing_ms=%d correlation_id=%s",
        outcome,
        error_code.value if error_code else "none",
        processing_ms,
        correlation_id,
    )
    return record


__all__ = ["hash_chat_id", "log_outcome"]
