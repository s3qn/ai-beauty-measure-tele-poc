# AI Face Measurements — Telegram Bot POC (F1)

A Telegram bot that accepts a selfie, detects a single face, and returns a
**versioned set of geometric face measurements** (numbers only — no "after" image,
no advice). It gives concise capture guidance for unusable photos and logs outcomes
**without storing the raw image or unnecessary PII**.

Scope is **F1 only** — see `CLAUDE.md` for the full spec and constraints.

## Architecture

```
bot.py                 # entrypoint shim -> app.bot.main
app/
  bot.py               # Telegram handlers (wiring only)
  pipeline.py          # validate -> decode -> quality -> detect -> measure -> respond
  detection.py         # mediapipe Tasks FaceLandmarker (478 pts), single-face enforcement
  measurements.py      # landmark -> versioned measurement schema (the core)
  quality.py           # brightness / blur / size / centering checks
  schema.py            # ErrorCode enum, Measurement/Result dataclasses, schema_version, copy
  telemetry.py         # structured JSON-lines logging; no image, no PII
  config.py            # env loading + defaults
scripts/export_telemetry.py   # aggregate telemetry into success/failure stats
tests/                 # pytest: measurements, quality, validation, temp-cleanup
```

Each request: in-memory processing, a UUID4 `correlation_id` in every reply and log
line, structured error codes, and one telemetry line per request.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**System dependency (Linux, headless):** mediapipe's native runtime needs OpenGL ES.
If you see `libGLESv2.so.2: cannot open shared object file`, install:

```bash
sudo apt-get install -y libgles2 libegl1 libglvnd0
```

The FaceLandmarker model bundle (`app/assets/face_landmarker.task`, ~3.8 MB) is
downloaded and cached automatically on first run.

Then add your token:

```bash
cp .env.example .env   # edit and set TELEGRAM_BOT_TOKEN
```

## Run

```bash
python bot.py
```

Send the bot a single, front-facing selfie. It replies with measurements (or capture
guidance if the photo is unusable), always including a `correlation_id`.

## Tests

```bash
pytest
```

Tests use synthetic landmark arrays and generated images — **no biometric fixtures**
(face images are treated as sensitive data, even in tests; see `CLAUDE.md` §9/§10).

## Telemetry

One JSON line per request to `TELEMETRY_PATH` (default `telemetry.jsonl`). Aggregate:

```bash
python scripts/export_telemetry.py telemetry.jsonl
```

## Environment variables

See `.env.example`. Key ones: `TELEGRAM_BOT_TOKEN` (required), quality thresholds
(`BRIGHTNESS_MIN`, `BLUR_THRESHOLD`, `MIN_FACE_RATIO`, `MIN_RESOLUTION`),
`TARGET_LATENCY_MS`, `TEMP_DIR`, `TELEMETRY_PATH`, `CHAT_ID_SALT`.

> `.env` is git-ignored — never commit your real token.
