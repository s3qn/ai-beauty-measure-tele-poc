"""Beauty App — Stage 1 (F1) face-measurement pipeline.

Package layout (see CLAUDE.md §3):
    bot.py          Telegram handlers (wiring only)
    pipeline.py     orchestration: validate -> detect -> measure -> respond
    detection.py    face + landmark detection, single-face enforcement
    measurements.py landmark -> versioned measurement schema
    quality.py      brightness/blur/size/centering checks + guidance
    schema.py       dataclasses/enums: MeasurementResult, ErrorCode, schema_version
    telemetry.py    structured logging; never raw image or PII
    config.py       env loading, defaults
"""
