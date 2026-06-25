#!/usr/bin/env python3
"""Aggregate the telemetry JSON-lines file into success/failure counts (CLAUDE.md §9).

Usage:
    python scripts/export_telemetry.py [telemetry.jsonl]

Prints a JSON summary: total requests, outcome counts, per-error_code counts,
success rate, and processing-time percentiles. Reads only telemetry — never images.
"""

from __future__ import annotations

import json
import sys
from collections import Counter


def aggregate(path: str) -> dict:
    outcomes: Counter[str] = Counter()
    error_codes: Counter[str] = Counter()
    durations: list[int] = []
    total = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            outcomes[rec.get("outcome", "unknown")] += 1
            if rec.get("error_code"):
                error_codes[rec["error_code"]] += 1
            ms = rec.get("processing_ms")
            if isinstance(ms, (int, float)):
                durations.append(int(ms))

    durations.sort()

    def pct(p: float) -> int | None:
        if not durations:
            return None
        idx = min(len(durations) - 1, int(round((p / 100.0) * (len(durations) - 1))))
        return durations[idx]

    success = outcomes.get("success", 0)
    return {
        "total_requests": total,
        "outcomes": dict(outcomes),
        "error_codes": dict(error_codes),
        "success_rate": round(success / total, 4) if total else None,
        "processing_ms": {
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "max": durations[-1] if durations else None,
        },
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl"
    try:
        summary = aggregate(path)
    except FileNotFoundError:
        print(f"No telemetry file at {path!r} yet.", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
