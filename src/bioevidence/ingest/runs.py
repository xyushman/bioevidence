"""Persist small per-source ingestion run summaries for quality diffs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bioevidence.ingest.common import IngestStats, utcnow

DEFAULT_RUN_DIR = Path("results/ingestion")


def record_ingest_run(
    source: str,
    stats: IngestStats,
    row_count: int,
    output_dir: Path = DEFAULT_RUN_DIR,
) -> Path:
    """Write one immutable JSON summary for an ingestion run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    payload: dict[str, Any] = {
        "source": source,
        "timestamp": timestamp.replace("T", "T"),
        "row_count": row_count,
        **stats.as_dict(),
    }
    path = output_dir / f"{source}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_run_history(source: str, output_dir: Path = DEFAULT_RUN_DIR) -> list[dict[str, Any]]:
    """Read run summaries for ``source`` in chronological order."""
    if not output_dir.exists():
        return []
    pattern = re.compile(rf"^{re.escape(source)}_\d{{8}}T\d{{12}}Z\.json$")
    paths = sorted(path for path in output_dir.glob(f"{source}_*.json") if pattern.match(path.name))
    history: list[dict[str, Any]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            history.append(value)
    return history


def latest_run_diff(source: str, output_dir: Path = DEFAULT_RUN_DIR) -> dict[str, int] | None:
    """Return counts compared with the previous run, when two runs exist."""
    history = read_run_history(source, output_dir)
    if len(history) < 2:
        return None
    previous, current = history[-2], history[-1]
    previous_count = int(previous.get("row_count", 0))
    current_count = int(current.get("row_count", 0))
    return {
        "new": int(current.get("inserted", 0)),
        "updated": int(current.get("updated", 0)),
        "dropped": max(0, previous_count - current_count),
        "previous_count": previous_count,
        "current_count": current_count,
    }
