"""Idempotent quarantine storage for records rejected during validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bioevidence.ingest.common import utcnow

DEFAULT_QUARANTINE_DIR = Path("data/processed")
_SEEN_IDS: dict[Path, set[str]] = {}


def _jsonable(value: Any) -> Any:
    """Convert common ORM/date values into JSON-safe data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _entry_id(source: str, reason: str, record: Any) -> str:
    payload = json.dumps(
        {"source": source, "reason": reason, "record": _jsonable(record)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quarantine_record(
    source: str,
    record: Any,
    reason: str,
    output_dir: Path = DEFAULT_QUARANTINE_DIR,
) -> Path:
    """Append a rejected record once, returning the quarantine file path."""
    path = output_dir / f"rejected_{source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = _SEEN_IDS.setdefault(path, set())
    if not seen and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("id"):
                seen.add(str(entry["id"]))

    entry_id = _entry_id(source, reason, record)
    if entry_id in seen:
        return path

    entry = {
        "id": entry_id,
        "quarantined_at": utcnow().isoformat() + "Z",
        "source": source,
        "reason": reason,
        "record": _jsonable(record),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    seen.add(entry_id)
    return path
