"""Shared helpers for official-source data ingestion."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class IngestStats:
    """Statistics for one ingestion run."""

    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable snapshot."""
        return asdict(self)


class RawCache:
    """Gzip JSON cache for raw official API responses."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json.gz"

    def get(self, key: str) -> Any | None:
        """Return cached data, or ``None`` on miss/corruption."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                content = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            msg = f"Failed to read raw cache {path}: {exc}"
            raise RuntimeError(msg) from exc
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    def set(self, key: str, data: Any) -> None:
        """Write data to the raw cache."""
        path = self._cache_path(key)
        try:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                if isinstance(data, str):
                    handle.write(data)
                else:
                    json.dump(data, handle, ensure_ascii=False)
        except (OSError, TypeError) as exc:
            msg = f"Failed to write raw cache {path}: {exc}"
            raise RuntimeError(msg) from exc


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            msg = "requests_per_second must be positive"
            raise ValueError(msg)
        self.min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next request is allowed."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


def clean_text(value: str | None) -> str | None:
    """Strip markup and normalize whitespace without changing meaning."""
    if value is None:
        return None
    without_tags = _HTML_TAG.sub(" ", value)
    unescaped = html.unescape(without_tags)
    normalized = _WHITESPACE.sub(" ", unescaped).strip()
    return normalized or None


def utcnow() -> datetime:
    """Return a naive UTC timestamp compatible with SQLAlchemy DateTime columns."""
    return datetime.now(UTC).replace(tzinfo=None)
