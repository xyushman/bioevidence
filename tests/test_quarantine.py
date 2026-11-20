"""Tests for idempotent quarantine storage."""

from __future__ import annotations

from pathlib import Path

from bioevidence.quality.quarantine import quarantine_record


def test_quarantine_writes_each_rejected_record_once(tmp_path: Path) -> None:
    path = quarantine_record("synthetic", {"pmid": "1"}, "invalid year", tmp_path)
    quarantine_record("synthetic", {"pmid": "1"}, "invalid year", tmp_path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"reason": "invalid year"' in lines[0]
