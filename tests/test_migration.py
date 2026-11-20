"""Tests for the Alembic migration definitions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_initial_migration_creates_schema_and_indexes() -> None:
    """Offline SQL must contain every required table, index, and extension."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    sql = result.stdout

    for table in (
        "trials",
        "trial_conditions",
        "trial_interventions",
        "trial_outcomes",
        "pubmed_articles",
        "drug_labels",
        "chunks",
    ):
        assert f"CREATE TABLE {table}" in sql

    for index in (
        "ix_chunks_tsv",
        "ix_chunks_embedding",
        "ix_chunks_embedding_ft",
        "ix_trials_phase",
        "ix_trials_start_date",
        "ix_trials_overall_status",
    ):
        assert f"CREATE INDEX {index}" in sql

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert sql.count("CREATE TYPE doctype") == 1
    assert sql.count("CREATE TYPE outcomekind") == 1
    assert sql.count("CREATE TYPE splittype") == 1
