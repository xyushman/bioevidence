"""Tests for CLI."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from bioevidence.__main__ import app


@pytest.fixture
def runner():
    return CliRunner()


def _stats() -> SimpleNamespace:
    return SimpleNamespace(fetched=1, inserted=1, updated=0, errors=0)


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "BioEvidence" in result.output


def test_cli_ingest_help(runner):
    """Test ingest subcommand help."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "source" in result.output


def test_cli_quality_help(runner):
    """Test quality subcommand help."""
    result = runner.invoke(app, ["quality", "--help"])
    assert result.exit_code == 0


def test_cli_callback(runner):
    """Test CLI callback."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "BioEvidence" in result.output


def test_ingest_ctgov_command_calls_ingestor(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_ingest(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _stats()

    monkeypatch.setattr("bioevidence.__main__.ingest_ctgov_fn", fake_ingest)
    result = runner.invoke(app, ["ingest-ctgov", "--max-studies", "5", "--max-pages", "1"])

    assert result.exit_code == 0, result.output
    assert captured["max_studies"] == 5
    assert captured["max_pages"] == 1
    assert captured["start_date_min"] == "2005-01-01"


def test_ingest_openfda_command_calls_ingestor(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_ingest(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _stats()

    monkeypatch.setattr("bioevidence.__main__.ingest_openfda_fn", fake_ingest)
    result = runner.invoke(app, ["ingest-openfda", "--max-labels", "3"])

    assert result.exit_code == 0, result.output
    assert captured["max_labels"] == 3
    assert captured["use_bulk"] is False


def test_db_migrate_command_calls_alembic(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    revisions: list[str] = []

    monkeypatch.setattr(
        "alembic.command.upgrade", lambda _config, revision: revisions.append(revision)
    )
    result = runner.invoke(app, ["db", "migrate"])

    assert result.exit_code == 0, result.output
    assert revisions == ["head"]


def test_splits_make_passes_typed_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_make_splits(config: Any) -> dict[str, Any]:
        captured["config"] = config
        return {
            "trials": {"train": [1], "dev": [2], "test": [3]},
            "pubmed": {"train": [], "dev": [], "test": []},
            "labels": {"train": [], "dev": [], "test": []},
        }

    monkeypatch.setattr("bioevidence.__main__.make_splits", fake_make_splits)
    result = runner.invoke(app, ["splits", "make", "--seed", "7"])

    assert result.exit_code == 0, result.output
    assert captured["config"].seed == 7


def test_sample_create_passes_seed(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("bioevidence.__main__.create_sample", fake_create)
    result = runner.invoke(app, ["sample", "create", "--seed", "11"])

    assert result.exit_code == 0, result.output
    assert captured["seed"] == 11
