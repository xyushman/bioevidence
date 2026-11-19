"""Tests for CLI."""

import pytest
from typer.testing import CliRunner

from bioevidence.__main__ import app


@pytest.fixture
def runner():
    return CliRunner()


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
