"""Tests for quality reports and ingestion run diffs."""

from __future__ import annotations

from pathlib import Path

from bioevidence.ingest.common import IngestStats
from bioevidence.ingest.runs import latest_run_diff, record_ingest_run
from bioevidence.quality import QualityReport, non_latin_ratio


def test_report_markdown_contains_quality_tables() -> None:
    report = QualityReport(
        stats={
            "completeness": {"trials": [{"field": "phase", "null_count": 1, "null_pct": 2.5}]},
            "text_length_distribution": {
                "pubmed_abstract_chars": {
                    "count": 10,
                    "min": 20,
                    "p50": 100,
                    "p90": 180,
                    "max": 200,
                }
            },
            "run_diffs": {
                "ctgov": {
                    "previous_count": 10,
                    "current_count": 12,
                    "new": 2,
                    "updated": 0,
                    "dropped": 0,
                }
            },
        }
    )
    markdown = report.to_markdown()

    assert "## Completeness" in markdown
    assert "| trials | phase | 1 | 2.5 |" in markdown
    assert "## Text Length Distribution" in markdown
    assert "| pubmed_abstract_chars | 10 | 20 | 100 | 180 | 200 |" in markdown
    assert "## Ingestion Run Diffs" in markdown
    assert "| ctgov | 10 | 12 | 2 | 0 | 0 |" in markdown


def test_non_latin_ratio_detects_non_english_script() -> None:
    assert non_latin_ratio(["English biomedical abstract."]) == 0.0
    ratio = non_latin_ratio(["English text with 中文 characters"])
    assert ratio > 0.0


def test_latest_run_diff_uses_two_latest_summaries(tmp_path: Path) -> None:
    first = IngestStats(fetched=10, inserted=10)
    second = IngestStats(fetched=12, inserted=2, updated=10)
    record_ingest_run("synthetic", first, 10, tmp_path)
    record_ingest_run("synthetic", second, 12, tmp_path)

    diff = latest_run_diff("synthetic", tmp_path)
    assert diff == {
        "new": 2,
        "updated": 10,
        "dropped": 0,
        "previous_count": 10,
        "current_count": 12,
    }
