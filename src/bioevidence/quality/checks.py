"""Typed data quality checks and markdown report generation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioevidence.chunking.splits import verify_no_db_leakage
from bioevidence.db.models import (
    Chunk,
    DrugLabel,
    PubMedArticle,
    SplitType,
    Trial,
    TrialCondition,
    TrialIntervention,
    TrialOutcome,
)
from bioevidence.db.session import get_session
from bioevidence.ingest.runs import latest_run_diff

logger = structlog.get_logger(__name__)

_NON_LATIN_SCRIPT = re.compile(
    "[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u0e00-\u0e7f]"
)
_COMPLETENESS_FIELDS: dict[str, tuple[Any, tuple[str, ...]]] = {
    "trials": (
        Trial,
        (
            "title",
            "brief_summary",
            "detailed_description",
            "phase",
            "overall_status",
            "start_date",
            "completion_date",
            "enrollment",
            "sponsor",
        ),
    ),
    "pubmed_articles": (
        PubMedArticle,
        ("title", "abstract", "journal", "year", "mesh_terms"),
    ),
    "drug_labels": (
        DrugLabel,
        (
            "brand_names",
            "generic_names",
            "sections",
            "effective_date",
            "matched_interventions",
            "fetched_at",
        ),
    ),
}


@dataclass
class QualityCheck:
    """Result of one quality check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete quality report."""

    checks: list[QualityCheck] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_failures(self) -> int:
        return sum(1 for check in self.checks if not check.passed and check.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for check in self.checks if not check.passed and check.severity == "warn")

    def to_markdown(self) -> str:
        """Render a human-readable report with tabular quality metrics."""
        lines = [
            "# Data Quality Report",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total checks: {len(self.checks)}",
            f"- Passed: {sum(1 for check in self.checks if check.passed)}",
            f"- Critical failures: {self.critical_failures}",
            f"- Warnings: {self.warnings}",
            "",
            "## Checks",
            "",
        ]
        for check in self.checks:
            status = "PASS" if check.passed else ("FAIL" if check.severity == "error" else "WARN")
            lines.append(f"- **{status}** `{check.name}`: {check.message}")

        lines.extend(["", "## Completeness", ""])
        lines.append("| Source | Field | Null count | Null % |")
        lines.append("| --- | --- | ---: | ---: |")
        completeness = self.stats.get("completeness", {})
        for source, fields in completeness.items():
            for item in fields:
                lines.append(
                    f"| {source} | {item['field']} | {item['null_count']} | "
                    f"{item['null_pct']:.1f} |"
                )

        lines.extend(["", "## Text Length Distribution", ""])
        lines.append("| Source | Count | Min | P50 | P90 | Max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for source, distribution in self.stats.get("text_length_distribution", {}).items():
            lines.append(
                f"| {source} | {distribution['count']} | {distribution['min']} | "
                f"{distribution['p50']} | {distribution['p90']} | {distribution['max']} |"
            )

        lines.extend(["", "## Ingestion Run Diffs", ""])
        lines.append("| Source | Previous | Current | Inserted | Updated | Dropped |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for source, diff in self.stats.get("run_diffs", {}).items():
            if diff is None:
                lines.append(f"| {source} | - | - | - | - | - |")
                continue
            lines.append(
                f"| {source} | {diff['previous_count']} | {diff['current_count']} | "
                f"{diff['new']} | {diff['updated']} | {diff['dropped']} |"
            )

        if self.stats.get("row_counts"):
            lines.extend(["", "## Row Counts", "", "```json"])
            lines.append(json.dumps(self.stats["row_counts"], indent=2, sort_keys=True))
            lines.append("```")
        return "\n".join(lines) + "\n"


def _count(session: Session, statement: Any) -> int:
    return int(session.scalar(statement) or 0)


def _distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0}

    def percentile(fraction: float) -> int:
        position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return ordered[position]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


def non_latin_ratio(texts: Iterable[str]) -> float:
    """Return the fraction of letters written in non-Latin scripts."""
    total = 0
    non_latin = 0
    for text in texts:
        for char in text:
            if not char.isalpha():
                continue
            total += 1
            if _NON_LATIN_SCRIPT.search(char):
                non_latin += 1
    return non_latin / total if total else 0.0


class QualityChecker:
    """Run all quality checks against the configured database."""

    def __init__(self) -> None:
        self.report = QualityReport()

    def _add_check(
        self,
        name: str,
        passed: bool,
        message: str,
        severity: str = "error",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.report.checks.append(
            QualityCheck(
                name=name,
                passed=passed,
                message=message,
                severity=severity,
                details=details or {},
            )
        )

    def run_all_checks(self) -> QualityReport:
        """Run every quality check and collect report statistics."""
        self.report = QualityReport()
        logger.info("starting_quality_checks")
        with get_session() as session:
            row_counts = self._collect_row_counts(session)
            self._check_volumes(row_counts)
            self._check_required_fields(session, row_counts)
            self._check_uniqueness(session)
            self._check_value_ranges(session)
            self._check_referential_integrity(session)
            self._check_chunks(session)
            self._check_splits()
            self._check_completeness(session, row_counts)
            self._check_text_lengths(session)
            self._check_language_sanity(session)
        self._check_run_diffs()
        logger.info(
            "quality_checks_complete",
            critical_failures=self.report.critical_failures,
            warnings=self.report.warnings,
        )
        return self.report

    @staticmethod
    def _collect_row_counts(session: Session) -> dict[str, int]:
        counts = {
            "trials": _count(session, select(func.count()).select_from(Trial)),
            "trial_conditions": _count(session, select(func.count()).select_from(TrialCondition)),
            "trial_interventions": _count(
                session, select(func.count()).select_from(TrialIntervention)
            ),
            "trial_outcomes": _count(session, select(func.count()).select_from(TrialOutcome)),
            "pubmed_articles": _count(session, select(func.count()).select_from(PubMedArticle)),
            "drug_labels": _count(session, select(func.count()).select_from(DrugLabel)),
            "chunks": _count(session, select(func.count()).select_from(Chunk)),
        }
        return counts

    def _check_volumes(self, counts: dict[str, int]) -> None:
        self.report.stats["row_counts"] = counts
        for name, minimum in (("trials", 1), ("pubmed_articles", 1), ("drug_labels", 1)):
            self._add_check(
                f"{name}_populated",
                counts[name] >= minimum,
                f"{name}: {counts[name]} rows",
                details={"count": counts[name], "minimum": minimum},
            )

    def _check_required_fields(self, session: Session, counts: dict[str, int]) -> None:
        bad_trials = _count(
            session,
            select(func.count())
            .select_from(Trial)
            .where(
                Trial.title.is_(None)
                | (Trial.brief_summary.is_(None) & Trial.detailed_description.is_(None))
            ),
        )
        self._add_check(
            "trials_have_title_and_text",
            bad_trials == 0,
            f"Trials missing title or narrative text: {bad_trials}",
            details={"count": bad_trials, "total": counts["trials"]},
        )

        bad_articles = _count(
            session,
            select(func.count())
            .select_from(PubMedArticle)
            .where(PubMedArticle.title.is_(None) | PubMedArticle.abstract.is_(None)),
        )
        self._add_check(
            "pubmed_have_title_and_abstract",
            bad_articles == 0,
            f"PubMed articles missing title or abstract: {bad_articles}",
            details={"count": bad_articles, "total": counts["pubmed_articles"]},
        )

        bad_labels = _count(
            session,
            select(func.count())
            .select_from(DrugLabel)
            .where(DrugLabel.sections.is_(None) | (DrugLabel.set_id.is_(None))),
        )
        self._add_check(
            "drug_labels_have_id_and_sections",
            bad_labels == 0,
            f"Drug labels missing set_id or sections: {bad_labels}",
            details={"count": bad_labels, "total": counts["drug_labels"]},
        )

    def _check_uniqueness(self, session: Session) -> None:
        for name, column in (
            ("trials_unique_nct_id", Trial.nct_id),
            ("pubmed_unique_pmid", PubMedArticle.pmid),
            ("drug_labels_unique_set_id", DrugLabel.set_id),
        ):
            duplicates = session.execute(
                select(column).group_by(column).having(func.count() > 1).limit(5)
            ).all()
            self._add_check(
                name,
                not duplicates,
                f"Duplicate natural keys: {len(duplicates)}",
                details={"examples": [row[0] for row in duplicates]},
            )

    def _check_value_ranges(self, session: Session) -> None:
        negative_enrollment = _count(
            session, select(func.count()).select_from(Trial).where(Trial.enrollment < 0)
        )
        self._add_check(
            "trials_enrollment_non_negative",
            negative_enrollment == 0,
            f"Trials with negative enrollment: {negative_enrollment}",
            details={"count": negative_enrollment},
        )

        bad_dates = _count(
            session,
            select(func.count())
            .select_from(Trial)
            .where(
                Trial.completion_date.is_not(None)
                & Trial.start_date.is_not(None)
                & (Trial.completion_date < Trial.start_date)
            ),
        )
        self._add_check(
            "trials_date_consistency",
            bad_dates == 0,
            f"Trials with completion before start: {bad_dates}",
            severity="warn",
            details={"count": bad_dates},
        )

        bad_years = _count(
            session,
            select(func.count())
            .select_from(PubMedArticle)
            .where((PubMedArticle.year < 1950) | (PubMedArticle.year > 2100)),
        )
        self._add_check(
            "pubmed_year_range",
            bad_years == 0,
            f"PubMed articles with implausible year: {bad_years}",
            severity="warn",
            details={"count": bad_years},
        )

    def _check_referential_integrity(self, session: Session) -> None:
        for name, model, parent in (
            ("ref_integrity_trial_conditions", TrialCondition, Trial.nct_id),
            ("ref_integrity_trial_interventions", TrialIntervention, Trial.nct_id),
            ("ref_integrity_trial_outcomes", TrialOutcome, Trial.nct_id),
        ):
            orphans = _count(
                session,
                select(func.count()).select_from(model).where(~model.nct_id.in_(select(parent))),
            )
            self._add_check(
                name,
                orphans == 0,
                f"Orphan {model.__tablename__} rows: {orphans}",
                details={"count": orphans},
            )

    def _check_chunks(self, session: Session) -> None:
        empty_text = _count(
            session,
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.text.is_(None) | (Chunk.text == "")),
        )
        self._add_check(
            "chunks_not_null_text",
            empty_text == 0,
            f"Chunks with null/empty text: {empty_text}",
            details={"count": empty_text},
        )
        token_stats = session.execute(
            select(
                func.coalesce(func.min(Chunk.token_count), 0),
                func.coalesce(func.max(Chunk.token_count), 0),
                func.coalesce(func.avg(Chunk.token_count), 0),
            )
        ).one()
        self.report.stats["chunk_token_stats"] = {
            "min": int(token_stats[0]),
            "max": int(token_stats[1]),
            "avg": round(float(token_stats[2]), 1),
        }

    def _check_splits(self) -> None:
        with get_session() as session:
            counts = {
                split.value: _count(
                    session,
                    select(func.count()).select_from(Chunk).where(Chunk.split == split),
                )
                for split in SplitType
            }
        self.report.stats["split_counts"] = counts
        self._add_check(
            "splits_populated",
            all(count > 0 for count in counts.values()),
            f"Chunk counts by split: {counts}",
            details=counts,
        )
        no_leakage = verify_no_db_leakage()
        self._add_check(
            "splits_no_leakage",
            no_leakage,
            "No document appears in more than one split"
            if no_leakage
            else "Split leakage detected",
        )

    def _check_completeness(self, session: Session, counts: dict[str, int]) -> None:
        completeness: dict[str, list[dict[str, Any]]] = {}
        for source, (model, columns) in _COMPLETENESS_FIELDS.items():
            total = counts.get(source, 0)
            entries: list[dict[str, Any]] = []
            for column_name in columns:
                null_count = _count(
                    session,
                    select(func.count())
                    .select_from(model)
                    .where(getattr(model, column_name).is_(None)),
                )
                entries.append(
                    {
                        "field": column_name,
                        "null_count": null_count,
                        "null_pct": round(null_count / total * 100, 1) if total else 0.0,
                    }
                )
            completeness[source] = entries
        self.report.stats["completeness"] = completeness

    def _check_text_lengths(self, session: Session) -> None:
        trial_lengths = [
            int(value or 0)
            for value in session.execute(
                select(
                    func.coalesce(func.length(Trial.brief_summary), 0)
                    + func.coalesce(func.length(Trial.detailed_description), 0)
                )
            ).scalars()
        ]
        pubmed_lengths = [
            int(value or 0)
            for value in session.execute(
                select(func.coalesce(func.length(PubMedArticle.abstract), 0))
            ).scalars()
        ]
        label_lengths = []
        for sections in session.execute(select(DrugLabel.sections)).scalars():
            if isinstance(sections, dict):
                label_lengths.append(sum(len(str(value)) for value in sections.values()))
        chunk_lengths = [
            int(value or 0) for value in session.execute(select(func.length(Chunk.text))).scalars()
        ]
        self.report.stats["text_length_distribution"] = {
            "trial_text_chars": _distribution(trial_lengths),
            "pubmed_abstract_chars": _distribution(pubmed_lengths),
            "drug_label_section_chars": _distribution(label_lengths),
            "chunk_chars": _distribution(chunk_lengths),
        }

    def _check_language_sanity(self, session: Session) -> None:
        samples: list[str] = []
        for value in session.execute(
            select(Trial.brief_summary).where(Trial.brief_summary.is_not(None)).limit(250)
        ).scalars():
            if value:
                samples.append(str(value))
        for value in session.execute(
            select(PubMedArticle.abstract).where(PubMedArticle.abstract.is_not(None)).limit(250)
        ).scalars():
            if value:
                samples.append(str(value))
        for value in session.execute(
            select(Chunk.text).where(Chunk.text.is_not(None)).limit(250)
        ).scalars():
            if value:
                samples.append(str(value))
        ratio = non_latin_ratio(str(value) for value in samples if value)
        self.report.stats["language_sanity"] = {
            "non_latin_letter_ratio": round(ratio, 4),
            "sample_size": len(samples),
        }
        self._add_check(
            "language_sanity_english",
            ratio < 0.01,
            f"Non-Latin letter ratio in sample: {ratio:.4f}",
            severity="warn",
            details={"ratio": round(ratio, 4), "sample_size": len(samples)},
        )

    def _check_run_diffs(self) -> None:
        diffs = {source: latest_run_diff(source) for source in ("ctgov", "pubmed", "openfda")}
        self.report.stats["run_diffs"] = diffs
        missing = [source for source, diff in diffs.items() if diff is None]
        self._add_check(
            "ingestion_run_summaries",
            not missing,
            "All sources have comparable run summaries" if not missing else f"Missing: {missing}",
            severity="warn",
            details={"missing": missing},
        )


def run_quality_checks(output_path: Path | None = None) -> QualityReport:
    """Run all checks, write markdown and a small JSON result, and return the report."""
    report = QualityChecker().run_all_checks()

    results_dir = Path("results/quality")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    result_path = results_dir / f"quality_{timestamp}.json"
    result_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "critical_failures": report.critical_failures,
                "warnings": report.warnings,
                "checks": [check.__dict__ for check in report.checks],
                "stats": report.stats,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("quality_result_written", path=str(result_path))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("quality_report_written", path=str(output_path))
    return report
