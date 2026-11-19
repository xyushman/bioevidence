"""Data quality checks for BioEvidence."""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

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

logger = structlog.get_logger(__name__)


@dataclass
class QualityCheck:
    """Result of a quality check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" or "warn"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete quality report."""

    checks: list[QualityCheck] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_failures(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warn")

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Data Quality Report",
            f"\nGenerated: {datetime.utcnow().isoformat()}Z",
            "\n## Summary",
            f"- Total checks: {len(self.checks)}",
            f"- Passed: {sum(1 for c in self.checks if c.passed)}",
            f"- Failed (critical): {self.critical_failures}",
            f"- Warnings: {self.warnings}",
            "\n## Checks",
        ]

        for check in self.checks:
            status = "✅" if check.passed else ("❌" if check.severity == "error" else "⚠️")
            lines.append(f"\n### {status} {check.name}")
            lines.append(f"**Severity:** {check.severity}")
            lines.append(f"**Message:** {check.message}")
            if check.details:
                lines.append("**Details:**")
                for k, v in check.details.items():
                    lines.append(f"- {k}: {v}")

        if self.stats:
            lines.append("\n## Statistics")
            lines.append("```json")
            lines.append(json.dumps(self.stats, indent=2, default=str))
            lines.append("```")

        return "\n".join(lines)


class QualityChecker:
    """Run data quality checks on the database."""

    def __init__(self):
        self.report = QualityReport()

    def _add_check(
        self,
        name: str,
        passed: bool,
        message: str,
        severity: str = "error",
        details: dict | None = None,
    ):
        """Add a check result to the report."""
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
        """Run all quality checks."""
        logger.info("starting_quality_checks")

        with get_session() as session:
            self._check_trials(session)
            self._check_pubmed(session)
            self._check_drug_labels(session)
            self._check_chunks(session)
            self._check_splits(session)
            self._check_referential_integrity(session)
            self._collect_stats(session)

        logger.info(
            "quality_checks_complete",
            critical_failures=self.report.critical_failures,
            warnings=self.report.warnings,
        )
        return self.report

    def _check_trials(self, session):
        """Check trials table quality."""
        # Not-null checks
        total = session.query(func.count(Trial.nct_id)).scalar()
        null_title = session.query(func.count(Trial.nct_id)).filter(Trial.title.is_(None)).scalar()
        null_summary = (
            session.query(func.count(Trial.nct_id))
            .filter(Trial.brief_summary.is_(None) & Trial.detailed_description.is_(None))
            .scalar()
        )

        self._add_check(
            "trials_not_null_nct_id", total > 0, f"Total trials: {total}", details={"total": total}
        )
        self._add_check(
            "trials_not_null_title",
            null_title == 0,
            f"Trials with null title: {null_title}",
            severity="error" if null_title > 0 else "error",
            details={"null_count": null_title},
        )
        self._add_check(
            "trials_has_text_content",
            null_summary == 0,
            f"Trials with no summary AND no description: {null_summary}",
            severity="error" if null_summary > 0 else "error",
            details={"null_count": null_summary},
        )

        # Uniqueness
        dup_nct = session.execute(
            select(Trial.nct_id, func.count()).group_by(Trial.nct_id).having(func.count() > 1)
        ).all()
        self._add_check(
            "trials_unique_nct_id",
            len(dup_nct) == 0,
            f"Duplicate NCT IDs: {len(dup_nct)}",
            details={"duplicates": [d[0] for d in dup_nct[:10]]},
        )

        # Range checks
        neg_enrollment = (
            session.query(func.count(Trial.nct_id)).filter(Trial.enrollment < 0).scalar()
        )
        self._add_check(
            "trials_enrollment_non_negative",
            neg_enrollment == 0,
            f"Trials with negative enrollment: {neg_enrollment}",
            details={"count": neg_enrollment},
        )

        # Date consistency
        bad_dates = (
            session.query(func.count(Trial.nct_id))
            .filter(
                Trial.completion_date.isnot(None)
                & Trial.start_date.isnot(None)
                & (Trial.completion_date < Trial.start_date)
            )
            .scalar()
        )
        self._add_check(
            "trials_date_consistency",
            bad_dates == 0,
            f"Trials with completion_date < start_date: {bad_dates}",
            severity="warn",
            details={"count": bad_dates},
        )

        # Year range
        future_trials = (
            session.query(func.count(Trial.nct_id)).filter(Trial.start_date > date.today()).scalar()
        )
        self._add_check(
            "trials_year_range",
            future_trials == 0,
            f"Trials with future start_date: {future_trials}",
            severity="warn",
            details={"count": future_trials},
        )

    def _check_pubmed(self, session):
        """Check PubMed articles quality."""
        total = session.query(func.count(PubMedArticle.pmid)).scalar()
        null_title = (
            session.query(func.count(PubMedArticle.pmid))
            .filter(PubMedArticle.title.is_(None))
            .scalar()
        )
        null_abstract = (
            session.query(func.count(PubMedArticle.pmid))
            .filter(PubMedArticle.abstract.is_(None))
            .scalar()
        )

        self._add_check(
            "pubmed_not_null_pmid",
            total > 0,
            f"Total PubMed articles: {total}",
            details={"total": total},
        )
        self._add_check(
            "pubmed_not_null_title",
            null_title == 0,
            f"Articles with null title: {null_title}",
            details={"null_count": null_title},
        )
        self._add_check(
            "pubmed_has_abstract",
            null_abstract == 0,
            f"Articles with null abstract: {null_abstract}",
            severity="warn",
            details={"null_count": null_abstract},
        )

        # Year range
        bad_years = (
            session.query(func.count(PubMedArticle.pmid))
            .filter((PubMedArticle.year < 1950) | (PubMedArticle.year > date.today().year + 1))
            .scalar()
        )
        self._add_check(
            "pubmed_year_range",
            bad_years == 0,
            f"Articles with year outside 1950..current+1: {bad_years}",
            severity="warn",
            details={"count": bad_years},
        )

        # Duplicate PMIDs
        dup_pmid = session.execute(
            select(PubMedArticle.pmid, func.count())
            .group_by(PubMedArticle.pmid)
            .having(func.count() > 1)
        ).all()
        self._add_check(
            "pubmed_unique_pmid",
            len(dup_pmid) == 0,
            f"Duplicate PMIDs: {len(dup_pmid)}",
            details={"duplicates": [d[0] for d in dup_pmid[:10]]},
        )

    def _check_drug_labels(self, session):
        """Check drug labels quality."""
        total = session.query(func.count(DrugLabel.set_id)).scalar()
        null_set_id = (
            session.query(func.count(DrugLabel.set_id)).filter(DrugLabel.set_id.is_(None)).scalar()
        )

        self._add_check(
            "drug_labels_not_null_set_id",
            total > 0 and null_set_id == 0,
            f"Total labels: {total}, null set_id: {null_set_id}",
            details={"total": total, "null_count": null_set_id},
        )

        # Duplicate set_ids
        dup_set = session.execute(
            select(DrugLabel.set_id, func.count())
            .group_by(DrugLabel.set_id)
            .having(func.count() > 1)
        ).all()
        self._add_check(
            "drug_labels_unique_set_id",
            len(dup_set) == 0,
            f"Duplicate set_ids: {len(dup_set)}",
            details={"duplicates": [d[0] for d in dup_set[:10]]},
        )

    def _check_chunks(self, session):
        """Check chunks table quality."""
        session.query(func.count(Chunk.id)).scalar()
        null_text = session.query(func.count(Chunk.id)).filter(Chunk.text.is_(None)).scalar()
        empty_text = session.query(func.count(Chunk.id)).filter(Chunk.text == "").scalar()
        null_embedding = (
            session.query(func.count(Chunk.id)).filter(Chunk.embedding.is_(None)).scalar()
        )

        self._add_check(
            "chunks_not_null_text",
            null_text == 0 and empty_text == 0,
            f"Chunks with null/empty text: {null_text + empty_text}",
            severity="error" if (null_text + empty_text) > 0 else "error",
            details={"null": null_text, "empty": empty_text},
        )
        self._add_check(
            "chunks_has_embedding",
            null_embedding == 0,
            f"Chunks missing embedding: {null_embedding}",
            severity="warn",
            details={"missing": null_embedding},
        )

        # Token count distribution
        token_stats = session.execute(
            select(
                func.min(Chunk.token_count),
                func.max(Chunk.token_count),
                func.avg(Chunk.token_count),
            )
        ).first()
        if token_stats:
            self.report.stats["chunk_token_stats"] = {
                "min": token_stats[0],
                "max": token_stats[1],
                "avg": round(token_stats[2], 1) if token_stats[2] else 0,
            }

    def _check_splits(self, session):
        """Check train/dev/test splits."""
        for split in SplitType:
            count = session.query(func.count(Chunk.id)).filter(Chunk.split == split).scalar()
            self.report.stats[f"split_{split.value}_count"] = count

        # Check no leakage: test chunks should not share doc_ids with train
        test_doc_ids = (
            session.query(Chunk.doc_id).filter(Chunk.split == SplitType.TEST).distinct().all()
        )
        test_doc_ids = {d[0] for d in test_doc_ids}
        train_doc_ids = (
            session.query(Chunk.doc_id).filter(Chunk.split == SplitType.TRAIN).distinct().all()
        )
        train_doc_ids = {d[0] for d in train_doc_ids}

        leakage = test_doc_ids & train_doc_ids
        self._add_check(
            "splits_no_leakage",
            len(leakage) == 0,
            f"Test doc_ids found in train: {len(leakage)}",
            severity="error",
            details={"leaked_doc_ids": list(leakage)[:10]},
        )

    def _check_referential_integrity(self, session):
        """Check foreign key relationships."""
        # Trial conditions -> trials
        orphan_conditions = session.execute(
            select(func.count(TrialCondition.id)).where(
                ~TrialCondition.nct_id.in_(select(Trial.nct_id))
            )
        ).scalar()
        self._add_check(
            "ref_integrity_trial_conditions",
            orphan_conditions == 0,
            f"Orphan trial conditions: {orphan_conditions}",
            details={"count": orphan_conditions},
        )

        # Trial interventions -> trials
        orphan_interventions = session.execute(
            select(func.count(TrialIntervention.id)).where(
                ~TrialIntervention.nct_id.in_(select(Trial.nct_id))
            )
        ).scalar()
        self._add_check(
            "ref_integrity_trial_interventions",
            orphan_interventions == 0,
            f"Orphan trial interventions: {orphan_interventions}",
            details={"count": orphan_interventions},
        )

        # Trial outcomes -> trials
        orphan_outcomes = session.execute(
            select(func.count(TrialOutcome.id)).where(
                ~TrialOutcome.nct_id.in_(select(Trial.nct_id))
            )
        ).scalar()
        self._add_check(
            "ref_integrity_trial_outcomes",
            orphan_outcomes == 0,
            f"Orphan trial outcomes: {orphan_outcomes}",
            details={"count": orphan_outcomes},
        )

    def _collect_stats(self, session):
        """Collect summary statistics."""
        self.report.stats["row_counts"] = {
            "trials": session.query(func.count(Trial.nct_id)).scalar(),
            "trial_conditions": session.query(func.count(TrialCondition.id)).scalar(),
            "trial_interventions": session.query(func.count(TrialIntervention.id)).scalar(),
            "trial_outcomes": session.query(func.count(TrialOutcome.id)).scalar(),
            "pubmed_articles": session.query(func.count(PubMedArticle.pmid)).scalar(),
            "drug_labels": session.query(func.count(DrugLabel.set_id)).scalar(),
            "chunks": session.query(func.count(Chunk.id)).scalar(),
        }

        # Completeness per column for trials
        trial_cols = [
            "phase",
            "overall_status",
            "start_date",
            "completion_date",
            "enrollment",
            "sponsor",
        ]
        completeness = {}
        total_trials = self.report.stats["row_counts"]["trials"]
        if total_trials > 0:
            for col in trial_cols:
                null_count = (
                    session.query(func.count(Trial.nct_id))
                    .filter(getattr(Trial, col).is_(None))
                    .scalar()
                )
                completeness[col] = {
                    "null_count": null_count,
                    "null_pct": round(null_count / total_trials * 100, 1),
                }
        self.report.stats["trial_completeness"] = completeness


def run_quality_checks(output_path: Path | None = None) -> QualityReport:
    """Run all quality checks and optionally write report."""
    checker = QualityChecker()
    report = checker.run_all_checks()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_markdown())
        logger.info("quality_report_written", path=str(output_path))

    return report
