"""Tests for the ClinicalTrials.gov API v2 ingestor."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioevidence.db.models import Trial, TrialCondition, TrialIntervention, TrialOutcome
from bioevidence.db.session import get_session
from bioevidence.ingest.ctgov import (
    CTGovClient,
    CTGovIngestor,
    parse_age,
    parse_date,
    parse_phase,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ctgov_study.json"


def _load_study() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _delete_trial(session: Session, nct_id: str) -> None:
    for model in (TrialCondition, TrialIntervention, TrialOutcome):
        session.query(model).filter(model.nct_id == nct_id).delete()
    session.query(Trial).filter(Trial.nct_id == nct_id).delete()
    session.commit()


class FakeCTGovClient:
    """Two-page fake API client for pagination tests."""

    def __init__(self, studies: list[dict[str, Any]]) -> None:
        self.studies = studies
        self.calls = 0

    def search_studies(self, **_: Any) -> dict[str, Any]:
        page = self.calls
        self.calls += 1
        if page >= len(self.studies):
            return {"studies": [], "nextPageToken": None}
        next_page = page + 1
        return {
            "studies": [self.studies[page]],
            "nextPageToken": f"token-{next_page}" if next_page < len(self.studies) else None,
        }

    def close(self) -> None:
        return None


def test_search_studies_uses_api_v2_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def json(self) -> dict[str, Any]:
            return {"studies": []}

    def fake_get(_self: CTGovClient, params: dict[str, Any]) -> FakeResponse:
        captured.update(params)
        return FakeResponse()

    monkeypatch.setattr(CTGovClient, "_get", fake_get)
    client = CTGovClient(rate_limit=1000)

    client.search_studies(
        condition="cancer",
        overall_status=["COMPLETED", "RECRUITING"],
        start_date_min="2005-01-01",
        page_size=5,
    )

    assert captured["filter.overallStatus"] == "COMPLETED,RECRUITING"
    assert captured["filter.advanced"] == (
        "AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[2005-01-01,MAX]"
    )
    assert "filter.startDate" not in captured
    assert "filter.studyType" not in captured
    client.close()


def test_parse_helpers() -> None:
    assert parse_date("2020-02") == (date(2020, 2, 1), True)
    assert parse_date("2020-02-29") == (date(2020, 2, 29), False)
    assert parse_age("18 Years") == 18.0
    assert parse_age("6 Months") == 0.5
    assert parse_phase(["PHASE2", "PHASE3"]) == "Phase 2/3"
    assert parse_phase("EARLY_PHASE1") == "Early Phase 1"


def test_parse_study_from_fixture() -> None:
    record = CTGovIngestor.parse_study(_load_study())

    assert record.nct_id == "NCT01234567"
    assert record.title.startswith("A Study of Pembrolizumab")
    assert record.brief_summary == (
        "This study evaluates pembrolizumab in advanced non-small cell lung cancer."
    )
    assert record.phase == "Phase 2/3"
    assert record.enrollment == 250
    assert record.start_date == date(2018, 4, 1)
    assert record.date_imputed is True
    assert record.countries == ["United States", "Canada"]
    assert len(record.conditions) == 2
    assert record.conditions[1].condition_mesh == "Lung Neoplasms"
    assert [item.name for item in record.interventions] == ["pembrolizumab", "carboplatin"]
    assert [item.kind for item in record.outcomes] == ["primary", "secondary", "secondary"]


def test_upsert_trial_is_idempotent() -> None:
    record = CTGovIngestor.parse_study(_load_study())
    with get_session() as session:
        _delete_trial(session, record.nct_id)
        first_trial, first_new = CTGovIngestor.upsert_trial(session, record)
        session.flush()
        counts = {
            "conditions": session.scalar(
                select(func.count())
                .select_from(TrialCondition)
                .where(TrialCondition.nct_id == record.nct_id)
            ),
            "interventions": session.scalar(
                select(func.count())
                .select_from(TrialIntervention)
                .where(TrialIntervention.nct_id == record.nct_id)
            ),
            "outcomes": session.scalar(
                select(func.count())
                .select_from(TrialOutcome)
                .where(TrialOutcome.nct_id == record.nct_id)
            ),
        }
        session.commit()

        second_trial, second_new = CTGovIngestor.upsert_trial(session, record)
        session.flush()
        second_counts = {
            "conditions": session.scalar(
                select(func.count())
                .select_from(TrialCondition)
                .where(TrialCondition.nct_id == record.nct_id)
            ),
            "interventions": session.scalar(
                select(func.count())
                .select_from(TrialIntervention)
                .where(TrialIntervention.nct_id == record.nct_id)
            ),
            "outcomes": session.scalar(
                select(func.count())
                .select_from(TrialOutcome)
                .where(TrialOutcome.nct_id == record.nct_id)
            ),
        }
        session.commit()

        assert first_new is True
        assert second_new is False
        assert first_trial.nct_id == second_trial.nct_id == record.nct_id
        assert second_counts == counts == {"conditions": 2, "interventions": 2, "outcomes": 3}
        _delete_trial(session, record.nct_id)


def test_ingest_pagination_and_cache(tmp_path: Path) -> None:
    base = _load_study()
    studies: list[dict[str, Any]] = []
    for suffix in ("01", "02"):
        study = json.loads(json.dumps(base))
        study["protocolSection"]["identificationModule"]["nctId"] = f"NCT99999{suffix}"
        studies.append(study)

    client = FakeCTGovClient(studies)
    ingestor = CTGovIngestor(cache_dir=tmp_path, client=client)  # type: ignore[arg-type]

    try:
        first = ingestor.ingest(
            conditions=["cancer"],
            statuses=["COMPLETED"],
            max_studies=2,
            page_size=1,
        )
        second = ingestor.ingest(
            conditions=["cancer"],
            statuses=["COMPLETED"],
            max_studies=2,
            page_size=1,
        )
    finally:
        for study in studies:
            with get_session() as session:
                _delete_trial(session, study["protocolSection"]["identificationModule"]["nctId"])

    assert client.calls == 2  # second run is fully cached
    assert first.fetched == 2
    assert first.inserted == 2
    assert second.inserted == 0
    assert second.updated == 2
    assert second.errors == 0
