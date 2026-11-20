"""Tests for the openFDA drug label ingestor."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from bioevidence.db.models import DrugLabel
from bioevidence.db.session import get_session
from bioevidence.ingest.openfda import (
    OpenFDAClient,
    OpenFDAIngestor,
    match_interventions,
    parse_effective_date,
    parse_label,
)
from bioevidence.quality.schemas import DrugLabelRecord

FIXTURE = Path(__file__).parent / "fixtures" / "openfda_label.json"
SET_ID = "openfda-test-001"


def _load_label() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _delete_label() -> None:
    with get_session() as session:
        session.query(DrugLabel).filter(DrugLabel.set_id == SET_ID).delete()
        session.commit()


class FakeOpenFDAClient:
    """One-page fake openFDA client that counts network calls."""

    def __init__(self, labels: list[dict[str, Any]]) -> None:
        self.labels = labels
        self.calls = 0

    def search_labels(self, search: str, limit: int, skip: int) -> dict[str, Any]:
        self.calls += 1
        if skip > 0 or "pembrolizumab" not in search:
            return {"results": [], "meta": {"results": {"total": 0}}}
        return {"results": self.labels, "meta": {"results": {"total": len(self.labels)}}}

    def close(self) -> None:
        return None


def test_search_labels_treats_404_as_empty_without_retry() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenFDAClient(rate_limit=1000, client=http_client)

    assert client.search_labels("missing-drug") == {
        "results": [],
        "meta": {"results": {"total": 0}},
    }
    assert len(calls) == 1


def test_build_search_terms_skips_unsafe_intervention_names() -> None:
    terms = OpenFDAIngestor.build_search_terms(
        ['"3+7"', "trasound", "pembrolizumab", "carboplatin", "placebo"],
        max_names=10,
    )

    assert terms == [
        'openfda.generic_name:"trasound"',
        'openfda.brand_name:"trasound"',
        'openfda.generic_name:"pembrolizumab"',
        'openfda.brand_name:"pembrolizumab"',
        'openfda.generic_name:"carboplatin"',
        'openfda.brand_name:"carboplatin"',
    ]


def test_match_interventions_is_conservative() -> None:
    assert match_interventions(["pembrolizumab"], ["KEYTRUDA"], ["pembrolizumab"]) == [
        "pembrolizumab"
    ]
    assert match_interventions(["sacubitril valsartan"], [], ["sacubitril/valsartan"]) == [
        "sacubitril/valsartan"
    ]
    assert match_interventions(["aspirin"], ["BAYER"], ["cancer"]) == []


def test_parse_label_from_fixture() -> None:
    record = parse_label(
        _load_label(),
        ["pembrolizumab"],
        "https://api.fda.gov/drug/label.json?search=set_id:openfda-test-001",
    )
    assert record is not None
    assert record.set_id == SET_ID
    assert record.brand_names == ["KEYTRUDA"]
    assert record.generic_names == ["pembrolizumab"]
    assert record.matched_interventions == ["pembrolizumab"]
    assert record.effective_date == date(2023, 1, 15)
    assert len(record.sections) == 7
    assert "metastatic non-small cell lung cancer" in record.sections["indications_and_usage"]
    assert parse_effective_date("202301") == date(2023, 1, 1)
    assert parse_effective_date("not-a-date") is None


def test_upsert_label_is_idempotent_and_keeps_newest() -> None:
    raw = _load_label()
    record = parse_label(raw, ["pembrolizumab"], "https://example.com/label")
    assert record is not None
    _delete_label()

    try:
        with get_session() as session:
            _, is_new, action = OpenFDAIngestor.upsert_label(session, record)
            assert is_new
            assert action == "inserted"
            _, is_new, action = OpenFDAIngestor.upsert_label(session, record)
            assert not is_new
            assert action == "updated"

            older = record.model_copy(update={"effective_date": date(2020, 1, 1)})
            _, is_new, action = OpenFDAIngestor.upsert_label(session, older)
            assert not is_new
            assert action == "kept"

            stored = session.get(DrugLabel, SET_ID)
            assert stored is not None
            assert stored.effective_date == date(2023, 1, 15)
            assert stored.matched_interventions == ["pembrolizumab"]
    finally:
        _delete_label()


def test_ingest_from_api_uses_cache_and_links_intervention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        OpenFDAIngestor,
        "load_intervention_names",
        staticmethod(lambda _session: ["pembrolizumab"]),
    )
    client = FakeOpenFDAClient([_load_label()])
    ingestor = OpenFDAIngestor(cache_dir=tmp_path, client=client)  # type: ignore[arg-type]
    _delete_label()

    try:
        first = ingestor.ingest_from_api(max_labels=1, limit=100)
        second = ingestor.ingest_from_api(max_labels=1, limit=100)
    finally:
        _delete_label()

    assert client.calls == 1
    assert first.fetched == 1
    assert first.inserted == 1
    assert second.inserted == 0
    assert second.updated == 1
    assert first.errors == second.errors == 0


def test_ingest_from_api_scans_catalog_when_targeted_search_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        OpenFDAIngestor,
        "load_intervention_names",
        staticmethod(lambda _session: ["pembrolizumab"]),
    )
    calls: list[tuple[str, int]] = []

    class CatalogFakeClient:
        def search_labels(self, search: str, limit: int, skip: int) -> dict[str, Any]:
            calls.append((search, skip))
            if "generic_name:*" in search and skip == 0:
                return {
                    "results": [_load_label()],
                    "meta": {"results": {"total": 1}},
                }
            return {"results": [], "meta": {"results": {"total": 0}}}

        def close(self) -> None:
            return None

    ingestor = OpenFDAIngestor(cache_dir=tmp_path, client=CatalogFakeClient())  # type: ignore[arg-type]
    _delete_label()
    try:
        stats = ingestor.ingest_from_api(
            search_terms=['openfda.generic_name:"does-not-exist"'],
            max_labels=1,
        )
    finally:
        _delete_label()

    assert stats.fetched == 1
    assert stats.inserted == 1
    assert any("generic_name:*" in search for search, _skip in calls)


def test_drug_label_record_includes_fetched_at() -> None:
    record = DrugLabelRecord(
        set_id="schema-test",
        sections={"indications_and_usage": "Test section."},
        source_url="https://example.com",
    )
    assert "fetched_at" in record.to_orm_kwargs()
