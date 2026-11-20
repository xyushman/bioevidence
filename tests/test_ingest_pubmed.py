"""Tests for the PubMed E-utilities ingestor."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from bioevidence.db.models import PubMedArticle
from bioevidence.db.session import get_session
from bioevidence.ingest.pubmed import (
    PubMedIngestor,
    parse_pubmed_xml,
    parse_pubmed_xml_with_rejections,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_articles.xml"


class FakePubMedClient:
    """Fake history-based PubMed client."""

    def __init__(self, xml: str) -> None:
        self.xml = xml
        self.search_calls = 0
        self.fetch_calls = 0

    def esearch(self, term: str, use_history: bool = True) -> dict[str, Any]:
        self.search_calls += 1
        return {
            "esearchresult": {
                "count": "2",
                "querykey": "1",
                "webenv": "fake-environment",
                "term": term,
            }
        }

    def efetch(self, **_: Any) -> str:
        self.fetch_calls += 1
        return self.xml

    def close(self) -> None:
        return None


def test_build_query_contains_scope_and_english() -> None:
    query = PubMedIngestor.build_query(["cancer", "heart failure"], ["Neoplasms"], 2010)
    assert '"cancer"[Title/Abstract]' in query
    assert '"heart failure"[Title/Abstract]' in query
    assert '"Neoplasms"[MeSH Terms]' in query
    assert "2010:3000[pdat]" in query
    assert "english[lang]" in query


def test_build_query_for_range_adds_publication_window() -> None:
    query = PubMedIngestor.build_query_for_range(
        ["cancer"], ["Neoplasms"], date(2026, 1, 1), date(2026, 1, 2)
    )

    assert '"cancer"[Title/Abstract]' in query
    assert '"Neoplasms"[MeSH Terms]' in query
    assert '"2026-01-01"[Date - Publication] : "2026-01-02"[Date - Publication]' in query
    assert "english[lang]" in query


def test_iter_date_ranges_is_newest_first_and_deterministic() -> None:
    ranges = list(PubMedIngestor.iter_date_ranges(2025, date(2025, 1, 3)))

    assert ranges == [
        (date(2025, 1, 3), date(2025, 1, 3)),
        (date(2025, 1, 2), date(2025, 1, 2)),
        (date(2025, 1, 1), date(2025, 1, 1)),
    ]


def test_parse_pubmed_xml_from_fixture() -> None:
    records = parse_pubmed_xml(FIXTURE.read_text(encoding="utf-8"))

    assert len(records) == 2  # record without abstract is skipped
    assert records[0].pmid == "30000001"
    assert records[0].title == "Efficacy of pembrolizumab in advanced lung cancer"
    assert records[0].year == 2021
    assert records[0].mesh_terms == ["Lung Neoplasms", "Immunotherapy"]
    assert "BACKGROUND: Advanced lung cancer" in records[0].abstract
    assert records[1].pmid == "30000002"
    assert records[1].year == 2019
    assert records[1].journal == "Annals of Rheumatology"


def test_parse_reports_rejections_and_abstract_less_records() -> None:
    bad_article = """
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">30000004</PMID>
      <Article>
        <Journal><Title>Bad Year Journal</Title><JournalIssue><PubDate><Year>1800</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Invalid publication year</ArticleTitle>
        <Abstract><AbstractText>This abstract is otherwise valid.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
"""
    xml = FIXTURE.read_text(encoding="utf-8").replace(
        "</PubmedArticleSet>", f"{bad_article}\n</PubmedArticleSet>"
    )
    records, rejected, skipped = parse_pubmed_xml_with_rejections(xml)

    assert len(records) == 2
    assert skipped == 1
    assert len(rejected) == 1
    assert rejected[0]["record"]["pmid"] == "30000004"
    assert "year" in rejected[0]["reason"]


def test_ingest_uses_history_cache_and_is_idempotent(tmp_path: Path) -> None:
    client = FakePubMedClient(FIXTURE.read_text(encoding="utf-8"))
    ingestor = PubMedIngestor(
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    with get_session() as session:
        session.query(PubMedArticle).filter(
            PubMedArticle.pmid.in_(["30000001", "30000002"])
        ).delete()
        session.commit()

    try:
        first = ingestor.ingest(max_articles=2, batch_size=200)
        second = ingestor.ingest(max_articles=2, batch_size=200)
    finally:
        with get_session() as session:
            session.query(PubMedArticle).filter(
                PubMedArticle.pmid.in_(["30000001", "30000002"])
            ).delete()
            session.commit()

    assert client.search_calls == 1
    assert client.fetch_calls == 1
    assert first.fetched == 2
    assert first.inserted == 2
    assert second.inserted == 0
    assert second.updated == 2
