"""PubMed E-utilities ingestor."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from bioevidence.config import Config, get_config
from bioevidence.db.models import PubMedArticle
from bioevidence.db.session import get_session
from bioevidence.ingest.common import IngestStats, RateLimiter, RawCache, clean_text, utcnow
from bioevidence.ingest.runs import record_ingest_run
from bioevidence.quality.quarantine import quarantine_record
from bioevidence.quality.schemas import PubMedRecord

logger = structlog.get_logger(__name__)

# E-utilities only exposes the first 9,999 matches for a PubMed query, and efetch
# rejects retstart values that would exceed that window.
MAX_SEARCH_WINDOW = 9_999

__all__ = [
    "IngestStats",
    "PubMedClient",
    "PubMedIngestor",
    "ingest_pubmed",
    "parse_pubmed_xml",
    "parse_pubmed_xml_with_rejections",
]


class PubMedClient:
    """Rate-limited client for NCBI E-utilities."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: float = 3.0,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.limiter = RateLimiter(rate_limit)
        self.client = httpx.Client(timeout=timeout)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _get(self, endpoint: str, params: dict[str, Any]) -> httpx.Response:
        self.limiter.wait()
        payload = {**params, "tool": "BioEvidence"}
        if self.api_key:
            payload["api_key"] = self.api_key
        response = self.client.get(f"{self.BASE_URL}/{endpoint}", params=payload)
        response.raise_for_status()
        return response

    def esearch(self, term: str, use_history: bool = True) -> dict[str, Any]:
        """Search PubMed and optionally keep the result set on NCBI's history server."""
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmax": 0 if use_history else 10_000,
            "retmode": "json",
        }
        if use_history:
            params["usehistory"] = "y"
        payload = self._get("esearch.fcgi", params).json()
        return payload if isinstance(payload, dict) else {}

    def efetch(
        self,
        query_key: str | None = None,
        web_env: str | None = None,
        id_list: list[str] | None = None,
        retstart: int = 0,
        retmax: int = 200,
    ) -> str:
        """Fetch one XML batch from a history set or an explicit ID list."""
        params: dict[str, Any] = {
            "db": "pubmed",
            "retmode": "xml",
            "retstart": retstart,
            "retmax": retmax,
        }
        if query_key and web_env:
            params.update({"query_key": query_key, "WebEnv": web_env})
        elif id_list:
            params["id"] = ",".join(id_list)
        else:
            msg = "efetch requires history keys or an ID list"
            raise ValueError(msg)
        return self._get("efetch.fcgi", params).text

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()


class RawPubMedCache(RawCache):
    """Raw cache specialized for PubMed responses."""

    def get_json(self, key: str) -> dict[str, Any] | None:
        """Return a cached JSON payload."""
        value = self.get(key)
        return value if isinstance(value, dict) else None

    def get_text(self, key: str) -> str | None:
        """Return a cached XML payload."""
        value = self.get(key)
        return value if isinstance(value, str) else None


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return "".join(element.itertext()).strip() or None


def _article_payload(article_elem: ET.Element) -> dict[str, Any]:
    """Return raw fields for one PubMed article element."""
    pmid = _element_text(article_elem.find(".//PMID"))
    abstract_parts: list[str] = []
    for abs_text in article_elem.findall(".//Abstract/AbstractText"):
        text = _element_text(abs_text)
        if not text:
            continue
        label = abs_text.get("Label") or abs_text.get("NlmCategory")
        abstract_parts.append(f"{label}: {text}" if label else text)

    year: int | None = None
    pub_date = article_elem.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year_text = _element_text(pub_date.find("Year"))
        if year_text is None:
            medline_date = _element_text(pub_date.find("MedlineDate"))
            year_text = medline_date.split()[0] if medline_date else None
        if year_text and year_text[:4].isdigit():
            year = int(year_text[:4])

    return {
        "pmid": clean_text(pmid) if pmid else None,
        "title": _element_text(article_elem.find(".//ArticleTitle")),
        "abstract": "\n".join(abstract_parts).strip() or None,
        "journal": _element_text(article_elem.find(".//Journal/Title")),
        "year": year,
        "mesh_terms": [
            term.text.strip()
            for term in article_elem.findall(".//MeshHeading/DescriptorName")
            if term.text and term.text.strip()
        ]
        or None,
    }


def parse_pubmed_xml_with_rejections(
    xml_text: str,
) -> tuple[list[PubMedRecord], list[dict[str, Any]], int]:
    """Parse PubMed XML, returning valid records, rejections, and abstract-less count."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        msg = "PubMed response was not valid XML"
        raise ValueError(msg) from exc

    records: list[PubMedRecord] = []
    rejected: list[dict[str, Any]] = []
    skipped = 0
    for article_elem in root.findall(".//PubmedArticle"):
        payload = _article_payload(article_elem)
        if not payload["pmid"]:
            rejected.append({"reason": "missing PMID", "record": payload})
            continue
        if not payload["abstract"]:
            skipped += 1
            logger.debug("pubmed_without_abstract_skipped", pmid=payload["pmid"])
            continue

        payload.update(
            {
                "title": payload["title"] or f"PubMed {payload['pmid']}",
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{payload['pmid']}/",
                "fetched_at": utcnow(),
            }
        )
        try:
            records.append(PubMedRecord(**payload))
        except ValidationError as exc:
            rejected.append({"reason": str(exc), "record": payload})
            logger.warning("pubmed_record_rejected", pmid=payload["pmid"], error=str(exc))
    return records, rejected, skipped


def parse_pubmed_xml(xml_text: str) -> list[PubMedRecord]:
    """Parse PubMed XML into validated records with abstracts only."""
    records, _, _ = parse_pubmed_xml_with_rejections(xml_text)
    return records


class PubMedIngestor:
    """Parse and upsert PubMed abstracts."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        rate_limit: float = 3.0,
        config: Config | None = None,
        client: PubMedClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.client = client or PubMedClient(api_key=api_key, rate_limit=rate_limit)
        raw_root = Path(self.config.llm.cache_dir).parent / "raw" / "pubmed"
        self.cache = RawPubMedCache(cache_dir or raw_root)
        self.stats = IngestStats()

    @staticmethod
    def _scope_keywords(config: Config) -> list[str]:
        keywords: list[str] = []
        for area in config.scope.therapeutic_areas:
            keywords.extend(area.keywords)
        return list(dict.fromkeys(keywords))

    @staticmethod
    def _scope_mesh(config: Config) -> list[str]:
        terms: list[str] = []
        for area in config.scope.therapeutic_areas:
            terms.extend(area.mesh_terms)
        return list(dict.fromkeys(terms))

    @staticmethod
    def _scope_query(conditions: list[str], mesh_terms: list[str]) -> list[str]:
        parts: list[str] = []
        if conditions:
            condition_query = " OR ".join(f'"{term}"[Title/Abstract]' for term in conditions)
            parts.append(f"({condition_query})")
        if mesh_terms:
            mesh_query = " OR ".join(f'"{term}"[MeSH Terms]' for term in mesh_terms)
            parts.append(f"({mesh_query})")
        return parts

    @staticmethod
    def build_query(conditions: list[str], mesh_terms: list[str], start_year: int = 2005) -> str:
        """Build a legacy unbounded query, kept for compatibility and tests."""
        parts = PubMedIngestor._scope_query(conditions, mesh_terms)
        parts.append(f"({start_year}:3000[pdat])")
        parts.append("english[lang]")
        return " AND ".join(parts)

    @staticmethod
    def build_query_for_range(
        conditions: list[str],
        mesh_terms: list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> str:
        """Build a PubMed query for a bounded publication-date window."""
        parts = PubMedIngestor._scope_query(conditions, mesh_terms)
        start = start_date.isoformat() if isinstance(start_date, date) else start_date
        end = end_date.isoformat() if isinstance(end_date, date) else end_date
        parts.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
        parts.append("english[lang]")
        return " AND ".join(parts)

    @staticmethod
    def iter_date_ranges(
        start_year: int, end_date: date | None = None
    ) -> Iterator[tuple[date, date]]:
        """Yield daily date windows, newest first, within the configured scope."""
        current = end_date or datetime.now(UTC).date()
        last = date(start_year, 1, 1)
        while current >= last:
            yield current, current
            current -= timedelta(days=1)

    @staticmethod
    def upsert_article(session: Session, record: PubMedRecord) -> tuple[PubMedArticle, bool]:
        """Upsert one PubMed article. Returns ``(article, is_new)``."""
        article = session.get(PubMedArticle, record.pmid)
        is_new = article is None
        values = record.to_orm_kwargs()
        if article is None:
            article = PubMedArticle(**values)
            session.add(article)
        else:
            for key, value in values.items():
                if key != "pmid":
                    setattr(article, key, value)
        return article, is_new

    def _search_key(self, query: str) -> str:
        return f"pubmed:esearch:{query}"

    def _fetch_key(self, query: str, retstart: int, batch_size: int) -> str:
        return f"pubmed:efetch:{query}:{retstart}:{batch_size}"

    def ingest(
        self,
        conditions: list[str] | None = None,
        mesh_terms: list[str] | None = None,
        start_year: int = 2005,
        max_articles: int | None = None,
        batch_size: int = 200,
    ) -> IngestStats:
        """Ingest abstracts via esearch history and batched efetch."""
        self.stats = IngestStats()
        targets = self.config.scope.targets
        target = (
            max_articles
            or targets.get("pubmed_articles")
            or targets.get("pubmed_abstracts")
            or 80_000
        )
        conditions = conditions or self._scope_keywords(self.config)
        mesh_terms = mesh_terms or self._scope_mesh(self.config)
        logger.info("pubmed_target", target=target, start_year=start_year)

        seen: set[str] = set()
        try:
            for window_start, window_end in self.iter_date_ranges(start_year):
                if len(seen) >= target:
                    break
                query = self.build_query_for_range(conditions, mesh_terms, window_start, window_end)
                search_key = self._search_key(query)
                search_data = self.cache.get_json(search_key)
                if search_data is None:
                    search_data = self.client.esearch(term=query, use_history=True)
                    self.cache.set(search_key, search_data)

                result = search_data.get("esearchresult", {}) if search_data else {}
                if error := result.get("ERROR"):
                    self.stats.errors += 1
                    logger.warning("pubmed_search_failed", date=str(window_start), error=str(error))
                    continue
                count = int(result.get("count", 0) or 0)
                if count <= 0:
                    continue
                query_key = result.get("querykey")
                web_env = result.get("webenv")
                if not query_key or not web_env:
                    self.stats.errors += 1
                    logger.warning("pubmed_missing_history_keys", date=str(window_start))
                    continue

                available = min(count, MAX_SEARCH_WINDOW)
                for retstart in range(0, available, batch_size):
                    if len(seen) >= target:
                        break
                    retmax = min(batch_size, MAX_SEARCH_WINDOW - retstart)
                    fetch_key = self._fetch_key(query, retstart, batch_size)
                    xml_data = self.cache.get_text(fetch_key)
                    if xml_data is None:
                        xml_data = self.client.efetch(
                            query_key=str(query_key),
                            web_env=str(web_env),
                            retstart=retstart,
                            retmax=retmax,
                        )
                        self.cache.set(fetch_key, xml_data)

                    try:
                        records, rejected, skipped = parse_pubmed_xml_with_rejections(xml_data)
                    except ValueError as exc:
                        self.stats.errors += 1
                        logger.warning(
                            "pubmed_batch_parse_failed", retstart=retstart, error=str(exc)
                        )
                        continue

                    self.stats.skipped += skipped
                    for item in rejected:
                        self.stats.skipped += 1
                        quarantine_record("pubmed", item["record"], str(item["reason"]))

                    with get_session() as session:
                        for record in records:
                            if record.pmid in seen or len(seen) >= target:
                                continue
                            seen.add(record.pmid)
                            _, is_new = self.upsert_article(session, record)
                            if is_new:
                                self.stats.inserted += 1
                            else:
                                self.stats.updated += 1
                        session.commit()

                logger.info(
                    "pubmed_window_complete",
                    date=str(window_start),
                    matches=count,
                    fetched=len(seen),
                )
        finally:
            self.client.close()

        self.stats.fetched = len(seen)
        with get_session() as session:
            row_count = int(session.scalar(select(func.count()).select_from(PubMedArticle)) or 0)
            record_ingest_run("pubmed", self.stats, row_count)
        logger.info("pubmed_ingest_complete", **self.stats.as_dict())
        return self.stats


def ingest_pubmed(
    conditions: list[str] | None = None,
    mesh_terms: list[str] | None = None,
    start_year: int = 2005,
    max_articles: int | None = None,
    batch_size: int = 200,
    rate_limit: float | None = None,
    api_key: str | None = None,
) -> IngestStats:
    """Main entry point for PubMed ingestion."""
    config = get_config()
    resolved_key = api_key or config.rate_limits.ncbi_api_key or None
    effective_rate = rate_limit or (10.0 if resolved_key else float(config.rate_limits.ncbi))
    raw_root = Path(config.llm.cache_dir).parent / "raw" / "pubmed"
    ingestor = PubMedIngestor(
        cache_dir=raw_root,
        api_key=resolved_key,
        rate_limit=effective_rate,
        config=config,
    )
    return ingestor.ingest(
        conditions=conditions,
        mesh_terms=mesh_terms,
        start_year=start_year,
        max_articles=max_articles,
        batch_size=batch_size,
    )
