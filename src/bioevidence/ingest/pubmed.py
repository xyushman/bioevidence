"""PubMed E-utilities ingestor."""

import contextlib
import gzip
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from bioevidence.config import get_config
from bioevidence.db.models import PubMedArticle
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)


@dataclass
class IngestStats:
    """Statistics for an ingestion run."""

    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0


class PubMedClient:
    """Client for PubMed E-utilities API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str | None = None, rate_limit: float = 3.0):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._last_request = 0.0
        self.client = httpx.Client(timeout=60.0)

    def _rate_limit(self):
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _get(self, endpoint: str, params: dict[str, Any]) -> httpx.Response:
        """Make a GET request with retry logic."""
        self._rate_limit()
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response

    def esearch(
        self,
        term: str,
        db: str = "pubmed",
        retmax: int = 10000,
        use_history: bool = True,
        retstart: int = 0,
    ) -> dict[str, Any]:
        """Search PubMed and return query key + web environment for history server."""
        params = {
            "db": db,
            "term": term,
            "retmax": retmax,
            "retstart": retstart,
            "retmode": "json",
        }
        if use_history:
            params["usehistory"] = "y"

        response = self._get("esearch.fcgi", params)
        return response.json()

    def efetch(
        self,
        id_list: list[str] | None = None,
        query_key: str | None = None,
        web_env: str | None = None,
        db: str = "pubmed",
        retmode: str = "xml",
        retstart: int = 0,
        retmax: int = 200,
    ) -> str:
        """Fetch records in XML format."""
        params = {
            "db": db,
            "retmode": retmode,
            "retstart": retstart,
            "retmax": retmax,
        }
        if id_list:
            params["id"] = ",".join(id_list)
        if query_key and web_env:
            params["query_key"] = query_key
            params["WebEnv"] = web_env

        response = self._get("efetch.fcgi", params)
        return response.text

    def close(self):
        """Close the HTTP client."""
        self.client.close()


class RawCache:
    """Disk-based raw response cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_").replace("?", "_")
        return self.cache_dir / f"{safe_key}.json.gz"

    def get(self, key: str) -> dict[str, Any] | str | None:
        """Get cached response if available."""
        path = self._cache_path(key)
        if path.exists():
            try:
                with gzip.open(path, "rt") as f:
                    if path.suffix == ".gz":
                        content = f.read()
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            return content
            except Exception as e:
                logger.warning("cache_read_failed", key=key, error=str(e))
        return None

    def set(self, key: str, data: dict[str, Any] | str) -> None:
        """Cache a response."""
        path = self._cache_path(key)
        try:
            with gzip.open(path, "wt") as f:
                if isinstance(data, dict):
                    json.dump(data, f)
                else:
                    f.write(data)
            logger.debug("cache_write", key=key)
        except Exception as e:
            logger.warning("cache_write_failed", key=key, error=str(e))


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML into article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.exception("xml_parse_error")
        return articles

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                continue

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_text = ""
            abstract_elem = article_elem.find(".//Abstract/AbstractText")
            if abstract_elem is not None:
                abstract_text = abstract_elem.text or ""
            else:
                # Handle structured abstracts
                abstract_parts = []
                for abs_text in article_elem.findall(".//AbstractText"):
                    label = abs_text.get("Label")
                    text = abs_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract_text = " ".join(abstract_parts)

            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else None

            # Year
            year = None
            pub_date = article_elem.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        year = int(year_elem.text)
                else:
                    medline_date = pub_date.find("MedlineDate")
                    if medline_date is not None and medline_date.text:
                        with contextlib.suppress(ValueError, IndexError):
                            year = int(medline_date.text.split()[0])

            # MeSH terms
            mesh_terms = []
            for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            article = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text if abstract_text else None,
                "journal": journal,
                "year": year,
                "mesh_terms": mesh_terms if mesh_terms else None,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "fetched_at": datetime.utcnow(),
            }
            articles.append(article)

        except Exception as e:
            logger.warning("article_parse_error", error=str(e))
            continue

    return articles


class PubMedIngestor:
    """Ingestor for PubMed data."""

    def __init__(
        self, cache_dir: Path | None = None, api_key: str | None = None, rate_limit: float = 3.0
    ):
        config = get_config()
        self.client = PubMedClient(
            api_key=api_key or config.rate_limits.ncbi, rate_limit=rate_limit
        )
        self.cache = RawCache(cache_dir or Path(config.llm.cache_dir).parent / "raw" / "pubmed")
        self.stats = IngestStats()

    def _build_query(
        self, conditions: list[str], mesh_terms: list[str], start_year: int = 2005
    ) -> str:
        """Build PubMed query from conditions and MeSH terms."""
        parts = []

        # Add condition keywords
        if conditions:
            cond_query = " OR ".join(f'"{c}"[Title/Abstract]' for c in conditions)
            parts.append(f"({cond_query})")

        # Add MeSH terms
        if mesh_terms:
            mesh_query = " OR ".join(f'"{m}"[MeSH Terms]' for m in mesh_terms)
            parts.append(f"({mesh_query})")

        # Add year filter
        parts.append(f"({start_year}:3000[pdat])")

        # English only
        parts.append("english[lang]")

        return " AND ".join(parts)

    def ingest(
        self,
        conditions: list[str] | None = None,
        mesh_terms: list[str] | None = None,
        start_year: int = 2005,
        max_articles: int | None = None,
        batch_size: int = 200,
    ) -> IngestStats:
        """Run ingestion for given conditions."""
        config = get_config()
        scope_config = config.scope

        conditions = conditions or [
            kw for area in scope_config.therapeutic_areas for kw in area.get("keywords", [])
        ]
        mesh_terms = mesh_terms or [
            mt for area in scope_config.therapeutic_areas for mt in area.get("mesh_terms", [])
        ]

        query = self._build_query(conditions, mesh_terms, start_year)
        logger.info("pubmed_query", query=query)

        # Search
        cache_key = f"esearch_{hash(query)}"
        search_data = self.cache.get(cache_key)

        if search_data is None:
            search_data = self.client.esearch(term=query, retmax=max_articles or 100000)
            self.cache.set(cache_key, search_data)

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        total_count = int(search_data.get("esearchresult", {}).get("count", 0))
        logger.info("pubmed_search_results", total=total_count, returned=len(id_list))

        if max_articles:
            id_list = id_list[:max_articles]

        # Fetch in batches
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i : i + batch_size]
            logger.info("fetching_batch", start=i, size=len(batch_ids))

            fetch_cache_key = f"efetch_{'_'.join(batch_ids[:5])}"
            xml_data = self.cache.get(fetch_cache_key)

            if xml_data is None:
                xml_data = self.client.efetch(id_list=batch_ids)
                self.cache.set(fetch_cache_key, xml_data)

            articles = parse_pubmed_xml(xml_data)

            for article in articles:
                try:
                    self.stats.fetched += 1

                    with get_session() as session:
                        existing = session.get(PubMedArticle, article["pmid"])
                        if existing:
                            # Update
                            for key, value in article.items():
                                if key != "pmid":
                                    setattr(existing, key, value)
                            self.stats.updated += 1
                        else:
                            # Insert
                            pubmed_article = PubMedArticle(**article)
                            session.add(pubmed_article)
                            self.stats.inserted += 1
                        session.commit()

                except Exception:
                    self.stats.errors += 1
                    logger.exception("ingest_error", pmid=article.get("pmid"))

        self.client.close()
        return self.stats


def ingest_pubmed(
    conditions: list[str] | None = None,
    mesh_terms: list[str] | None = None,
    start_year: int = 2005,
    max_articles: int | None = None,
    batch_size: int = 200,
    rate_limit: float = 3.0,
    api_key: str | None = None,
) -> IngestStats:
    """Main entry point for PubMed ingestion."""
    config = get_config()
    cache_dir = Path(config.llm.cache_dir).parent / "raw" / "pubmed"
    ingestor = PubMedIngestor(cache_dir=cache_dir, api_key=api_key, rate_limit=rate_limit)
    return ingestor.ingest(conditions, mesh_terms, start_year, max_articles, batch_size)
