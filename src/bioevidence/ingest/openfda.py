"""openFDA drug labels ingestor."""

import contextlib
import gzip
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from bioevidence.config import get_config
from bioevidence.db.models import DrugLabel
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


class OpenFDAClient:
    """Client for openFDA API."""

    BASE_URL = "https://api.fda.gov/"
    BULK_DOWNLOAD_URL = "https://download.open.fda.gov/drug/label/"

    def __init__(self, rate_limit: float = 10.0):
        self.rate_limit = rate_limit
        self._last_request = 0.0
        self.client = httpx.Client(timeout=120.0)

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
    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Make a GET request with retry logic."""
        self._rate_limit()
        url = urljoin(self.BASE_URL, endpoint)
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response

    def search_labels(
        self,
        search: str | None = None,
        limit: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        """Search drug labels."""
        params = {"limit": limit, "skip": skip}
        if search:
            params["search"] = search
        response = self._get("drug/label.json", params)
        return response.json()

    def get_bulk_files_list(self) -> list[str]:
        """Get list of bulk download files."""
        # This would typically be scraped from the bulk download page
        # For now, return known quarterly files pattern
        return [
            "drug-label-0001-of-0004.json.zip",
            "drug-label-0002-of-0004.json.zip",
            "drug-label-0003-of-0004.json.zip",
            "drug-label-0004-of-0004.json.zip",
        ]

    def download_bulk_file(self, filename: str, dest_path: Path) -> Path:
        """Download a bulk file."""
        self._rate_limit()
        url = urljoin(self.BULK_DOWNLOAD_URL, filename)
        with self.client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        return dest_path

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

    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached response if available."""
        path = self._cache_path(key)
        if path.exists():
            try:
                with gzip.open(path, "rt") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("cache_read_failed", key=key, error=str(e))
        return None

    def set(self, key: str, data: dict[str, Any]) -> None:
        """Cache a response."""
        path = self._cache_path(key)
        try:
            with gzip.open(path, "wt") as f:
                json.dump(data, f)
            logger.debug("cache_write", key=key)
        except Exception as e:
            logger.warning("cache_write_failed", key=key, error=str(e))


SECTIONS_TO_KEEP = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "clinical_studies",
    "boxed_warning",
    "description",
    "clinical_pharmacology",
    "nonclinical_toxicology",
    "use_in_specific_populations",
    "overdosage",
    "patient_counseling_information",
]


def extract_sections(label: dict) -> dict[str, str]:
    """Extract relevant sections from a drug label."""
    sections = {}
    openfda = label.get("openfda", {})

    # Extract structured sections
    for section in SECTIONS_TO_KEEP:
        if section in label:
            content = label[section]
            if isinstance(content, list):
                sections[section] = "\n\n".join(str(c) for c in content if c)
            elif isinstance(content, str):
                sections[section] = content
            elif isinstance(content, dict):
                sections[section] = json.dumps(content)

    # Extract brand and generic names
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])

    # Set ID
    set_id = label.get("set_id") or label.get("id")

    # Effective date
    effective_date_str = label.get("effective_time") or label.get("effective_date")
    effective_date = None
    if effective_date_str:
        with contextlib.suppress(ValueError, TypeError):
            effective_date = datetime.strptime(effective_date_str[:8], "%Y%m%d").date()

    return {
        "set_id": set_id,
        "brand_names": brand_names if brand_names else None,
        "generic_names": generic_names if generic_names else None,
        "sections": sections,
        "effective_date": effective_date,
        "source_url": f"https://api.fda.gov/drug/label.json?search=set_id:{set_id}"
        if set_id
        else None,
    }


def match_intervention_to_label(
    generic_names: list[str] | None,
    brand_names: list[str] | None,
    intervention_names: list[str],
) -> bool:
    """Check if a label matches any intervention name."""
    if not generic_names and not brand_names:
        return False

    all_label_names = set()
    if generic_names:
        all_label_names.update(n.lower() for n in generic_names)
    if brand_names:
        all_label_names.update(n.lower() for n in brand_names)

    for intervention in intervention_names:
        if intervention.lower() in all_label_names:
            return True
        # Check partial matches for compound names
        for label_name in all_label_names:
            if label_name in intervention.lower() or intervention.lower() in label_name:
                return True
    return False


class OpenFDAIngestor:
    """Ingestor for openFDA drug labels."""

    def __init__(self, cache_dir: Path | None = None, rate_limit: float = 10.0):
        config = get_config()
        self.client = OpenFDAClient(rate_limit=rate_limit)
        self.cache = RawCache(cache_dir or Path(config.llm.cache_dir).parent / "raw" / "openfda")
        self.stats = IngestStats()
        self._intervention_names: list[str] = []

    def load_intervention_names(self, session) -> list[str]:
        """Load intervention names from trials table for matching."""
        from bioevidence.db.models import TrialIntervention

        interventions = session.query(TrialIntervention.name).distinct().all()
        self._intervention_names = [i[0] for i in interventions if i[0]]
        logger.info("loaded_interventions", count=len(self._intervention_names))
        return self._intervention_names

    def _should_keep_label(self, label: dict) -> bool:
        """Determine if label should be kept based on intervention matching."""
        if not self._intervention_names:
            return True  # Keep all if no interventions loaded

        extracted = extract_sections(label)
        return match_intervention_to_label(
            extracted["generic_names"],
            extracted["brand_names"],
            self._intervention_names,
        )

    def ingest_from_bulk(
        self,
        bulk_dir: Path | None = None,
        download: bool = False,
    ) -> IngestStats:
        """Ingest from bulk download files."""
        if bulk_dir is None:
            config = get_config()
            bulk_dir = Path(config.llm.cache_dir).parent / "raw" / "openfda" / "bulk"

        bulk_dir.mkdir(parents=True, exist_ok=True)

        files = self.client.get_bulk_files_list()

        for filename in files:
            file_path = bulk_dir / filename
            bulk_dir / filename.replace(".zip", "")

            if not file_path.exists() and download:
                logger.info("downloading_bulk", file=filename)
                self.client.download_bulk_file(filename, file_path)

            if file_path.exists():
                logger.info("processing_bulk", file=filename)
                # Would need to unzip and process
                # For now, skip bulk processing
                pass

        return self.stats

    def ingest_from_api(
        self,
        search_terms: list[str] | None = None,
        max_labels: int | None = None,
        limit: int = 100,
    ) -> IngestStats:
        """Ingest from openFDA API with search."""
        # Load intervention names for matching
        with get_session() as session:
            self.load_intervention_names(session)

        if search_terms is None:
            # Build search from intervention names
            if self._intervention_names:
                # Search for labels matching intervention names (sample)
                search_terms = [
                    f'openfda.generic_name:"{name}"' for name in self._intervention_names[:50]
                ]
            else:
                search_terms = ["_exists_:openfda.brand_name"]

        for search_term in search_terms:
            logger.info("searching_labels", term=search_term)
            skip = 0

            while True:
                if max_labels and self.stats.fetched >= max_labels:
                    break

                cache_key = f"search_{search_term}_{skip}"
                data = self.cache.get(cache_key)

                if data is None:
                    data = self.client.search_labels(search=search_term, limit=limit, skip=skip)
                    self.cache.set(cache_key, data)

                results = data.get("results", [])
                if not results:
                    break

                for label in results:
                    try:
                        if not self._should_keep_label(label):
                            self.stats.skipped += 1
                            continue

                        self.stats.fetched += 1
                        extracted = extract_sections(label)

                        if not extracted["set_id"]:
                            continue

                        with get_session() as session:
                            existing = session.get(DrugLabel, extracted["set_id"])
                            if existing:
                                for key, value in extracted.items():
                                    if key != "set_id":
                                        setattr(existing, key, value)
                                self.stats.updated += 1
                            else:
                                drug_label = DrugLabel(**extracted)
                                session.add(drug_label)
                                self.stats.inserted += 1
                            session.commit()

                    except Exception:
                        self.stats.errors += 1
                        logger.exception("ingest_error", set_id=label.get("set_id"))

                skip += limit
                if len(results) < limit:
                    break

        self.client.close()
        return self.stats


def ingest_openfda(
    search_terms: list[str] | None = None,
    max_labels: int | None = None,
    limit: int = 100,
    rate_limit: float = 10.0,
    use_bulk: bool = False,
    bulk_dir: Path | None = None,
    download_bulk: bool = False,
) -> IngestStats:
    """Main entry point for openFDA ingestion."""
    config = get_config()
    cache_dir = Path(config.llm.cache_dir).parent / "raw" / "openfda"
    ingestor = OpenFDAIngestor(cache_dir=cache_dir, rate_limit=rate_limit)

    if use_bulk:
        return ingestor.ingest_from_bulk(bulk_dir, download_bulk)
    else:
        return ingestor.ingest_from_api(search_terms, max_labels, limit)
