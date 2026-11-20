"""openFDA drug label ingestion with intervention linkage."""

from __future__ import annotations

import gzip
import json
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import structlog
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tenacity import RetryError, retry, retry_if_exception, stop_after_attempt, wait_exponential

from bioevidence.config import Config, get_config
from bioevidence.db.models import DrugLabel, TrialIntervention
from bioevidence.db.session import get_session
from bioevidence.ingest.common import (
    IngestStats,
    RateLimiter,
    RawCache,
    clean_text,
    utcnow,
)
from bioevidence.ingest.runs import record_ingest_run
from bioevidence.quality.quarantine import quarantine_record
from bioevidence.quality.schemas import DrugLabelRecord

logger = structlog.get_logger(__name__)


def _retry_openfda_request(exc: BaseException) -> bool:
    """Retry transient failures but treat a 404 as a normal empty result."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code != 404
    return True


def _http_status(exc: BaseException) -> int | None:
    """Extract an HTTP status from a direct or tenacity-wrapped HTTP error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, RetryError):
        last_error = exc.last_attempt.exception()
        if isinstance(last_error, httpx.HTTPStatusError):
            return last_error.response.status_code
    return None


__all__ = [
    "IngestStats",
    "OpenFDAClient",
    "OpenFDAIngestor",
    "ingest_openfda",
    "match_interventions",
    "parse_effective_date",
    "parse_label",
]

SECTIONS_TO_KEEP = (
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "clinical_studies",
)

NON_DRUG_INTERVENTIONS = {
    "placebo",
    "standard of care",
    "no intervention",
    "usual care",
    "surgery",
    "radiotherapy",
    "radiation therapy",
    "chemotherapy",
    "behavioral therapy",
    "exercise",
    "diet",
    "observation",
    "active comparator",
    "sham",
    "device",
    "procedure",
}

_MAX_TERM_SKIP = 10_000
_MAX_CATALOG_SKIP = 60_000
_CATALOG_SEARCH = "openfda.generic_name:*"


def normalize_name(value: str) -> str:
    """Normalize a drug name for conservative exact/token matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def match_interventions(
    generic_names: list[str] | None,
    brand_names: list[str] | None,
    intervention_names: list[str],
) -> list[str]:
    """Return intervention names whose normalized tokens match a label name."""
    label_names = {
        normalized
        for name in (*(generic_names or []), *(brand_names or []))
        if (normalized := normalize_name(str(name)))
    }
    matched: list[str] = []
    for intervention in intervention_names:
        key = normalize_name(intervention)
        if not key or len(key) < 3:
            continue
        key_tokens = set(key.split())
        exact_or_subset = any(
            key == label_name or key_tokens <= set(label_name.split()) <= key_tokens
            for label_name in label_names
        )
        if exact_or_subset:
            matched.append(intervention)
    return sorted(set(matched))


def parse_effective_date(value: Any) -> date | None:
    """Parse openFDA ``YYYYMMDD``/``YYYYMM`` effective dates."""
    if not value:
        return None
    text = str(value).strip()
    try:
        if len(text) == 8:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        if len(text) == 6:
            return date(int(text[:4]), int(text[4:6]), 1)
    except ValueError:
        return None
    return None


def _section_text(value: Any) -> str | None:
    """Flatten an openFDA section into clean text."""
    if isinstance(value, list):
        parts = [clean_text(str(item)) for item in value if item]
    else:
        parts = [clean_text(str(value))] if value else []
    useful = [part for part in parts if part]
    return "\n\n".join(useful) or None


def parse_label(
    label: dict[str, Any],
    intervention_names: list[str],
    source_url: str,
) -> DrugLabelRecord | None:
    """Validate a raw openFDA label and link it to trial intervention names."""
    raw_openfda = label.get("openfda")
    openfda: dict[str, Any] = dict(raw_openfda) if isinstance(raw_openfda, dict) else {}
    brand_names = [
        cleaned
        for name in openfda.get("brand_name", []) or []
        if (cleaned := clean_text(str(name)))
    ]
    generic_names = [
        cleaned
        for name in openfda.get("generic_name", []) or []
        if (cleaned := clean_text(str(name)))
    ]
    matched = match_interventions(generic_names, brand_names, intervention_names)
    if not matched:
        return None

    sections: dict[str, str] = {}
    for section in SECTIONS_TO_KEEP:
        text = _section_text(label.get(section))
        if text:
            sections[section] = text
    if not sections:
        return None

    set_id = clean_text(str(label.get("set_id") or label.get("id") or ""))
    if not set_id:
        return None

    return DrugLabelRecord(
        set_id=set_id,
        brand_names=brand_names or None,
        generic_names=generic_names or None,
        sections=sections,
        effective_date=parse_effective_date(label.get("effective_time")),
        matched_interventions=matched,
        source_url=source_url,
        fetched_at=utcnow(),
    )


class OpenFDAClient:
    """Rate-limited client for the official openFDA label API."""

    BASE_URL = "https://api.fda.gov/drug/label.json"

    def __init__(
        self,
        rate_limit: float = 10.0,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.limiter = RateLimiter(rate_limit)
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_retry_openfda_request),
    )
    def _get(self, params: dict[str, Any]) -> httpx.Response:
        self.limiter.wait()
        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response

    def search_labels(self, search: str, limit: int = 100, skip: int = 0) -> dict[str, Any]:
        """Fetch one page; an empty result is represented as ``{"results": []}``."""
        try:
            response = self._get({"search": search, "limit": limit, "skip": skip})
        except (httpx.HTTPStatusError, RetryError) as exc:
            status = _http_status(exc)
            if status != 404:
                logger.warning("openfda_search_http_error", search=search, status=status)
            return {"results": [], "meta": {"results": {"total": 0}}}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def close(self) -> None:
        """Close the underlying HTTP client when owned here."""
        if self._owns_client:
            self.client.close()


class OpenFDAIngestor:
    """Parse, validate, and upsert openFDA drug labels."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: float = 10.0,
        config: Config | None = None,
        client: OpenFDAClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.client = client or OpenFDAClient(rate_limit=rate_limit)
        raw_root = Path(self.config.llm.cache_dir).parent / "raw" / "openfda"
        self.cache = RawCache(cache_dir or raw_root)
        self.stats = IngestStats()
        self._intervention_names: list[str] = []

    @staticmethod
    def load_intervention_names(session: Session) -> list[str]:
        """Load distinct intervention names, most frequently used first."""
        rows = session.execute(
            select(TrialIntervention.name, func.count())
            .group_by(TrialIntervention.name)
            .order_by(func.count().desc(), TrialIntervention.name)
        ).all()
        names: list[str] = []
        seen: set[str] = set()
        for raw_name, _count in rows:
            name = str(raw_name).strip()
            key = normalize_name(name)
            if name and key and key not in seen:
                names.append(name)
                seen.add(key)
        return names

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _is_searchable_drug_name(name: str) -> bool:
        """Return whether an intervention name is safe for an openFDA search."""
        value = name.strip().strip("\"'").strip()
        if not value or value.lower() in NON_DRUG_INTERVENTIONS:
            return False
        if any(char in value for char in ('"', "'", "\\", "+", "(", ")", ",", ";", "/")):
            return False
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]{2,60}", value))

    @staticmethod
    def build_search_terms(intervention_names: list[str], max_names: int = 250) -> list[str]:
        """Build generic/brand name searches for candidate interventions."""
        candidates = [
            name.strip().strip("\"'").strip()
            for name in intervention_names
            if OpenFDAIngestor._is_searchable_drug_name(name) and len(normalize_name(name)) >= 3
        ]
        terms: list[str] = []
        for name in list(dict.fromkeys(candidates))[:max_names]:
            escaped = OpenFDAIngestor._escape_query_value(name)
            terms.append(f'openfda.generic_name:"{escaped}"')
            terms.append(f'openfda.brand_name:"{escaped}"')
        return list(dict.fromkeys(terms))

    @staticmethod
    def _row_count(session: Session) -> int:
        return int(session.scalar(select(func.count()).select_from(DrugLabel)) or 0)

    @staticmethod
    def upsert_label(session: Session, record: DrugLabelRecord) -> tuple[DrugLabel, bool, str]:
        """Upsert a label, keeping the newest effective version. Returns action."""
        existing = session.get(DrugLabel, record.set_id)
        if existing is None:
            label = DrugLabel(**record.to_orm_kwargs())
            session.add(label)
            return label, True, "inserted"

        if (
            existing.effective_date
            and record.effective_date
            and record.effective_date < existing.effective_date
        ):
            return existing, False, "kept"

        values = record.to_orm_kwargs()
        for key, value in values.items():
            if key != "set_id":
                setattr(existing, key, value)
        return existing, False, "updated"

    def _process_labels(
        self,
        session: Session,
        labels: list[dict[str, Any]],
        source_url_for_label: Any,
        seen: set[str],
        target: int,
    ) -> None:
        for raw in labels:
            if len(seen) >= target:
                return
            try:
                record = parse_label(
                    raw,
                    self._intervention_names,
                    source_url_for_label(raw),
                )
            except ValidationError as exc:
                self.stats.errors += 1
                quarantine_record("openfda", raw, str(exc))
                continue
            except Exception as exc:
                self.stats.errors += 1
                quarantine_record("openfda", raw, str(exc))
                continue

            if record is None:
                self.stats.skipped += 1
                continue
            if record.set_id in seen:
                continue
            seen.add(record.set_id)
            self.stats.fetched += 1
            _, _, action = self.upsert_label(session, record)
            if action == "inserted":
                self.stats.inserted += 1
            elif action == "updated":
                self.stats.updated += 1
            else:
                self.stats.skipped += 1

    def _scan_catalog(self, seen: set[str], target: int, page_size: int) -> None:
        """Scan the official label catalog when targeted searches fall short."""
        skip = 0
        scanned = 0
        while skip < _MAX_CATALOG_SKIP and len(seen) < target:
            key = f"openfda:catalog:{_CATALOG_SEARCH}:{page_size}:{skip}"
            data = self.cache.get(key)
            if data is None:
                data = self.client.search_labels(_CATALOG_SEARCH, page_size, skip)
                self.cache.set(key, data)
            labels = data.get("results", []) if isinstance(data, dict) else []
            if not labels:
                break
            scanned += len(labels)
            with get_session() as session:
                self._process_labels(
                    session,
                    [label for label in labels if isinstance(label, dict)],
                    lambda raw: (
                        "https://api.fda.gov/drug/label.json"
                        f"?search=set_id:{raw.get('set_id') or raw.get('id') or ''}"
                    ),
                    seen,
                    target,
                )
            skip += page_size
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            total = int(meta.get("results", {}).get("total", 0))
            if total and skip >= total:
                break
        logger.info("openfda_catalog_scan_complete", scanned=scanned, matched=len(seen))

    def ingest_from_api(
        self,
        search_terms: list[str] | None = None,
        max_labels: int | None = None,
        limit: int = 100,
        max_terms: int = 250,
    ) -> IngestStats:
        """Ingest labels via targeted searches, then a bounded catalog scan."""
        self.stats = IngestStats()
        target = max_labels or int(self.config.scope.targets.get("drug_labels", 3_000))
        with get_session() as session:
            self._intervention_names = self.load_intervention_names(session)
        if not self._intervention_names:
            msg = "No trial interventions found; ingest CT.gov studies before openFDA labels"
            raise RuntimeError(msg)

        terms = search_terms or self.build_search_terms(self._intervention_names, max_terms)
        page_size = min(max(1, limit), 1_000)
        seen: set[str] = set()
        try:
            for search in terms:
                if len(seen) >= target:
                    break
                skip = 0
                while skip < _MAX_TERM_SKIP and len(seen) < target:
                    key = f"openfda:search:{search}:{page_size}:{skip}"
                    data = self.cache.get(key)
                    if data is None:
                        data = self.client.search_labels(search, page_size, skip)
                        self.cache.set(key, data)
                    labels = data.get("results", []) if isinstance(data, dict) else []
                    if not labels:
                        break

                    with get_session() as session:
                        self._process_labels(
                            session,
                            [label for label in labels if isinstance(label, dict)],
                            lambda raw: (
                                "https://api.fda.gov/drug/label.json"
                                f"?search=set_id:{raw.get('set_id') or raw.get('id') or ''}"
                            ),
                            seen,
                            target,
                        )

                    skip += page_size
                    meta = data.get("meta", {}) if isinstance(data, dict) else {}
                    total = int(meta.get("results", {}).get("total", 0))
                    if len(labels) < page_size or (total and skip >= total):
                        break
                logger.info("openfda_search_complete", search=search, matched=len(seen))

            if len(seen) < target:
                self._scan_catalog(seen, target, page_size)
        finally:
            self.client.close()

        with get_session() as session:
            record_ingest_run("openfda", self.stats, self._row_count(session))
        logger.info("openfda_ingest_complete", **self.stats.as_dict())
        return self.stats

    def ingest_from_bulk(
        self,
        bulk_dir: Path | None = None,
        max_labels: int | None = None,
    ) -> IngestStats:
        """Ingest labels from locally available official bulk JSON files."""
        self.stats = IngestStats()
        target = max_labels or int(self.config.scope.targets.get("drug_labels", 3_000))
        directory = bulk_dir or Path(self.config.llm.cache_dir).parent / "raw" / "openfda" / "bulk"
        if not directory.exists():
            logger.warning("openfda_bulk_dir_missing", path=str(directory))
            return self.stats

        with get_session() as session:
            self._intervention_names = self.load_intervention_names(session)

        seen: set[str] = set()
        for path in sorted(directory.glob("*.json*")):
            if len(seen) >= target:
                break
            for labels in _iter_bulk_labels(path):
                with get_session() as session:
                    self._process_labels(
                        session,
                        labels,
                        lambda raw: (
                            "https://download.open.fda.gov/drug/label/"
                            f"{raw.get('set_id') or raw.get('id') or ''}"
                        ),
                        seen,
                        target,
                    )

        with get_session() as session:
            record_ingest_run("openfda_bulk", self.stats, self._row_count(session))
        logger.info("openfda_bulk_complete", **self.stats.as_dict())
        return self.stats


def _iter_bulk_labels(path: Path) -> list[list[dict[str, Any]]]:
    """Yield batches from an openFDA bulk JSON, gzip JSON, or zip file."""
    batches: list[list[dict[str, Any]]] = []
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                payload = json.loads(archive.read(name).decode("utf-8"))
                batches.append(_as_label_list(payload))
        return batches

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        content = handle.read()
    try:
        batches.append(_as_label_list(json.loads(content)))
    except json.JSONDecodeError:
        records = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip() and line.strip().startswith("{")
        ]
        for start in range(0, len(records), 500):
            batches.append(records[start : start + 500])
    return batches


def _as_label_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    return []


def ingest_openfda(
    search_terms: list[str] | None = None,
    max_labels: int | None = None,
    limit: int = 100,
    rate_limit: float = 10.0,
    use_bulk: bool = False,
    bulk_dir: Path | None = None,
) -> IngestStats:
    """Main entry point for openFDA ingestion with API fallback."""
    config = get_config()
    effective_rate = rate_limit or float(config.rate_limits.openfda)
    raw_root = Path(config.llm.cache_dir).parent / "raw" / "openfda"
    ingestor = OpenFDAIngestor(cache_dir=raw_root, rate_limit=effective_rate, config=config)

    if use_bulk:
        stats = ingestor.ingest_from_bulk(bulk_dir=bulk_dir, max_labels=max_labels)
        if stats.fetched > 0:
            return stats
        logger.warning("openfda_bulk_empty_falling_back_to_api")

    return ingestor.ingest_from_api(
        search_terms=search_terms,
        max_labels=max_labels,
        limit=limit,
    )
