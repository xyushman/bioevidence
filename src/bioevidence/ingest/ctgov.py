"""ClinicalTrials.gov API v2 ingestor."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import structlog
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from bioevidence.config import Config, get_config
from bioevidence.db.models import (
    OutcomeKind,
    Trial,
    TrialCondition,
    TrialIntervention,
    TrialOutcome,
)
from bioevidence.db.session import get_session
from bioevidence.ingest.common import IngestStats, RateLimiter, RawCache, clean_text, utcnow
from bioevidence.ingest.runs import record_ingest_run
from bioevidence.quality.quarantine import quarantine_record
from bioevidence.quality.schemas import (
    ConditionRecord,
    InterventionRecord,
    OutcomeRecord,
    TrialRecord,
)

logger = structlog.get_logger(__name__)

__all__ = ["CTGovIngestor", "IngestStats", "ingest_ctgov"]

STUDY_FIELDS = (
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.descriptionModule.detailedDescription",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct.date",
    "protocolSection.statusModule.completionDateStruct.date",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.designInfo.allocation",
    "protocolSection.designModule.designInfo.maskingInfo.masking",
    "protocolSection.designModule.enrollmentInfo.count",
    "protocolSection.designModule.enrollmentInfo.type",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.contactsLocationsModule.locations.country",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.outcomesModule.secondaryOutcomes",
    "derivedSection.conditionBrowseModule.meshes",
)

_AGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", re.IGNORECASE)


def nested_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted key from a nested API response."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_date(date_str: str | None) -> tuple[date | None, bool]:
    """Parse an ISO date, returning ``(date, was_imputed)``."""
    if not date_str:
        return None, False
    parts = date_str.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2])), False
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1), True
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1), True
    except ValueError:
        return None, False
    return None, False


def parse_age(age_str: str | None) -> float | None:
    """Convert age strings such as ``'18 Years'`` or ``'6 Months'`` to years."""
    if not age_str:
        return None
    match = _AGE_PATTERN.search(age_str)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    factors = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}
    return round(value * factors[unit], 3)


def parse_phase(phases: str | list[str] | None) -> str | None:
    """Normalize API v2 phase values to display strings."""
    if not phases:
        return None
    raw_values = [phases] if isinstance(phases, str) else phases
    normalized: list[str] = []
    for raw in raw_values:
        if not raw:
            continue
        for token in str(raw).upper().split("/"):
            token = token.strip()
            if not token or token == "NA":
                continue
            if token.startswith("EARLY_PHASE"):
                normalized.append("Early Phase 1")
                continue
            number = token.replace("PHASE", "").strip()
            if number:
                normalized.append(f"Phase {number}")
    phases = list(dict.fromkeys(normalized))
    if not phases:
        return None
    if all(phase.startswith("Phase ") for phase in phases):
        return "Phase " + "/".join(phase.removeprefix("Phase ") for phase in phases)
    return "/".join(phases)


class CTGovClient:
    """Rate-limited client for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, rate_limit: float = 3.0, timeout: float = 60.0) -> None:
        self.limiter = RateLimiter(rate_limit)
        self.client = httpx.Client(timeout=timeout)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _get(self, params: dict[str, Any]) -> httpx.Response:
        self.limiter.wait()
        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response

    def search_studies(
        self,
        condition: str | None = None,
        overall_status: list[str] | None = None,
        start_date_min: str | None = None,
        page_size: int = 1000,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one page of studies using API v2 filter syntax."""
        params: dict[str, Any] = {
            "pageSize": page_size,
            "format": "json",
            "fields": ",".join(STUDY_FIELDS),
        }
        advanced_filters = ["AREA[StudyType]INTERVENTIONAL"]
        if condition:
            params["query.cond"] = condition
        if overall_status:
            params["filter.overallStatus"] = ",".join(overall_status)
        if start_date_min:
            advanced_filters.append(f"AREA[StartDate]RANGE[{start_date_min},MAX]")
        params["filter.advanced"] = " AND ".join(advanced_filters)
        if page_token:
            params["pageToken"] = page_token
        payload = self._get(params).json()
        return payload if isinstance(payload, dict) else {}

    def get_study(self, nct_id: str) -> dict[str, Any]:
        """Fetch one study by NCT ID."""
        self.limiter.wait()
        response = self.client.get(f"{self.BASE_URL}/{nct_id}", params={"format": "json"})
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()


class CTGovIngestor:
    """Parse and upsert ClinicalTrials.gov studies."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: float = 3.0,
        config: Config | None = None,
        client: CTGovClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.client = client or CTGovClient(rate_limit=rate_limit)
        raw_root = Path(self.config.llm.cache_dir).parent / "raw" / "ctgov"
        self.cache = RawCache(cache_dir or raw_root)
        self.stats = IngestStats()

    @staticmethod
    def _condition_records(study: dict[str, Any]) -> list[ConditionRecord]:
        conditions = nested_get(study, "protocolSection.conditionsModule.conditions", []) or []
        meshes = nested_get(study, "derivedSection.conditionBrowseModule.meshes", []) or []
        mesh_terms = [mesh.get("term", "") for mesh in meshes if isinstance(mesh, dict)]
        records: list[ConditionRecord] = []
        for condition in conditions:
            if not condition:
                continue
            lowered = str(condition).lower()
            match = next(
                (term for term in mesh_terms if lowered in term.lower() or term.lower() in lowered),
                None,
            )
            records.append(
                ConditionRecord(
                    condition=clean_text(str(condition)) or str(condition), condition_mesh=match
                )
            )
        return records

    @staticmethod
    def _intervention_records(study: dict[str, Any]) -> list[InterventionRecord]:
        interventions = (
            nested_get(study, "protocolSection.armsInterventionsModule.interventions", []) or []
        )
        records: list[InterventionRecord] = []
        for item in interventions:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            records.append(
                InterventionRecord(
                    name=item["name"],
                    type=item.get("type"),
                    description=item.get("description"),
                )
            )
        if records:
            return records

        # Some partial field sets expose only arm group intervention names.
        for arm in nested_get(study, "protocolSection.armsInterventionsModule.armGroups", []) or []:
            for raw_name in arm.get("interventionNames", []) or []:
                name = str(raw_name).split(":", 1)[-1].strip()
                if name:
                    records.append(
                        InterventionRecord(
                            name=name,
                            type=arm.get("type"),
                            description=arm.get("description"),
                        )
                    )
        return records

    @staticmethod
    def _outcome_records(study: dict[str, Any]) -> list[OutcomeRecord]:
        records: list[OutcomeRecord] = []
        for kind, path in (
            ("primary", "protocolSection.outcomesModule.primaryOutcomes"),
            ("secondary", "protocolSection.outcomesModule.secondaryOutcomes"),
        ):
            for item in nested_get(study, path, []) or []:
                if not isinstance(item, dict) or not item.get("measure"):
                    continue
                records.append(
                    OutcomeRecord(
                        kind=kind,
                        measure=item["measure"],
                        time_frame=item.get("timeFrame"),
                        description=item.get("description"),
                    )
                )
        return records

    @staticmethod
    def parse_study(study: dict[str, Any]) -> TrialRecord:
        """Convert one API v2 study into a validated record."""
        nct_id = nested_get(study, "protocolSection.identificationModule.nctId")
        if not nct_id:
            msg = "Study is missing protocolSection.identificationModule.nctId"
            raise ValueError(msg)

        start_date, start_imputed = parse_date(
            nested_get(study, "protocolSection.statusModule.startDateStruct.date")
        )
        completion_date, completion_imputed = parse_date(
            nested_get(study, "protocolSection.statusModule.completionDateStruct.date")
        )
        countries = []
        for location in (
            nested_get(study, "protocolSection.contactsLocationsModule.locations", []) or []
        ):
            country = location.get("country") if isinstance(location, dict) else None
            if country and country not in countries:
                countries.append(country)

        phase = parse_phase(nested_get(study, "protocolSection.designModule.phases"))
        return TrialRecord(
            nct_id=str(nct_id),
            title=nested_get(study, "protocolSection.identificationModule.briefTitle")
            or nested_get(study, "protocolSection.identificationModule.officialTitle")
            or f"Study {nct_id}",
            brief_summary=nested_get(study, "protocolSection.descriptionModule.briefSummary"),
            detailed_description=nested_get(
                study, "protocolSection.descriptionModule.detailedDescription"
            ),
            eligibility_text=nested_get(
                study, "protocolSection.eligibilityModule.eligibilityCriteria"
            ),
            phase=phase,
            overall_status=nested_get(study, "protocolSection.statusModule.overallStatus"),
            study_type=nested_get(study, "protocolSection.designModule.studyType"),
            allocation=nested_get(study, "protocolSection.designModule.designInfo.allocation"),
            masking=nested_get(
                study, "protocolSection.designModule.designInfo.maskingInfo.masking"
            ),
            enrollment=nested_get(study, "protocolSection.designModule.enrollmentInfo.count"),
            enrollment_type=nested_get(study, "protocolSection.designModule.enrollmentInfo.type"),
            start_date=start_date,
            completion_date=completion_date,
            sponsor=nested_get(
                study, "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
            ),
            sponsor_class=nested_get(
                study, "protocolSection.sponsorCollaboratorsModule.leadSponsor.class"
            ),
            min_age_years=parse_age(
                nested_get(study, "protocolSection.eligibilityModule.minimumAge")
            ),
            max_age_years=parse_age(
                nested_get(study, "protocolSection.eligibilityModule.maximumAge")
            ),
            sex=nested_get(study, "protocolSection.eligibilityModule.sex"),
            countries=countries or None,
            num_arms=len(
                nested_get(study, "protocolSection.armsInterventionsModule.armGroups", []) or []
            ),
            date_imputed=start_imputed or completion_imputed,
            source_url=f"https://clinicaltrials.gov/study/{nct_id}",
            fetched_at=utcnow(),
            conditions=CTGovIngestor._condition_records(study),
            interventions=CTGovIngestor._intervention_records(study),
            outcomes=CTGovIngestor._outcome_records(study),
        )

    @staticmethod
    def upsert_trial(session: Session, record: TrialRecord) -> tuple[Trial, bool]:
        """Upsert a trial and replace its related rows. Returns ``(trial, is_new)``."""
        trial = session.get(Trial, record.nct_id)
        is_new = trial is None
        if trial is None:
            trial = Trial(nct_id=record.nct_id, **record.to_orm_kwargs())
            session.add(trial)
        else:
            for key, value in record.to_orm_kwargs().items():
                if key != "nct_id":
                    setattr(trial, key, value)
        session.flush()

        for model in (TrialCondition, TrialIntervention, TrialOutcome):
            session.query(model).filter(model.nct_id == record.nct_id).delete()

        for condition in record.conditions:
            session.add(TrialCondition(nct_id=record.nct_id, **condition.model_dump()))
        for intervention in record.interventions:
            session.add(TrialIntervention(nct_id=record.nct_id, **intervention.model_dump()))
        for outcome in record.outcomes:
            session.add(
                TrialOutcome(
                    nct_id=record.nct_id,
                    kind=OutcomeKind(outcome.kind),
                    measure=outcome.measure,
                    time_frame=outcome.time_frame,
                    description=outcome.description,
                )
            )
        return trial, is_new

    @staticmethod
    def _scope_keywords(config: Config) -> list[str]:
        keywords: list[str] = []
        for area in config.scope.therapeutic_areas:
            keywords.extend(area.keywords)
        return list(dict.fromkeys(keywords)) or ["cancer"]

    @staticmethod
    def _scope_statuses(config: Config) -> list[str]:
        configured = config.scope.trial_filters.get("overall_statuses") or []
        return [str(status).upper().replace(", ", "_").replace(" ", "_") for status in configured]

    def _cache_key(self, params: dict[str, Any]) -> str:
        return f"ctgov:{json.dumps(params, sort_keys=True)}"

    def ingest(
        self,
        conditions: list[str] | None = None,
        statuses: list[str] | None = None,
        start_date_min: str = "2005-01-01",
        max_pages: int | None = None,
        max_studies: int | None = None,
        page_size: int = 1000,
    ) -> IngestStats:
        """Ingest interventional studies until the configured volume target is met."""
        self.stats = IngestStats()
        target = max_studies or self.config.scope.targets.get("trials", 20_000)
        conditions = conditions or self._scope_keywords(self.config)
        statuses = statuses or self._scope_statuses(self.config) or ["COMPLETED", "RECRUITING"]
        seen: set[str] = set()

        try:
            for condition in conditions:
                if len(seen) >= target:
                    break
                page_token: str | None = None
                page = 0
                logger.info("ctgov_condition_start", condition=condition, target=target)

                while True:
                    if max_pages is not None and page >= max_pages:
                        break
                    params: dict[str, Any] = {
                        "condition": condition,
                        "statuses": statuses,
                        "start_date_min": start_date_min,
                        "page_size": page_size,
                        "page_token": page_token,
                    }
                    cache_key = self._cache_key(params)
                    data = self.cache.get(cache_key)
                    if data is None:
                        data = self.client.search_studies(
                            condition=condition,
                            overall_status=statuses,
                            start_date_min=start_date_min,
                            page_size=page_size,
                            page_token=page_token,
                        )
                        self.cache.set(cache_key, data)

                    studies = data.get("studies", [])
                    if not studies:
                        break

                    with get_session() as session:
                        for study in studies:
                            if len(seen) >= target:
                                break
                            try:
                                record = self.parse_study(study)
                            except ValidationError as exc:
                                self.stats.skipped += 1
                                quarantine_record("ctgov", study, str(exc))
                                logger.warning("ctgov_record_rejected", error=str(exc))
                                continue
                            except Exception as exc:
                                self.stats.errors += 1
                                quarantine_record("ctgov", study, str(exc))
                                logger.warning("ctgov_parse_failed", error=str(exc))
                                continue
                            if record.nct_id in seen:
                                continue
                            seen.add(record.nct_id)
                            _, is_new = self.upsert_trial(session, record)
                            if is_new:
                                self.stats.inserted += 1
                            else:
                                self.stats.updated += 1
                        session.commit()

                    if len(seen) >= target:
                        break

                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
                    page += 1
        finally:
            self.client.close()

        self.stats.fetched = len(seen)
        with get_session() as session:
            row_count = int(session.scalar(select(func.count()).select_from(Trial)) or 0)
            record_ingest_run("ctgov", self.stats, row_count)
        logger.info("ctgov_ingest_complete", **self.stats.as_dict())
        return self.stats


def ingest_ctgov(
    conditions: list[str] | None = None,
    statuses: list[str] | None = None,
    start_date_min: str = "2005-01-01",
    max_pages: int | None = None,
    max_studies: int | None = None,
    rate_limit: float = 3.0,
) -> IngestStats:
    """Main entry point for ClinicalTrials.gov ingestion."""
    config = get_config()
    raw_root = Path(config.llm.cache_dir).parent / "raw" / "ctgov"
    ingestor = CTGovIngestor(cache_dir=raw_root, rate_limit=rate_limit, config=config)
    return ingestor.ingest(
        conditions=conditions,
        statuses=statuses,
        start_date_min=start_date_min,
        max_pages=max_pages,
        max_studies=max_studies,
    )
