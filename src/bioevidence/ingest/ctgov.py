"""ClinicalTrials.gov v2 API ingestor."""

import gzip
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from bioevidence.config import get_config
from bioevidence.db.models import Trial, TrialCondition, TrialIntervention, TrialOutcome
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


class CTGovClient:
    """Client for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, rate_limit: float = 3.0):
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
    def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """Make a GET request with retry logic."""
        self._rate_limit()
        response = self.client.get(url, params=params)
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
        """Search for studies with pagination."""
        params = {
            "pageSize": page_size,
            "format": "json",
            "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
            "Phase,OverallStatus,StudyType,Allocation,Masking,Enrollment,"
            "EnrollmentType,StartDate,CompletionDate,Sponsor,LeadSponsor,"
            "MinimumAge,MaximumAge,Sex,LocationCountry,ArmGroup,"
            "PrimaryOutcome,SecondaryOutcome,Condition,Keyword",
        }
        if condition:
            params["query.cond"] = condition
        if overall_status:
            params["filter.overallStatus"] = "|".join(overall_status)
        if start_date_min:
            params["filter.startDate"] = f"{start_date_min}:MAX"
        if page_token:
            params["pageToken"] = page_token

        response = self._get(self.BASE_URL, params)
        return response.json()

    def get_study(self, nct_id: str) -> dict[str, Any]:
        """Get a single study by NCT ID."""
        url = f"{self.BASE_URL}/{nct_id}"
        params = {"format": "json"}
        response = self._get(url, params)
        return response.json()

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
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json.gz"

    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached response if available."""
        path = self._cache_path(key)
        if path.exists():
            try:
                with gzip.open(path, "rt") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("cache_read_failed", key=key, error=str(e))
            else:
                logger.debug("cache_hit", key=key)
                return data
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


def parse_date(date_str: str | None) -> tuple[date | None, bool]:
    """Parse date string, handling partial dates. Returns (date, imputed_flag)."""
    if not date_str:
        return None, False

    # Handle formats: YYYY-MM-DD, YYYY-MM, YYYY
    parts = date_str.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2])), False
        elif len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1), True
        elif len(parts) == 1:
            return date(int(parts[0]), 1, 1), True
    except ValueError:
        pass
    return None, False


def parse_age(age_str: str | None) -> float | None:
    """Parse age string like '18 Years' to numeric years."""
    if not age_str:
        return None
    try:
        # Remove units and convert
        num_str = "".join(c for c in age_str if c.isdigit() or c == ".")
        return float(num_str) if num_str else None
    except ValueError:
        return None


def parse_phase(phase_str: str | None) -> str | None:
    """Normalize phase string."""
    if not phase_str:
        return None
    # Handle "PHASE2/PHASE3" -> "Phase 2/3"
    phases = []
    for p in phase_str.replace("PHASE", "").split("/"):
        p = p.strip()
        if p:
            phases.append(f"Phase {p}")
    return "/".join(phases) if phases else None


class CTGovIngestor:
    """Ingestor for ClinicalTrials.gov data."""

    def __init__(self, cache_dir: Path | None = None, rate_limit: float = 3.0):
        config = get_config()
        self.client = CTGovClient(rate_limit=rate_limit)
        self.cache = RawCache(cache_dir or Path(config.llm.cache_dir).parent / "raw" / "ctgov")
        self.stats = IngestStats()

    def _extract_conditions(self, study: dict) -> list[dict]:
        """Extract conditions from study."""
        conditions = []
        for cond in study.get("Condition", []):
            conditions.append(
                {
                    "condition": cond,
                    "condition_mesh": None,  # Could be enhanced with MeSH mapping
                }
            )
        return conditions

    def _extract_interventions(self, study: dict) -> list[dict]:
        """Extract interventions from study."""
        interventions = []
        for arm in study.get("ArmGroup", []):
            for intervention in arm.get("InterventionName", []):
                interventions.append(
                    {
                        "name": intervention,
                        "type": arm.get("ArmGroupType"),
                        "description": arm.get("Description"),
                    }
                )
        return interventions

    def _extract_outcomes(self, study: dict) -> list[dict]:
        """Extract outcomes from study."""
        outcomes = []
        for outcome in study.get("PrimaryOutcome", []):
            outcomes.append(
                {
                    "kind": "primary",
                    "measure": outcome.get("Measure"),
                    "time_frame": outcome.get("TimeFrame"),
                    "description": outcome.get("Description"),
                }
            )
        for outcome in study.get("SecondaryOutcome", []):
            outcomes.append(
                {
                    "kind": "secondary",
                    "measure": outcome.get("Measure"),
                    "time_frame": outcome.get("TimeFrame"),
                    "description": outcome.get("Description"),
                }
            )
        return outcomes

    def _parse_study(self, study: dict) -> dict:
        """Parse study into trial dict."""
        protocol = study.get("ProtocolSection", {})
        identification = protocol.get("IdentificationModule", {})
        status = protocol.get("StatusModule", {})
        design = protocol.get("DesignModule", {})
        description = protocol.get("DescriptionModule", {})
        eligibility = protocol.get("EligibilityModule", {})
        sponsor = protocol.get("SponsorCollaboratorsModule", {})
        arms = protocol.get("ArmsInterventionsModule", {})
        locations = protocol.get("LocationsModule", {})

        nct_id = identification.get("NCTId")
        if not nct_id:
            raise ValueError("Missing NCTId")

        # Parse dates
        start_date_str = status.get("StartDateStruct", {}).get("Date") or status.get("StartDate")
        completion_date_str = status.get("CompletionDateStruct", {}).get("Date") or status.get(
            "CompletionDate"
        )
        start_date, start_imputed = parse_date(start_date_str)
        completion_date, completion_imputed = parse_date(completion_date_str)

        # Parse age
        min_age = parse_age(eligibility.get("MinimumAge"))
        max_age = parse_age(eligibility.get("MaximumAge"))

        # Parse phase
        phase = parse_phase(design.get("Phase"))

        # Countries
        countries = locations.get("LocationCountry", [])

        # Arms count
        num_arms = len(arms.get("ArmGroup", []))

        trial_data = {
            "nct_id": nct_id,
            "title": identification.get("BriefTitle")
            or identification.get("OfficialTitle")
            or "Unknown",
            "brief_summary": description.get("BriefSummary"),
            "detailed_description": description.get("DetailedDescription"),
            "eligibility_text": eligibility.get("EligibilityCriteria"),
            "phase": phase,
            "overall_status": status.get("OverallStatus"),
            "study_type": design.get("StudyType"),
            "allocation": design.get("Allocation"),
            "masking": design.get("Masking"),
            "enrollment": design.get("Enrollment"),
            "enrollment_type": design.get("EnrollmentType"),
            "start_date": start_date,
            "completion_date": completion_date,
            "sponsor": sponsor.get("LeadSponsor", {}).get("Name"),
            "sponsor_class": sponsor.get("LeadSponsor", {}).get("Class"),
            "min_age_years": min_age,
            "max_age_years": max_age,
            "sex": eligibility.get("Sex"),
            "countries": countries if countries else None,
            "num_arms": num_arms,
            "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
            "fetched_at": datetime.utcnow(),
            "date_imputed": start_imputed or completion_imputed,
        }

        return trial_data

    def _upsert_trial(self, session, trial_data: dict, study: dict) -> tuple[Trial, bool]:
        """Upsert trial and related data. Returns (trial, is_new)."""
        nct_id = trial_data["nct_id"]

        # Check existing
        existing = session.get(Trial, nct_id)
        is_new = existing is None

        if is_new:
            trial = Trial(**trial_data)
            session.add(trial)
        else:
            # Update fields
            for key, value in trial_data.items():
                if key != "nct_id":
                    setattr(existing, key, value)
            trial = existing

        session.flush()

        # Upsert conditions
        session.query(TrialCondition).filter(TrialCondition.nct_id == nct_id).delete()
        for cond_data in self._extract_conditions(study):
            cond = TrialCondition(nct_id=nct_id, **cond_data)
            session.add(cond)

        # Upsert interventions
        session.query(TrialIntervention).filter(TrialIntervention.nct_id == nct_id).delete()
        for int_data in self._extract_interventions(study):
            intervention = TrialIntervention(nct_id=nct_id, **int_data)
            session.add(intervention)

        # Upsert outcomes
        session.query(TrialOutcome).filter(TrialOutcome.nct_id == nct_id).delete()
        for out_data in self._extract_outcomes(study):
            outcome = TrialOutcome(nct_id=nct_id, **out_data)
            session.add(outcome)

        return trial, is_new

    def ingest(
        self,
        conditions: list[str] | None = None,
        statuses: list[str] | None = None,
        start_date_min: str = "2005-01-01",
        max_pages: int | None = None,
    ) -> IngestStats:
        """Run ingestion for given conditions."""
        conditions = conditions or ["cancer"]
        statuses = statuses or ["Completed", "Active, not recruiting", "Recruiting"]

        for condition in conditions:
            logger.info("ingesting_condition", condition=condition)
            page_token = None
            page = 0

            while True:
                if max_pages and page >= max_pages:
                    break

                cache_key = f"search_{condition}_{page_token or 'first'}"
                data = self.cache.get(cache_key)

                if data is None:
                    data = self.client.search_studies(
                        condition=condition,
                        overall_status=statuses,
                        start_date_min=start_date_min,
                        page_token=page_token,
                    )
                    self.cache.set(cache_key, data)

                studies = data.get("studies", [])
                if not studies:
                    break

                for study in studies:
                    try:
                        self.stats.fetched += 1
                        trial_data = self._parse_study(study)

                        with get_session() as session:
                            trial, is_new = self._upsert_trial(session, trial_data, study)
                            if is_new:
                                self.stats.inserted += 1
                            else:
                                self.stats.updated += 1
                            session.commit()

                    except Exception:
                        self.stats.errors += 1
                        logger.exception(
                            "ingest_error",
                            nct_id=study.get("ProtocolSection", {})
                            .get("IdentificationModule", {})
                            .get("NCTId"),
                        )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                page += 1

        self.client.close()
        return self.stats


def ingest_ctgov(
    conditions: list[str] | None = None,
    statuses: list[str] | None = None,
    start_date_min: str = "2005-01-01",
    max_pages: int | None = None,
    rate_limit: float = 3.0,
) -> IngestStats:
    """Main entry point for CT.gov ingestion."""
    config = get_config()
    cache_dir = Path(config.llm.cache_dir).parent / "raw" / "ctgov"
    ingestor = CTGovIngestor(cache_dir=cache_dir, rate_limit=rate_limit)
    return ingestor.ingest(conditions, statuses, start_date_min, max_pages)
