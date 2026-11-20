"""Pydantic validation schemas for ingested biomedical records."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bioevidence.ingest.common import clean_text, utcnow


class SourceRecord(BaseModel):
    """Fields shared by every ingested document."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_url: str = Field(min_length=1)
    fetched_at: datetime = Field(default_factory=utcnow)


class ConditionRecord(BaseModel):
    """A normalized trial condition."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    condition: str = Field(min_length=1)
    condition_mesh: str | None = None


class InterventionRecord(BaseModel):
    """A normalized trial intervention."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    type: str | None = None
    description: str | None = None

    _clean = field_validator("name", "type", "description", mode="before")(clean_text)


class OutcomeRecord(BaseModel):
    """A normalized trial outcome."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: str = Field(pattern="^(primary|secondary)$")
    measure: str = Field(min_length=1)
    time_frame: str | None = None
    description: str | None = None

    _clean = field_validator("measure", "time_frame", "description", mode="before")(clean_text)


class TrialRecord(SourceRecord):
    """Validated trial row plus its related records."""

    nct_id: str = Field(min_length=1, max_length=20)
    title: str = Field(min_length=1)
    brief_summary: str | None = None
    detailed_description: str | None = None
    eligibility_text: str | None = None
    phase: str | None = None
    overall_status: str | None = None
    study_type: str | None = None
    allocation: str | None = None
    masking: str | None = None
    enrollment: int | None = Field(default=None, ge=0)
    enrollment_type: str | None = None
    start_date: date | None = None
    completion_date: date | None = None
    sponsor: str | None = None
    sponsor_class: str | None = None
    min_age_years: float | None = Field(default=None, ge=0, le=120)
    max_age_years: float | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    countries: list[str] | None = None
    num_arms: int | None = Field(default=None, ge=0)
    date_imputed: bool = False
    conditions: list[ConditionRecord] = Field(default_factory=list)
    interventions: list[InterventionRecord] = Field(default_factory=list)
    outcomes: list[OutcomeRecord] = Field(default_factory=list)

    _clean = field_validator(
        "title",
        "brief_summary",
        "detailed_description",
        "eligibility_text",
        "phase",
        "overall_status",
        "study_type",
        "allocation",
        "masking",
        "enrollment_type",
        "sponsor",
        "sponsor_class",
        "sex",
        mode="before",
    )(clean_text)

    @model_validator(mode="after")
    def require_narrative_text(self) -> TrialRecord:
        """Reject trials that cannot produce a useful chunk."""
        if not self.brief_summary and not self.detailed_description:
            msg = "Trial has neither a brief summary nor a detailed description"
            raise ValueError(msg)
        if self.completion_date and self.start_date and self.completion_date < self.start_date:
            msg = "completion_date precedes start_date"
            raise ValueError(msg)
        return self

    def to_orm_kwargs(self) -> dict[str, Any]:
        """Return only columns that exist on the Trial ORM model."""
        return self.model_dump(
            exclude={"nct_id", "conditions", "interventions", "outcomes"},
        )


class PubMedRecord(SourceRecord):
    """Validated PubMed article."""

    pmid: str = Field(min_length=1, max_length=20)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    journal: str | None = None
    year: int | None = Field(default=None, ge=1950, le=2100)
    mesh_terms: list[str] | None = None

    _clean = field_validator("title", "abstract", "journal", mode="before")(clean_text)

    def to_orm_kwargs(self) -> dict[str, Any]:
        """Return ORM-ready values."""
        return self.model_dump()


class DrugLabelRecord(SourceRecord):
    """Validated openFDA drug label."""

    set_id: str = Field(min_length=1, max_length=100)
    brand_names: list[str] | None = None
    generic_names: list[str] | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    effective_date: date | None = None
    matched_interventions: list[str] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def sections_must_have_text(cls, value: dict[str, str]) -> dict[str, str]:
        """Drop empty label sections."""
        return {name: text for name, text in value.items() if text and text.strip()}

    def to_orm_kwargs(self) -> dict[str, Any]:
        """Return ORM-ready values."""
        return self.model_dump()
