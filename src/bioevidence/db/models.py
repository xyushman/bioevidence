"""Database models for BioEvidence."""

from datetime import date, datetime
from enum import Enum as PyEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocType(PyEnum):
    TRIAL = "trial"
    PUBMED = "pubmed"
    LABEL = "label"


class OutcomeKind(PyEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class SplitType(PyEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Trial(Base):
    __tablename__ = "trials"

    nct_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    brief_summary: Mapped[str | None] = mapped_column(Text)
    detailed_description: Mapped[str | None] = mapped_column(Text)
    eligibility_text: Mapped[str | None] = mapped_column(Text)
    phase: Mapped[str | None] = mapped_column(String(50))
    overall_status: Mapped[str | None] = mapped_column(String(50))
    study_type: Mapped[str | None] = mapped_column(String(50))
    allocation: Mapped[str | None] = mapped_column(String(50))
    masking: Mapped[str | None] = mapped_column(String(100))
    enrollment: Mapped[int | None] = mapped_column(Integer)
    enrollment_type: Mapped[str | None] = mapped_column(String(50))
    start_date: Mapped[date | None] = mapped_column(Date)
    completion_date: Mapped[date | None] = mapped_column(Date)
    sponsor: Mapped[str | None] = mapped_column(Text)
    sponsor_class: Mapped[str | None] = mapped_column(String(50))
    min_age_years: Mapped[float | None] = mapped_column(Integer)
    max_age_years: Mapped[float | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(20))
    countries: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    num_arms: Mapped[int | None] = mapped_column(Integer)
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    date_imputed: Mapped[bool] = mapped_column(Boolean, default=False)

    conditions: Mapped[list["TrialCondition"]] = relationship(
        back_populates="trial", cascade="all, delete-orphan"
    )
    interventions: Mapped[list["TrialIntervention"]] = relationship(
        back_populates="trial", cascade="all, delete-orphan"
    )
    outcomes: Mapped[list["TrialOutcome"]] = relationship(
        back_populates="trial", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_trials_phase", "phase"),
        Index("ix_trials_start_date", "start_date"),
        Index("ix_trials_overall_status", "overall_status"),
    )


class TrialCondition(Base):
    __tablename__ = "trial_conditions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nct_id: Mapped[str] = mapped_column(
        ForeignKey("trials.nct_id", ondelete="CASCADE"), nullable=False
    )
    condition: Mapped[str] = mapped_column(Text, nullable=False)
    condition_mesh: Mapped[str | None] = mapped_column(Text)

    trial: Mapped["Trial"] = relationship(back_populates="conditions")

    __table_args__ = (
        Index("ix_trial_conditions_nct_id", "nct_id"),
        Index("ix_trial_conditions_mesh", "condition_mesh"),
    )


class TrialIntervention(Base):
    __tablename__ = "trial_interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nct_id: Mapped[str] = mapped_column(
        ForeignKey("trials.nct_id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str | None] = mapped_column(String(50))
    description: Mapped[str | None] = mapped_column(Text)

    trial: Mapped["Trial"] = relationship(back_populates="interventions")

    __table_args__ = (
        Index("ix_trial_interventions_nct_id", "nct_id"),
        Index("ix_trial_interventions_name", "name"),
    )


class TrialOutcome(Base):
    __tablename__ = "trial_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nct_id: Mapped[str] = mapped_column(
        ForeignKey("trials.nct_id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[OutcomeKind] = mapped_column(Enum(OutcomeKind), nullable=False)
    measure: Mapped[str] = mapped_column(Text, nullable=False)
    time_frame: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    trial: Mapped["Trial"] = relationship(back_populates="outcomes")

    __table_args__ = (
        Index("ix_trial_outcomes_nct_id", "nct_id"),
        Index("ix_trial_outcomes_kind", "kind"),
    )


class PubMedArticle(Base):
    __tablename__ = "pubmed_articles"

    pmid: Mapped[str] = mapped_column(String(20), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    journal: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    mesh_terms: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (Index("ix_pubmed_year", "year"),)


class DrugLabel(Base):
    __tablename__ = "drug_labels"

    set_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    brand_names: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    generic_names: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    sections: Mapped[dict] = mapped_column(JSON, default=dict)
    effective_date: Mapped[date | None] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(Text)

    __table_args__ = (Index("ix_drug_labels_effective_date", "effective_date"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_type: Mapped[DocType] = mapped_column(Enum(DocType), nullable=False)
    doc_id: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    embedding_ft: Mapped[list[float] | None] = mapped_column(Vector(384))
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)
    split: Mapped[SplitType] = mapped_column(
        Enum(SplitType), nullable=False, default=SplitType.TRAIN
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_chunks_doc_type", "doc_type"),
        Index("ix_chunks_doc_id", "doc_id"),
        Index("ix_chunks_split", "split"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        # HNSW indexes created separately via SQL
    )


# Event to update tsvector on insert/update


@event.listens_for(Chunk, "before_insert")
@event.listens_for(Chunk, "before_update")
def update_tsv(mapper, connection, target):
    """Update tsvector from text before insert/update."""
    if target.text:
        target.tsv = func.to_tsvector("english", target.text)
