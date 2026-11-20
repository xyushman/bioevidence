"""Deterministic chunking for BioEvidence documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, DrugLabel, PubMedArticle, SplitType, Trial
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_tokenizer_cache: dict[str, Any] = {}


class DeterministicTokenizer:
    """Offline tokenizer fallback with stable, reproducible word/punctuation tokens."""

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self.tokenize(text)

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        """Join tokens with readable spacing while preserving punctuation."""
        text = ""
        for token in tokens:
            if not text or token in ".,!?;:%)]}" or text[-1] in "([{":
                text += token
            else:
                text += f" {token}"
        return text


def get_tokenizer(model_name: str | None = None) -> Any:
    """Return a cached tokenizer, falling back offline when no local model exists."""
    config = get_config()
    name = model_name or config.embedding.model
    if name not in _tokenizer_cache:
        try:
            _tokenizer_cache[name] = AutoTokenizer.from_pretrained(
                name,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("tokenizer_fallback_used", model=name, error=str(exc))
            _tokenizer_cache[name] = DeterministicTokenizer()
    return _tokenizer_cache[name]


def _tokenize_text(text: str, model_name: str | None = None) -> list[Any]:
    tokenizer = get_tokenizer(model_name)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return list(encode(text, add_special_tokens=False))
    return list(tokenizer.tokenize(text))


def _decode_tokens(tokens: list[Any], model_name: str | None = None) -> str:
    tokenizer = get_tokenizer(model_name)
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(tokens, skip_special_tokens=True))
    return " ".join(str(token) for token in tokens)


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens using the configured tokenizer or the offline fallback."""
    return len(_tokenize_text(text, model_name))


@dataclass
class ChunkConfig:
    """Configuration for chunking."""

    max_tokens: int = 512
    overlap: int = 32
    trial_section_overlap: int = 40
    label_section_overlap: int = 40
    pubmed_max_tokens: int | None = 256
    label_max_tokens: int | None = 350

    def limit_for(self, doc_type: str) -> int:
        """Return the effective token limit for a document type."""
        limit = {
            "trial": self.max_tokens,
            "pubmed": self.pubmed_max_tokens or self.max_tokens,
            "label": self.label_max_tokens or self.max_tokens,
        }.get(doc_type, self.max_tokens)
        return min(limit, self.max_tokens)


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap: int,
    model_name: str | None = None,
) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    if not text:
        return []
    if max_tokens <= 0:
        msg = "max_tokens must be positive"
        raise ValueError(msg)
    if overlap < 0 or overlap >= max_tokens:
        msg = "overlap must be non-negative and smaller than max_tokens"
        raise ValueError(msg)

    tokens = _tokenize_text(text, model_name)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]
        chunk_text = _decode_tokens(window, model_name)
        while len(window) > 1 and count_tokens(chunk_text, model_name) > max_tokens:
            window = window[:-1]
            chunk_text = _decode_tokens(window, model_name)
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def split_text_by_paragraphs(
    text: str,
    max_tokens: int,
    overlap: int,
    model_name: str | None = None,
) -> list[str]:
    """Split long sections on paragraph boundaries when possible."""
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|\n+", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(split_text_by_tokens(paragraph, max_tokens, overlap, model_name))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if count_tokens(candidate, model_name) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
    if current:
        chunks.append(current)
    return chunks


def _make_chunk_dicts(
    section: str,
    text: str,
    config: ChunkConfig,
    overlap: int,
    model_name: str | None = None,
    header: str = "",
    max_tokens: int | None = None,
    split_by_paragraphs: bool = False,
) -> list[dict[str, Any]]:
    """Create one or more token-safe chunk records for a section."""
    if not text or not text.strip():
        return []
    limit = max_tokens or config.max_tokens
    full_text = header + text
    token_count = count_tokens(full_text, model_name)
    if token_count <= limit:
        return [
            {
                "section": section,
                "text": full_text,
                "token_count": token_count,
            }
        ]

    splitter = split_text_by_paragraphs if split_by_paragraphs else split_text_by_tokens
    header_tokens = count_tokens(header, model_name) if header else 0
    effective_overlap = min(overlap, limit - 1)
    if header_tokens >= limit:
        candidates = splitter(full_text, limit, effective_overlap, model_name)
    else:
        available = limit - header_tokens
        body_parts = splitter(
            text, available, min(effective_overlap, max(0, available - 1)), model_name
        )
        candidates = [header + part for part in body_parts]

    chunks: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        safe_text = candidate
        while count_tokens(safe_text, model_name) > limit and len(safe_text) > 1:
            safe_text = safe_text[:-1]
        if not safe_text.strip():
            continue
        chunks.append(
            {
                "section": section if len(candidates) == 1 else f"{section}_part{index + 1}",
                "text": safe_text,
                "token_count": count_tokens(safe_text, model_name),
            }
        )
    return chunks


def split_trial_sections(
    trial: Trial, config: ChunkConfig, model_name: str | None = None
) -> list[dict[str, Any]]:
    """Split a trial into section-based chunks."""
    header = f"[Trial {trial.nct_id}"
    if trial.phase:
        header += f" | {trial.phase}"
    if trial.conditions:
        cond_names = [c.condition for c in trial.conditions[:3]]
        header += f" | Condition: {', '.join(cond_names)}"
    header += "] "

    sections = [
        ("brief_summary", trial.brief_summary),
        ("detailed_description", trial.detailed_description),
        ("eligibility_inclusion", extract_inclusion_criteria(trial.eligibility_text)),
        ("eligibility_exclusion", extract_exclusion_criteria(trial.eligibility_text)),
        ("outcomes", extract_outcomes_text(trial)),
    ]

    chunks: list[dict[str, Any]] = []
    for section_name, text in sections:
        if text:
            chunks.extend(
                _make_chunk_dicts(
                    section_name,
                    text,
                    config,
                    config.trial_section_overlap,
                    model_name,
                    header=header,
                    max_tokens=config.limit_for("trial"),
                )
            )
    return chunks


def extract_inclusion_criteria(eligibility_text: str | None) -> str | None:
    """Extract inclusion criteria from eligibility text."""
    if not eligibility_text:
        return None

    patterns = [
        r"inclusion criteria[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
        r"inclusion[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
        r"criteria for inclusion[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, eligibility_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_exclusion_criteria(eligibility_text: str | None) -> str | None:
    """Extract exclusion criteria from eligibility text."""
    if not eligibility_text:
        return None

    patterns = [
        r"exclusion criteria[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
        r"exclusion[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
        r"criteria for exclusion[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, eligibility_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_outcomes_text(trial: Trial) -> str | None:
    """Extract outcomes as text."""
    if not trial.outcomes:
        return None

    parts: list[str] = []
    for outcome in trial.outcomes:
        kind_str = "Primary" if outcome.kind.value == "primary" else "Secondary"
        parts.append(f"{kind_str} Outcome: {outcome.measure}")
        if outcome.time_frame:
            parts.append(f"  Time Frame: {outcome.time_frame}")
        if outcome.description:
            parts.append(f"  Description: {outcome.description}")
    return "\n".join(parts) or None


def split_pubmed_article(
    article: PubMedArticle, config: ChunkConfig, model_name: str | None = None
) -> list[dict[str, Any]]:
    """Split a PubMed article into title/abstract chunks."""
    if not article.abstract:
        return []
    text = f"Title: {article.title}\n\nAbstract: {article.abstract}"
    return _make_chunk_dicts(
        "title_abstract",
        text,
        config,
        config.overlap,
        model_name,
        max_tokens=config.limit_for("pubmed"),
    )


def split_drug_label(
    label: DrugLabel, config: ChunkConfig, model_name: str | None = None
) -> list[dict[str, Any]]:
    """Split a drug label into token-safe section chunks."""
    chunks: list[dict[str, Any]] = []
    for section_name, section_text in (label.sections or {}).items():
        if not section_text:
            continue
        header = f"[Drug Label {label.set_id} | Section: {section_name}] "
        chunks.extend(
            _make_chunk_dicts(
                section_name,
                section_text,
                config,
                config.label_section_overlap,
                model_name,
                header=header,
                max_tokens=config.limit_for("label"),
                split_by_paragraphs=True,
            )
        )
    return chunks


class Chunker:
    """Build idempotent chunks for every document in the database."""

    def __init__(self, config: ChunkConfig | None = None, model_name: str | None = None) -> None:
        self.config = config or ChunkConfig()
        self.model_name = model_name
        self.stats: dict[str, int] = {"total": 0, "trials": 0, "pubmed": 0, "labels": 0}

    def build_chunks(self, batch_size: int = 1000) -> int:
        """Build chunks for all documents."""
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        total_chunks = 0
        with get_session() as session:
            total_chunks += self._chunk_trials(session, batch_size)
            total_chunks += self._chunk_pubmed(session, batch_size)
            total_chunks += self._chunk_labels(session, batch_size)
        return total_chunks

    @staticmethod
    def _replace_chunks(
        session: Session,
        doc_type: DocType,
        doc_id: str,
        chunk_data: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> int:
        session.query(Chunk).filter(
            Chunk.doc_type == doc_type,
            Chunk.doc_id == doc_id,
        ).delete()
        for item in chunk_data:
            session.add(
                Chunk(
                    doc_type=doc_type,
                    doc_id=doc_id,
                    section=item["section"],
                    text=item["text"],
                    token_count=item["token_count"],
                    chunk_metadata=metadata,
                    split=SplitType.TRAIN,
                )
            )
        return len(chunk_data)

    def _chunk_trials(self, session: Session, batch_size: int) -> int:
        count = 0
        offset = 0
        while True:
            trials = (
                session.query(Trial).order_by(Trial.nct_id).offset(offset).limit(batch_size).all()
            )
            if not trials:
                break
            for trial in trials:
                chunk_data = split_trial_sections(trial, self.config, self.model_name)
                count += self._replace_chunks(
                    session,
                    DocType.TRIAL,
                    trial.nct_id,
                    chunk_data,
                    {
                        "phase": trial.phase,
                        "status": trial.overall_status,
                        "start_year": trial.start_date.year if trial.start_date else None,
                    },
                )
            session.commit()
            offset += batch_size
            self.stats["trials"] += len(trials)
        return count

    def _chunk_pubmed(self, session: Session, batch_size: int) -> int:
        count = 0
        offset = 0
        while True:
            articles = (
                session.query(PubMedArticle)
                .order_by(PubMedArticle.pmid)
                .offset(offset)
                .limit(batch_size)
                .all()
            )
            if not articles:
                break
            for article in articles:
                chunk_data = split_pubmed_article(article, self.config, self.model_name)
                count += self._replace_chunks(
                    session,
                    DocType.PUBMED,
                    article.pmid,
                    chunk_data,
                    {"journal": article.journal, "year": article.year},
                )
            session.commit()
            offset += batch_size
            self.stats["pubmed"] += len(articles)
        return count

    def _chunk_labels(self, session: Session, batch_size: int) -> int:
        count = 0
        offset = 0
        while True:
            labels = (
                session.query(DrugLabel)
                .order_by(DrugLabel.set_id)
                .offset(offset)
                .limit(batch_size)
                .all()
            )
            if not labels:
                break
            for label in labels:
                chunk_data = split_drug_label(label, self.config, self.model_name)
                count += self._replace_chunks(
                    session,
                    DocType.LABEL,
                    label.set_id,
                    chunk_data,
                    {
                        "brand_names": label.brand_names,
                        "generic_names": label.generic_names,
                    },
                )
            session.commit()
            offset += batch_size
            self.stats["labels"] += len(labels)
        return count


def build_chunks(
    max_tokens: int = 512,
    overlap: int = 32,
    trial_section_overlap: int = 40,
    label_section_overlap: int = 40,
    model_name: str | None = None,
    batch_size: int = 1000,
    output_dir: Path = Path("results/chunking"),
) -> dict[str, int]:
    """Build chunks and write a small immutable result summary."""
    config = ChunkConfig(
        max_tokens=max_tokens,
        overlap=overlap,
        trial_section_overlap=trial_section_overlap,
        label_section_overlap=label_section_overlap,
    )
    chunker = Chunker(config=config, model_name=model_name)
    total = chunker.build_chunks(batch_size=batch_size)
    chunker.stats["total"] = total

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"chunks_{timestamp}.json"
    path.write_text(
        json.dumps({"timestamp": timestamp, **chunker.stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("chunking_complete", total=total, result=str(path))
    return chunker.stats
