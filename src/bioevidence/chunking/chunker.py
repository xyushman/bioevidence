"""Chunker for BioEvidence documents."""

import re
from dataclasses import dataclass

from transformers import AutoTokenizer

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, DrugLabel, PubMedArticle, SplitType, Trial
from bioevidence.db.session import get_session

# Load tokenizer for token counting
_tokenizer = None


def get_tokenizer(model_name: str | None = None):
    """Get or create tokenizer for token counting."""
    global _tokenizer
    if _tokenizer is None:
        config = get_config()
        model = model_name or config.embedding.model
        _tokenizer = AutoTokenizer.from_pretrained(model)
    return _tokenizer


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens in text using the embedding model tokenizer."""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


@dataclass
class ChunkConfig:
    """Configuration for chunking."""

    max_tokens: int = 512
    overlap: int = 32
    trial_section_overlap: int = 40
    label_section_overlap: int = 40


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap: int,
    model_name: str | None = None,
) -> list[str]:
    """Split text into chunks by token count with overlap."""
    if not text:
        return []

    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap

    return chunks


def split_trial_sections(
    trial: Trial, config: ChunkConfig, model_name: str | None = None
) -> list[dict]:
    """Split trial into section-based chunks."""
    chunks = []

    # Build header prefix
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

    for section_name, text in sections:
        if not text:
            continue

        full_text = header + text
        token_count = count_tokens(full_text, model_name)

        if token_count <= config.max_tokens:
            chunks.append(
                {
                    "section": section_name,
                    "text": full_text,
                    "token_count": token_count,
                }
            )
        else:
            # Split long sections
            sub_chunks = split_text_by_tokens(
                full_text,
                config.max_tokens,
                config.trial_section_overlap,
                model_name,
            )
            for i, sub_chunk in enumerate(sub_chunks):
                chunks.append(
                    {
                        "section": f"{section_name}_part{i + 1}"
                        if len(sub_chunks) > 1
                        else section_name,
                        "text": sub_chunk,
                        "token_count": count_tokens(sub_chunk, model_name),
                    }
                )

    return chunks


def extract_inclusion_criteria(eligibility_text: str | None) -> str | None:
    """Extract inclusion criteria from eligibility text."""
    if not eligibility_text:
        return None

    # Look for inclusion criteria section
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

    parts = []
    for outcome in trial.outcomes:
        kind_str = "Primary" if outcome.kind.value == "primary" else "Secondary"
        parts.append(f"{kind_str} Outcome: {outcome.measure}")
        if outcome.time_frame:
            parts.append(f"  Time Frame: {outcome.time_frame}")
        if outcome.description:
            parts.append(f"  Description: {outcome.description}")

    return "\n".join(parts) if parts else None


def split_pubmed_article(
    article: PubMedArticle, config: ChunkConfig, model_name: str | None = None
) -> list[dict]:
    """Split PubMed article into chunks (title + abstract as one chunk)."""
    if not article.abstract:
        return []

    text = f"Title: {article.title}\n\nAbstract: {article.abstract}"
    token_count = count_tokens(text, model_name)

    if token_count <= config.max_tokens:
        return [
            {
                "section": "title_abstract",
                "text": text,
                "token_count": token_count,
            }
        ]
    else:
        # Split long abstract
        chunks = split_text_by_tokens(text, config.max_tokens, config.overlap, model_name)
        return [
            {
                "section": f"title_abstract_part{i + 1}",
                "text": chunk,
                "token_count": count_tokens(chunk, model_name),
            }
            for i, chunk in enumerate(chunks)
        ]


def split_drug_label(
    label: DrugLabel, config: ChunkConfig, model_name: str | None = None
) -> list[dict]:
    """Split drug label by sections."""
    chunks = []

    for section_name, section_text in (label.sections or {}).items():
        if not section_text:
            continue

        header = f"[Drug Label {label.set_id} | Section: {section_name}] "
        full_text = header + section_text
        token_count = count_tokens(full_text, model_name)

        if token_count <= config.max_tokens:
            chunks.append(
                {
                    "section": section_name,
                    "text": full_text,
                    "token_count": token_count,
                }
            )
        else:
            # Split long sections at paragraph boundaries
            paragraphs = section_text.split("\n\n")
            current_chunk = header
            current_tokens = count_tokens(current_chunk, model_name)

            for para in paragraphs:
                para_tokens = count_tokens(para, model_name)
                if current_tokens + para_tokens > config.max_tokens and current_chunk != header:
                    chunks.append(
                        {
                            "section": section_name,
                            "text": current_chunk,
                            "token_count": current_tokens,
                        }
                    )
                    current_chunk = header + para
                    current_tokens = count_tokens(current_chunk, model_name)
                else:
                    current_chunk += "\n\n" + para if current_chunk != header else para
                    current_tokens = count_tokens(current_chunk, model_name)

            if current_chunk != header:
                chunks.append(
                    {
                        "section": section_name,
                        "text": current_chunk,
                        "token_count": current_tokens,
                    }
                )

    return chunks


class Chunker:
    """Main chunker class."""

    def __init__(self, config: ChunkConfig | None = None, model_name: str | None = None):
        self.config = config or ChunkConfig()
        self.model_name = model_name
        self.stats = {"total": 0, "trials": 0, "pubmed": 0, "labels": 0}

    def build_chunks(self, batch_size: int = 1000) -> int:
        """Build chunks for all documents."""
        total_chunks = 0

        with get_session() as session:
            # Process trials
            total_chunks += self._chunk_trials(session, batch_size)

            # Process PubMed articles
            total_chunks += self._chunk_pubmed(session, batch_size)

            # Process drug labels
            total_chunks += self._chunk_labels(session, batch_size)

        return total_chunks

    def _chunk_trials(self, session, batch_size: int) -> int:
        """Chunk trials."""
        count = 0
        offset = 0

        while True:
            trials = session.query(Trial).offset(offset).limit(batch_size).all()
            if not trials:
                break

            for trial in trials:
                # Delete existing chunks for this trial
                session.query(Chunk).filter(
                    Chunk.doc_type == DocType.TRIAL,
                    Chunk.doc_id == trial.nct_id,
                ).delete()

                trial_chunks = split_trial_sections(trial, self.config, self.model_name)
                for chunk_data in trial_chunks:
                    chunk = Chunk(
                        doc_type=DocType.TRIAL,
                        doc_id=trial.nct_id,
                        section=chunk_data["section"],
                        text=chunk_data["text"],
                        token_count=chunk_data["token_count"],
                        chunk_metadata={
                            "phase": trial.phase,
                            "status": trial.overall_status,
                            "start_year": trial.start_date.year if trial.start_date else None,
                        },
                        split=SplitType.TRAIN,  # Will be updated by splits make
                    )
                    session.add(chunk)
                    count += 1

            session.commit()
            offset += batch_size
            self.stats["trials"] += len(trials)

        return count

    def _chunk_pubmed(self, session, batch_size: int) -> int:
        """Chunk PubMed articles."""
        count = 0
        offset = 0

        while True:
            articles = session.query(PubMedArticle).offset(offset).limit(batch_size).all()
            if not articles:
                break

            for article in articles:
                session.query(Chunk).filter(
                    Chunk.doc_type == DocType.PUBMED,
                    Chunk.doc_id == article.pmid,
                ).delete()

                article_chunks = split_pubmed_article(article, self.config, self.model_name)
                for chunk_data in article_chunks:
                    chunk = Chunk(
                        doc_type=DocType.PUBMED,
                        doc_id=article.pmid,
                        section=chunk_data["section"],
                        text=chunk_data["text"],
                        token_count=chunk_data["token_count"],
                        chunk_metadata={
                            "journal": article.journal,
                            "year": article.year,
                        },
                        split=SplitType.TRAIN,
                    )
                    session.add(chunk)
                    count += 1

            session.commit()
            offset += batch_size
            self.stats["pubmed"] += len(articles)

        return count

    def _chunk_labels(self, session, batch_size: int) -> int:
        """Chunk drug labels."""
        count = 0
        offset = 0

        while True:
            labels = session.query(DrugLabel).offset(offset).limit(batch_size).all()
            if not labels:
                break

            for label in labels:
                session.query(Chunk).filter(
                    Chunk.doc_type == DocType.LABEL,
                    Chunk.doc_id == label.set_id,
                ).delete()

                label_chunks = split_drug_label(label, self.config, self.model_name)
                for chunk_data in label_chunks:
                    chunk = Chunk(
                        doc_type=DocType.LABEL,
                        doc_id=label.set_id,
                        section=chunk_data["section"],
                        text=chunk_data["text"],
                        token_count=chunk_data["token_count"],
                        chunk_metadata={
                            "brand_names": label.brand_names,
                            "generic_names": label.generic_names,
                        },
                        split=SplitType.TRAIN,
                    )
                    session.add(chunk)
                    count += 1

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
) -> dict:
    """Main entry point for chunk building."""
    config = ChunkConfig(
        max_tokens=max_tokens,
        overlap=overlap,
        trial_section_overlap=trial_section_overlap,
        label_section_overlap=label_section_overlap,
    )
    chunker = Chunker(config=config, model_name=model_name)
    total = chunker.build_chunks(batch_size=batch_size)
    chunker.stats["total"] = total
    return chunker.stats
