"""Training data generation for embedding fine-tuning."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, SplitType
from bioevidence.db.session import get_session

logger = get_config().__class__.__module__  # placeholder
import structlog
logger = structlog.get_logger(__name__)


@dataclass
class Pair:
    """Query-positive pair for training."""
    query: str
    positive: str
    hard_negatives: list[str]
    query_type: str  # "factual", "structured", "clinician"
    doc_type: str


def load_chunks_by_type(doc_type: DocType, split: SplitType = SplitType.TRAIN, limit: int | None = None) -> list[Chunk]:
    """Load chunks from database."""
    with get_session() as session:
        query = session.query(Chunk).filter(
            Chunk.doc_type == doc_type,
            Chunk.split == split,
        )
        if limit:
            query = query.limit(limit)
        return query.all()


def generate_synthetic_pairs(
    num_pairs: int = 20000,
    hard_negatives_per_query: int = 3,
    seed: int = 42,
) -> list[Pair]:
    """Generate synthetic query-positive pairs from TRAIN chunks.

    Uses templates to create queries from chunk content.
    """
    random.seed(seed)

    # Load chunks from TRAIN split
    trial_chunks = load_chunks_by_type(DocType.TRIAL, limit=5000)
    pubmed_chunks = load_chunks_by_type(DocType.PUBMED, limit=5000)
    label_chunks = load_chunks_by_type(DocType.LABEL, limit=2000)

    all_chunks = {
        "trial": trial_chunks,
        "pubmed": pubmed_chunks,
        "label": label_chunks,
    }

    pairs = []

    # Distribute pairs across doc types
    per_type = num_pairs // 3

    for doc_type, chunks in all_chunks.items():
        if not chunks:
            continue

        random.shuffle(chunks)

        for i, chunk in enumerate(chunks[:per_type]):
            if len(pairs) >= num_pairs:
                break

            # Generate query from chunk
            query, query_type = _generate_query_from_chunk(chunk, doc_type)

            # Positive is the chunk text itself
            positive = chunk.text

            # Select hard negatives from same doc_type (different chunks)
            negatives = _select_hard_negatives(chunk, chunks, hard_negatives_per_query, doc_type)

            pairs.append(Pair(
                query=query,
                positive=positive,
                hard_negatives=negatives,
                query_type=query_type,
                doc_type=doc_type,
            ))

    logger.info("generated_synthetic_pairs", count=len(pairs))
    return pairs


def _generate_query_from_chunk(chunk: Chunk, doc_type: str) -> tuple[str, str]:
    """Generate a query from chunk content using templates."""
    templates = {
        "trial": [
            ("What were the primary outcomes of the {nct_id} trial?", "structured"),
            ("What is the eligibility criteria for {nct_id}?", "structured"),
            ("What interventions were tested in {nct_id}?", "factual"),
            ("In patients with {condition}, what is the efficacy of {intervention}?", "clinician"),
        ],
        "pubmed": [
            ("What did the study find about {topic}?", "factual"),
            ("What are the results of the {topic} study?", "factual"),
            ("What is the evidence for {intervention} in {condition}?", "clinician"),
        ],
        "label": [
            ("What are the contraindications for {drug}?", "factual"),
            ("What is the dosage for {drug}?", "factual"),
            ("What are the adverse reactions of {drug}?", "factual"),
            ("What are the boxed warnings for {drug}?", "clinician"),
        ],
    }

    template, q_type = random.choice(templates.get(doc_type, [("What is {topic}?", "factual")]))

    # Extract entities from chunk
    entities = _extract_entities(chunk, doc_type)

    query = template.format(**entities)
    return query, q_type


def _extract_entities(chunk: Chunk, doc_type: str) -> dict[str, str]:
    """Extract entities from chunk for query templating."""
    metadata = chunk.metadata or {}
    section = chunk.section or ""
    text = chunk.text[:200]  # First 200 chars

    entities = {
        "nct_id": metadata.get("nct_id", "the trial"),
        "topic": section.replace("_", " ") if section else "the study",
        "condition": "the condition",
        "intervention": "the intervention",
        "drug": "the drug",
    }

    # Try to extract from text
    import re
    if doc_type == "label":
        # Look for drug names
        for word in text.split():
            if word[0].isupper() and len(word) > 3:
                entities["drug"] = word
                break
    elif doc_type == "trial":
        # Look for condition keywords
        for word in text.split():
            if word.lower() in ["cancer", "tumor", "arthritis", "psoriasis", "heart", "failure", "osteoporosis"]:
                entities["condition"] = word
                break

    return entities


def _select_hard_negatives(
    anchor: Chunk,
    candidates: list[Chunk],
    k: int,
    doc_type: str,
) -> list[str]:
    """Select hard negatives from same doc_type.

    In production, use a retriever to find false positives.
    For now, random sampling from different documents.
    """
    # Filter out chunks from same document
    other_chunks = [c for c in candidates if c.doc_id != anchor.doc_id]

    if not other_chunks:
        return []

    # Sample hard negatives
    selected = random.sample(other_chunks, min(k, len(other_chunks)))
    return [c.text for c in selected]


def save_pairs(pairs: list[Pair], path: Path) -> None:
    """Save pairs to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps({
                "query": pair.query,
                "positive": pair.positive,
                "hard_negatives": pair.hard_negatives,
                "query_type": pair.query_type,
                "doc_type": pair.doc_type,
            }) + "\n")


def load_pairs(path: Path) -> list[Pair]:
    """Load pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            pairs.append(Pair(**data))
    return pairs


def main():
    """Generate and save training pairs."""
    config = get_config()
    data_dir = Path(config.llm.cache_dir).parent.parent / "processed"

    pairs = generate_synthetic_pairs(
        num_pairs=20000,
        hard_negatives_per_query=3,
        seed=42,
    )

    save_pairs(pairs, data_dir / "synth_pairs.jsonl")
    logger.info("saved_pairs", path=str(data_dir / "synth_pairs.jsonl"))


if __name__ == "__main__":
    main()