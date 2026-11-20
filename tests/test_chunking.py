"""Tests for deterministic chunking."""

from __future__ import annotations

from bioevidence.chunking import (
    ChunkConfig,
    count_tokens,
    split_drug_label,
    split_pubmed_article,
    split_text_by_paragraphs,
    split_text_by_tokens,
)
from bioevidence.db.models import DrugLabel, PubMedArticle

FALLBACK_MODEL = "deterministic-test-model"


def test_count_tokens_uses_offline_fallback() -> None:
    assert count_tokens("alpha beta gamma", FALLBACK_MODEL) == 3
    assert count_tokens("alpha, beta!", FALLBACK_MODEL) == 4


def test_split_text_by_tokens_is_bounded_and_deterministic() -> None:
    text = " ".join(f"word{index}" for index in range(100))
    first = split_text_by_tokens(text, max_tokens=12, overlap=3, model_name=FALLBACK_MODEL)
    second = split_text_by_tokens(text, max_tokens=12, overlap=3, model_name=FALLBACK_MODEL)

    assert first == second
    assert len(first) > 1
    assert all(count_tokens(chunk, FALLBACK_MODEL) <= 12 for chunk in first)


def test_long_drug_label_sections_are_split_safely() -> None:
    label = DrugLabel(
        set_id="chunk-test-label",
        sections={
            "warnings_and_cautions": " ".join(f"warning{index}" for index in range(80)),
        },
        source_url="https://example.com/label",
    )
    chunks = split_drug_label(label, ChunkConfig(max_tokens=16, overlap=2), FALLBACK_MODEL)

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 16 for chunk in chunks)
    assert all(chunk["text"].startswith("[Drug Label") for chunk in chunks)
    assert chunks[0]["section"].startswith("warnings_and_cautions_part")


def test_chunk_limits_are_document_specific() -> None:
    config = ChunkConfig()

    assert config.limit_for("trial") == 512
    assert config.limit_for("pubmed") == 256
    assert config.limit_for("label") == 350
    assert ChunkConfig(max_tokens=16).limit_for("label") == 16


def test_pubmed_chunks_respect_smaller_limit() -> None:
    article = PubMedArticle(
        pmid="chunk-pubmed",
        title="Long article",
        abstract=" ".join(f"term{index}" for index in range(400)),
        source_url="https://example.com/pubmed",
    )
    chunks = split_pubmed_article(article, ChunkConfig(), FALLBACK_MODEL)

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 256 for chunk in chunks)


def test_paragraph_splitting_preserves_boundaries() -> None:
    text = "alpha beta gamma delta\n\nepsilon zeta eta theta"
    chunks = split_text_by_paragraphs(text, max_tokens=6, overlap=1, model_name=FALLBACK_MODEL)

    assert len(chunks) == 2
    assert chunks[0].startswith("alpha")
    assert "epsilon" in chunks[1]
    assert all(count_tokens(chunk, FALLBACK_MODEL) <= 6 for chunk in chunks)
