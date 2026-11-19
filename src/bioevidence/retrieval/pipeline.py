"""Retrieval pipeline for BioEvidence."""

from dataclasses import dataclass
from typing import Any

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType
from bioevidence.db.session import get_session

logger = get_config().__class__.__module__  # placeholder
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Hit:
    """A retrieval hit."""

    chunk_id: int
    score: float
    text: str
    metadata: dict[str, Any]
    doc_type: DocType
    doc_id: str
    section: str | None = None


class BaseRetriever:
    """Base class for retrievers."""

    def __init__(self, config=None):
        self.config = config or get_config()

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[Hit]:
        """Search for relevant chunks."""
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    """Dense vector retrieval using pgvector."""

    def __init__(self, config=None, embedding_model: str | None = None):
        super().__init__(config)
        self.embedding_model = embedding_model or self.config.embedding.model
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query."""
        embedder = self._get_embedder()
        return embedder.encode(query, normalize_embeddings=True).tolist()

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[Hit]:
        """Search using dense vectors."""
        query_embedding = self._embed_query(query)

        with get_session() as session:
            # Build filter conditions
            filter_conditions = []
            if filters:
                if "doc_type" in filters:
                    filter_conditions.append(Chunk.doc_type == filters["doc_type"])
                if "phase" in filters:
                    filter_conditions.append(Chunk.metadata["phase"].astext == filters["phase"])
                if "year_min" in filters:
                    filter_conditions.append(
                        Chunk.metadata["year"].astext.cast(int) >= filters["year_min"]
                    )
                if "year_max" in filters:
                    filter_conditions.append(
                        Chunk.metadata["year"].astext.cast(int) <= filters["year_max"]
                    )

            # Query using cosine similarity
            from pgvector.sqlalchemy import cosine_distance
            from sqlalchemy import select

            stmt = select(Chunk).where(*filter_conditions) if filter_conditions else select(Chunk)
            stmt = stmt.order_by(cosine_distance(Chunk.embedding, query_embedding)).limit(k)

            chunks = session.execute(stmt).scalars().all()

            hits = []
            for chunk in chunks:
                hits.append(
                    Hit(
                        chunk_id=chunk.id,
                        score=1.0 - cosine_distance(chunk.embedding, query_embedding),
                        text=chunk.text,
                        metadata=chunk.metadata or {},
                        doc_type=chunk.doc_type,
                        doc_id=chunk.doc_id,
                        section=chunk.section,
                    )
                )

            return hits


class BM25Retriever(BaseRetriever):
    """BM25 retrieval using Postgres FTS."""

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[Hit]:
        """Search using BM25/tsvector."""
        with get_session() as session:
            from sqlalchemy import func, select

            # Build filter conditions
            filter_conditions = []
            if filters:
                if "doc_type" in filters:
                    filter_conditions.append(Chunk.doc_type == filters["doc_type"])
                if "phase" in filters:
                    filter_conditions.append(Chunk.metadata["phase"].astext == filters["phase"])
                if "year_min" in filters:
                    filter_conditions.append(
                        Chunk.metadata["year"].astext.cast(int) >= filters["year_min"]
                    )
                if "year_max" in filters:
                    filter_conditions.append(
                        Chunk.metadata["year"].astext.cast(int) <= filters["year_max"]
                    )

            # Query using ts_rank_cd
            stmt = select(
                Chunk,
                func.ts_rank_cd(Chunk.tsv, func.plainto_tsquery("english", query)).label("rank"),
            ).where(Chunk.tsv.op("@@")(func.plainto_tsquery("english", query)))

            if filter_conditions:
                stmt = stmt.where(*filter_conditions)

            stmt = stmt.order_by(
                func.ts_rank_cd(Chunk.tsv, func.plainto_tsquery("english", query)).desc()
            ).limit(k)

            results = session.execute(stmt).all()

            hits = []
            for chunk, rank in results:
                hits.append(
                    Hit(
                        chunk_id=chunk.id,
                        score=float(rank),
                        text=chunk.text,
                        metadata=chunk.metadata or {},
                        doc_type=chunk.doc_type,
                        doc_id=chunk.doc_id,
                        section=chunk.section,
                    )
                )

            return hits


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval with RRF fusion."""

    def __init__(
        self,
        config=None,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ):
        super().__init__(config)
        self.dense = dense_retriever or DenseRetriever(config)
        self.bm25 = bm25_retriever or BM25Retriever(config)
        self.rrf_k = config.retrieval.rrf_k if config else 60

    def _rrf_fuse(self, dense_hits: list[Hit], bm25_hits: list[Hit], k: int) -> list[Hit]:
        """Fuse results using Reciprocal Rank Fusion."""
        # Build rank maps
        dense_ranks = {hit.chunk_id: i + 1 for i, hit in enumerate(dense_hits)}
        bm25_ranks = {hit.chunk_id: i + 1 for i, hit in enumerate(bm25_hits)}

        # All unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build fused hits
        chunk_map = {hit.chunk_id: hit for hit in dense_hits + bm25_hits}
        fused_hits = []
        for chunk_id in sorted_chunk_ids[:k]:
            if chunk_id in chunk_map:
                hit = chunk_map[chunk_id]
                fused_hits.append(
                    Hit(
                        chunk_id=hit.chunk_id,
                        score=rrf_scores[chunk_id],
                        text=hit.text,
                        metadata=hit.metadata,
                        doc_type=hit.doc_type,
                        doc_id=hit.doc_id,
                        section=hit.section,
                    )
                )

        return fused_hits

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[Hit]:
        """Search using hybrid RRF."""
        # Retrieve more candidates for fusion
        candidate_pool = self.config.retrieval.candidate_pool if self.config else 50

        dense_hits = self.dense.search(query, k=candidate_pool, filters=filters)
        bm25_hits = self.bm25.search(query, k=candidate_pool, filters=filters)

        return self._rrf_fuse(dense_hits, bm25_hits, k)


def get_retriever(system: str, config=None) -> BaseRetriever:
    """Factory function to get retriever by system name."""
    config = config or get_config()

    if system in ["A0", "A1b", "A2", "A3", "A4", "A4b", "A5"]:
        if system == "A0":
            # Baseline: dense only with off-the-shelf bge-small
            return DenseRetriever(config, embedding_model="BAAI/bge-small-en-v1.5")
        elif system == "A1b":
            # Dense only (same as A0 but explicit)
            return DenseRetriever(config, embedding_model="BAAI/bge-small-en-v1.5")
        elif system == "A2":
            # Hybrid with off-the-shelf bge-small
            return HybridRetriever(
                config, DenseRetriever(config, embedding_model="BAAI/bge-small-en-v1.5")
            )
        elif system in ["A3", "A4", "A4b", "A5"]:
            # Fine-tuned embeddings (placeholder - would use fine-tuned model)
            return HybridRetriever(
                config, DenseRetriever(config, embedding_model="bioevidence/bge-small-ft")
            )
    elif system == "A1a":
        # BM25 only
        return BM25Retriever(config)

    raise ValueError(f"Unknown system: {system}")
