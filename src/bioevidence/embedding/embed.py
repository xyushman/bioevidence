"""Embedding generation for BioEvidence."""

from typing import Any

import structlog
from sentence_transformers import SentenceTransformer

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)


class Embedder:
    """Generate and store embeddings for chunks."""

    def __init__(self, model_name: str | None = None, use_finetuned: bool = False):
        config = get_config()
        self.model_name = model_name or config.embedding.model
        self.use_finetuned = use_finetuned
        self.config = config
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_column(self) -> str:
        """Get the column name for this embedder."""
        return "embedding_ft" if self.use_finetuned else "embedding"

    def build_embeddings(
        self,
        batch_size: int = 64,
        doc_types: list[DocType] | None = None,
        split: str | None = None,
    ) -> dict[str, Any]:
        """Build embeddings for all chunks."""
        doc_types = doc_types or [DocType.TRIAL, DocType.PUBMED, DocType.LABEL]

        stats = {"total": 0, "by_type": {}}

        with get_session() as session:
            for doc_type in doc_types:
                # Build query
                query = session.query(Chunk).filter(Chunk.doc_type == doc_type)
                if split:
                    query = query.filter(Chunk.split == split)

                chunks = query.all()
                logger.info("embedding_chunks", doc_type=doc_type.value, count=len(chunks))

                if not chunks:
                    continue

                # Process in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    texts = [chunk.text for chunk in batch]

                    # Generate embeddings
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=self.config.embedding.normalize,
                        show_progress_bar=False,
                    )

                    # Store embeddings
                    for chunk, embedding in zip(batch, embeddings, strict=False):
                        setattr(chunk, self.embedding_column, embedding.tolist())

                    session.commit()

                stats["by_type"][doc_type.value] = len(chunks)
                stats["total"] += len(chunks)

        logger.info("embeddings_complete", stats=stats)
        return stats

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.model.encode(
            query,
            normalize_embeddings=self.config.embedding.normalize,
        ).tolist()


def build_embeddings(
    model: str | None = None,
    batch_size: int = 64,
    use_finetuned: bool = False,
) -> dict[str, Any]:
    """Main entry point for building embeddings."""
    embedder = Embedder(model_name=model, use_finetuned=use_finetuned)
    return embedder.build_embeddings(batch_size=batch_size)


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def build(
        model: str = typer.Option(None, "--model", help="Embedding model"),
        batch_size: int = typer.Option(64, "--batch-size", help="Batch size"),
        finetuned: bool = typer.Option(False, "--finetuned", help="Use fine-tuned column"),
    ):
        stats = build_embeddings(model=model, batch_size=batch_size, use_finetuned=finetuned)
        print(f"Embeddings built: {stats}")

    app()
