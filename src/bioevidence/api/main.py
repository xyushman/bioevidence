"""FastAPI application for BioEvidence API."""

import structlog
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bioevidence.agent import BioEvidenceAgent, AgentResponse, create_agent
from bioevidence.config import get_config
from bioevidence.retrieval.pipeline import Hit, get_retriever

logger = structlog.get_logger(__name__)

# Global agent instance
_agent: BioEvidenceAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _agent
    config = get_config()
    _agent = create_agent(system=config.retrieval.system or "A5")
    logger.info("api_startup", system=config.retrieval.system)
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="BioEvidence API",
    description="Biomedical Evidence Copilot API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., min_length=1, max_length=2000)
    system: str | None = Field(None, pattern="^(A0|A1a|A1b|A2|A3|A4|A4b|A5)$")


class QuestionResponse(BaseModel):
    """Response model for question answer."""
    answer: str
    citations: list[int]
    tools_used: list[str]
    sql_queries: list[str]
    confident: bool
    steps: int


class SearchRequest(BaseModel):
    """Request model for search."""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=50)
    filters: dict[str, Any] | None = None
    system: str | None = Field(None, pattern="^(A0|A1a|A1b|A2|A3|A4|A4b|A5)$")


class SearchResponse(BaseModel):
    """Response model for search."""
    hits: list[dict]
    query: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the BioEvidence agent."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Use requested system or default
    system = request.system or "A5"
    agent = create_agent(system=system) if system != "A5" else _agent

    try:
        response = agent.answer(request.question)
        return QuestionResponse(
            answer=response.answer,
            citations=response.citations,
            tools_used=response.tools_used,
            sql_queries=response.sql_queries,
            confident=response.confident,
            steps=response.steps,
        )
    except Exception as e:
        logger.exception("ask_question_failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_evidence(request: SearchRequest):
    """Search the evidence corpus."""
    system = request.system or "A5"
    retriever = get_retriever(system)

    try:
        hits = retriever.search(request.query, k=request.k, filters=request.filters)
        return SearchResponse(
            hits=[
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "text": hit.text,
                    "metadata": hit.metadata,
                    "doc_type": hit.doc_type.value,
                    "doc_id": hit.doc_id,
                    "section": hit.section,
                }
                for hit in hits
            ],
            query=request.query,
        )
    except Exception as e:
        logger.exception("search_failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/systems")
async def list_systems():
    """List available retrieval systems."""
    return {
        "systems": [
            {"id": "A0", "name": "Baseline Dense", "description": "Off-the-shelf bge-small, dense only"},
            {"id": "A1a", "name": "BM25 Only", "description": "Postgres FTS only"},
            {"id": "A1b", "name": "Dense Only", "description": "Off-the-shelf bge-small, dense only"},
            {"id": "A2", "name": "Hybrid (RRF)", "description": "BM25 + dense with RRF fusion"},
            {"id": "A3", "name": "Fine-tuned Embeddings", "description": "Hybrid + fine-tuned bge-small embeddings"},
            {"id": "A4", "name": "Off-the-shelf Reranker", "description": "Fine-tuned embeddings + hybrid + bge-reranker-base"},
            {"id": "A4b", "name": "Fine-tuned Reranker", "description": "Fine-tuned embeddings + hybrid + fine-tuned MiniLM reranker"},
            {"id": "A5", "name": "Full System", "description": "Fine-tuned embeddings + hybrid + fine-tuned reranker + metadata filters"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)