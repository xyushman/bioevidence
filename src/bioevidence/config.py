"""Configuration management for BioEvidence."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TherapeuticAreaConfig(BaseModel):
    """A configurable therapeutic scope entry."""

    name: str
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)


class ScopeConfig(BaseModel):
    therapeutic_areas: list[TherapeuticAreaConfig] = Field(default_factory=list)
    condition_keywords: list[str] = Field(default_factory=list)
    trial_filters: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, int] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


class LLMProviderConfig(BaseModel):
    api_key: str = ""
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    provider: str = "gemini"
    gemini: LLMProviderConfig
    anthropic: LLMProviderConfig
    openai: LLMProviderConfig
    cache_dir: str = "data/processed/llm_cache"


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 64
    normalize: bool = True


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    system: str = "A5"
    top_k: int = 5
    candidate_pool: int = 50
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class ChunkingConfig(BaseModel):
    max_tokens: int = 512
    overlap: int = 32
    trial_section_overlap: int = 40
    label_section_overlap: int = 40


class AgentConfig(BaseModel):
    max_steps: int = 6
    timeout_seconds: int = 30
    token_budget: int = 8000
    sql_timeout_seconds: int = 3
    sql_limit: int = 200


class ForecastingConfig(BaseModel):
    test_start_year: int = 2020
    max_duration_months: int = 180
    min_duration_months: int = 1


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class RateLimitsConfig(BaseModel):
    ncbi: int = 3
    ncbi_api_key: str = ""
    openfda: int = 10


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "bioevidence"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        nested_model_default_partial_update=True,
        env_nested_delimiter="__",
        env_prefix="BIOEVIDENCE_",
    )

    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig(url=""))
    llm: LLMConfig = Field(
        default_factory=lambda: LLMConfig(
            provider="gemini",
            gemini=LLMProviderConfig(model="gemini-1.5-flash"),
            anthropic=LLMProviderConfig(model="claude-3-haiku-20240307"),
            openai=LLMProviderConfig(model="gpt-4o-mini"),
        )
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    seed: int = 42
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


ENV_OVERRIDES: dict[str, str] = {
    "database.url": "DATABASE_URL",
    "llm.provider": "LLM_PROVIDER",
    "llm.gemini.api_key": "GEMINI_API_KEY",
    "llm.anthropic.api_key": "ANTHROPIC_API_KEY",
    "llm.openai.api_key": "OPENAI_API_KEY",
    "embedding.model": "EMBEDDING_MODEL",
    "embedding.dimension": "EMBEDDING_DIM",
    "reranker.model": "RERANKER_MODEL",
    "retrieval.top_k": "RETRIEVAL_TOP_K",
    "retrieval.candidate_pool": "RETRIEVAL_CANDIDATE_POOL",
    "retrieval.rrf_k": "RRF_K",
    "agent.max_steps": "AGENT_MAX_STEPS",
    "agent.timeout_seconds": "AGENT_TIMEOUT_SECONDS",
    "agent.token_budget": "AGENT_TOKEN_BUDGET",
    "forecasting.test_start_year": "FORECAST_TEST_START_YEAR",
    "api.host": "API_HOST",
    "api.port": "API_PORT",
    "api.cors_origins": "API_CORS_ORIGINS",
    "rate_limits.ncbi": "NCBI_RATE_LIMIT",
    "rate_limits.ncbi_api_key": "NCBI_API_KEY",
    "rate_limits.openfda": "OPENFDA_RATE_LIMIT",
    "seed": "SEED",
    "mlflow.tracking_uri": "MLFLOW_TRACKING_URI",
}


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a value to a dotted key in a nested dictionary."""
    parts = dotted_key.split(".")
    target = data
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _interpolate_env(obj: Any) -> Any:
    """Expand ``${VAR}`` placeholders recursively."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {key: _interpolate_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(value) for value in obj]
    return obj


def _coerce_env_value(raw_value: str) -> Any:
    """Parse JSON-like environment values and pass scalars through."""
    if raw_value.startswith(("[", "{")) or raw_value.lower() in {"true", "false", "null"}:
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            if raw_value.startswith("["):
                return [item.strip() for item in raw_value.strip("[]").split(",") if item.strip()]
    return raw_value


def _apply_env_overrides(config_data: dict[str, Any]) -> dict[str, Any]:
    """Apply the documented unprefixed environment variables."""
    for dotted_key, env_name in ENV_OVERRIDES.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        _set_nested(config_data, dotted_key, _coerce_env_value(raw_value))

    for field_name in Config.model_fields:
        env_name = f"BIOEVIDENCE_{field_name.upper()}"
        raw_value = os.getenv(env_name)
        if raw_value is not None and raw_value != "":
            config_data[field_name] = _coerce_env_value(raw_value)

        nested_prefix = f"BIOEVIDENCE_{field_name.upper()}__"
        for env_name, raw_value in os.environ.items():
            if not env_name.startswith(nested_prefix) or raw_value == "":
                continue
            nested_path = f"{field_name}.{env_name[len(nested_prefix) :].lower()}"
            _set_nested(config_data, nested_path, _coerce_env_value(raw_value))
    return config_data


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    default_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    resolved_path = Path(config_path) if config_path is not None else default_path

    config_data: dict[str, Any] = {}
    if resolved_path.exists():
        with resolved_path.open(encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}

    config_data = _apply_env_overrides(_interpolate_env(config_data))
    return Config(**config_data)


def get_config() -> Config:
    """Get the global configuration instance."""
    return load_config()
