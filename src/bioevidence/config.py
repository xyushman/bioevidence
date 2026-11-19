"""Configuration management for BioEvidence."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScopeConfig(BaseModel):
    therapeutic_areas: list[dict[str, Any]] = Field(default_factory=list)
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


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"

    config_data: dict[str, Any] = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Environment variable interpolation for nested configs
    import os

    def interpolate_env(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: interpolate_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [interpolate_env(v) for v in obj]
        return obj

    config_data = interpolate_env(config_data)
    return Config(**config_data)


def get_config() -> Config:
    """Get the global configuration instance."""
    return load_config()
