"""Tests for configuration."""

from pathlib import Path

import yaml

from bioevidence.config import Config, load_config


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    assert config.seed == 42
    assert config.llm.provider == "gemini"
    assert config.embedding.model == "BAAI/bge-small-en-v1.5"
    assert config.embedding.dimension == 384


def test_config_from_yaml(tmp_path: Path):
    """Test loading configuration from YAML."""
    config_data = {
        "seed": 123,
        "llm": {
            "provider": "anthropic",
            "gemini": {"model": "gemini-1.5-flash"},
            "anthropic": {"model": "claude-3-haiku-20240307", "api_key": "test-key"},
            "openai": {"model": "gpt-4o-mini"},
        },
        "embedding": {"model": "test-model", "dimension": 768},
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.seed == 123
    assert config.llm.provider == "anthropic"
    assert config.embedding.model == "test-model"
    assert config.embedding.dimension == 768


def test_config_env_override(monkeypatch):
    """Test environment variable overrides for top-level fields."""
    monkeypatch.setenv("BIOEVIDENCE_SEED", "999")

    # Need to clear the lru_cache
    from bioevidence.config import load_config

    load_config.cache_clear()

    config = load_config()
    assert config.seed == 999


def test_config_nested_env(monkeypatch, tmp_path: Path):
    """Test nested environment variable interpolation."""
    config_data = {
        "database": {"url": "${DATABASE_URL}"},
        "llm": {
            "provider": "gemini",
            "gemini": {"api_key": "${GEMINI_API_KEY}", "model": "gemini-1.5-flash"},
            "anthropic": {"api_key": "${ANTHROPIC_API_KEY}", "model": "claude-3-haiku-20240307"},
            "openai": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4o-mini"},
        },
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    from bioevidence.config import load_config

    load_config.cache_clear()

    config = load_config(config_file)
    assert config.database.url == "postgresql://test:test@localhost/test"
    assert config.llm.gemini.api_key == "test-gemini-key"
