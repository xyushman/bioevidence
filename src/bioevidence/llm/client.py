"""LLM Client for BioEvidence with pluggable providers and disk caching."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from bioevidence.config import Config, get_config

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    model: str = ""
    latency_ms: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 8192):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError as e:
                raise RuntimeError(
                    "google-generativeai not installed. Run: uv add google-generativeai"
                ) from e
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        client = self._get_client()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        try:
            response = client.generate_content(full_prompt, generation_config=generation_config)
            text = response.text or ""
            latency_ms = (time.perf_counter() - start) * 1000

            # Token counting is approximate for Gemini
            input_tokens = len(full_prompt) // 4
            output_tokens = len(text) // 4

            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("gemini_completion_failed")
            raise

    def get_model_name(self) -> str:
        return self.model


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise RuntimeError("anthropic not installed. Run: uv add anthropic") from e
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        client = self._get_client()

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.content[0].text if response.content else ""
            latency_ms = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=self.model,
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("anthropic_completion_failed")
            raise

    def get_model_name(self) -> str:
        return self.model


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise RuntimeError("openai not installed. Run: uv add openai") from e
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        client = self._get_client()

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
            latency_ms = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=text,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=self.model,
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("openai_completion_failed")
            raise

    def get_model_name(self) -> str:
        return self.model


class StubProvider(LLMProvider):
    """Stub provider for testing (no network calls)."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_log: list[dict[str, Any]] = []

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        self.call_log.append({"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs})

        # Check for exact match first
        if prompt in self.responses:
            return LLMResponse(
                text=self.responses[prompt],
                input_tokens=len(prompt) // 4,
                output_tokens=len(self.responses[prompt]) // 4,
                model="stub",
                latency_ms=1.0,
                cached=False,
            )

        # Return a generic response based on prompt content
        if "sql" in prompt.lower():
            return LLMResponse(
                text='{"answer": "Found 5 trials", "sql": "SELECT COUNT(*) FROM trials WHERE phase ILIKE \'%Phase 3%\'", "citations": []}',
                model="stub",
                latency_ms=1.0,
            )
        if "extract" in prompt.lower():
            return LLMResponse(
                text='{"primary_endpoint": "Overall Survival", "endpoint_category": "overall_survival", "intervention_names": ["Drug A"], "intervention_class": "small_molecule", "randomized": true, "blinding": "double", "population_age_group": "adult", "key_inclusion": ["Age >= 18"], "key_exclusion": ["Pregnancy"]}',
                model="stub",
                latency_ms=1.0,
            )

        return LLMResponse(
            text=self.responses.get("default", "Stub response"),
            model="stub",
            latency_ms=1.0,
        )

    def get_model_name(self) -> str:
        return "stub"


class LLMCache:
    """Disk-based cache for LLM calls."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir) if cache_dir != ":memory:" else None
        self._memory_cache: dict[str, LLMResponse] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, model: str, prompt: str, system_prompt: str | None, params: dict[str, Any]
    ) -> str:
        """Create a cache key from model, prompt, and parameters."""
        content = f"{model}|{system_prompt or ''}|{prompt}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self, model: str, prompt: str, system_prompt: str | None, params: dict[str, Any]
    ) -> LLMResponse | None:
        """Get a cached response if available."""
        key = self._make_key(model, prompt, system_prompt, params)

        if self.cache_dir is None:
            # In-memory cache
            if key in self._memory_cache:
                self.hits += 1
                logger.debug("llm_cache_hit", key=key)
                cached = self._memory_cache[key]
                return LLMResponse(
                    text=cached.text,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    cached=True,
                    model=cached.model,
                    latency_ms=cached.latency_ms,
                )
            self.misses += 1
            return None

        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                self.hits += 1
                logger.debug("llm_cache_hit", key=key)
                return LLMResponse(
                    text=data["text"],
                    input_tokens=data.get("input_tokens", 0),
                    output_tokens=data.get("output_tokens", 0),
                    cached=True,
                    model=data.get("model", model),
                    latency_ms=data.get("latency_ms", 0.0),
                )
            except Exception as e:
                logger.warning("llm_cache_read_failed", key=key, error=str(e))

        self.misses += 1
        return None

    def set(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        params: dict[str, Any],
        response: LLMResponse,
    ) -> None:
        """Cache a response."""
        key = self._make_key(model, prompt, system_prompt, params)

        if self.cache_dir is None:
            self._memory_cache[key] = response
            return

        cache_file = self.cache_dir / f"{key}.json"

        try:
            data = {
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "cached_at": time.time(),
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.debug("llm_cache_write", key=key)
        except Exception as e:
            logger.warning("llm_cache_write_failed", key=key, error=str(e))

    def stats(self) -> dict[str, float]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses),
        }


class LLMClient:
    """Main LLM client with provider selection and caching."""

    def __init__(
        self,
        config: Config | None = None,
        cache: LLMCache | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.config = config or get_config()
        self.cache = cache or LLMCache(self.config.llm.cache_dir)
        self.provider = provider or self._create_provider()

    def _create_provider(self) -> LLMProvider:
        """Create the configured provider."""
        provider_name = self.config.llm.provider.lower()
        llm_config = self.config.llm

        if provider_name == "gemini":
            return GeminiProvider(
                api_key=llm_config.gemini.api_key,
                model=llm_config.gemini.model,
                temperature=llm_config.gemini.temperature,
                max_tokens=llm_config.gemini.max_tokens,
            )
        elif provider_name == "anthropic":
            return AnthropicProvider(
                api_key=llm_config.anthropic.api_key,
                model=llm_config.anthropic.model,
                temperature=llm_config.anthropic.temperature,
                max_tokens=llm_config.anthropic.max_tokens,
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                api_key=llm_config.openai.api_key,
                model=llm_config.openai.model,
                temperature=llm_config.openai.temperature,
                max_tokens=llm_config.openai.max_tokens,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with optional caching."""
        model_name = self.provider.get_model_name()
        params: dict[str, Any] = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        # Check cache first
        if use_cache:
            cached = self.cache.get(model_name, prompt, system_prompt, params)
            if cached:
                return cached

        # Call provider
        response = self.provider.complete(prompt, system_prompt, **kwargs)

        # Cache the response
        if use_cache:
            self.cache.set(model_name, prompt, system_prompt, params, response)

        return response

    def get_cache_stats(self) -> dict[str, float]:
        """Get cache statistics."""
        return self.cache.stats()


def create_stub_client(responses: dict[str, str] | None = None) -> LLMClient:
    """Create a client with a stub provider for testing."""
    return LLMClient(provider=StubProvider(responses), cache=LLMCache(":memory:"))
