"""Tests for LLM client."""

from bioevidence.llm.client import (
    LLMCache,
    LLMClient,
    LLMResponse,
    StubProvider,
    create_stub_client,
)


class TestStubProvider:
    """Tests for the stub provider."""

    def test_stub_basic_response(self):
        """Test basic stub response."""
        provider = StubProvider({"hello": "world"})
        response = provider.complete("hello")
        assert response.text == "world"
        assert response.model == "stub"

    def test_stub_default_response(self):
        """Test default response when prompt not found."""
        provider = StubProvider({"default": "default response"})
        response = provider.complete("unknown prompt")
        assert response.text == "default response"

    def test_stub_sql_response(self):
        """Test SQL-like response."""
        provider = StubProvider()
        response = provider.complete("Generate SQL query")
        assert "sql" in response.text.lower()
        assert "SELECT" in response.text.upper()

    def test_stub_extract_response(self):
        """Test extraction response."""
        provider = StubProvider()
        response = provider.complete("Extract trial fields")
        assert "primary_endpoint" in response.text
        assert "intervention_names" in response.text

    def test_stub_call_logging(self):
        """Test that calls are logged."""
        provider = StubProvider()
        provider.complete("test prompt")
        provider.complete("another prompt")
        assert len(provider.call_log) == 2
        assert provider.call_log[0]["prompt"] == "test prompt"


class TestLLMCache:
    """Tests for the LLM cache."""

    def test_cache_miss_then_hit(self, tmp_path):
        """Test cache miss followed by hit."""
        cache = LLMCache(tmp_path)
        response = LLMResponse(
            text="cached response", model="test-model", input_tokens=10, output_tokens=5
        )

        # First call - miss
        cached = cache.get("test-model", "prompt", None, {})
        assert cached is None
        assert cache.misses == 1

        # Store
        cache.set("test-model", "prompt", None, {}, response)

        # Second call - hit
        cached = cache.get("test-model", "prompt", None, {})
        assert cached is not None
        assert cached.text == "cached response"
        assert cached.cached is True
        assert cache.hits == 1

    def test_cache_different_prompts(self, tmp_path):
        """Test cache separates different prompts."""
        cache = LLMCache(tmp_path)
        response1 = LLMResponse(text="response 1", model="test")
        response2 = LLMResponse(text="response 2", model="test")

        cache.set("test", "prompt 1", None, {}, response1)
        cache.set("test", "prompt 2", None, {}, response2)

        cached1 = cache.get("test", "prompt 1", None, {})
        cached2 = cache.get("test", "prompt 2", None, {})

        assert cached1.text == "response 1"
        assert cached2.text == "response 2"

    def test_cache_different_params(self, tmp_path):
        """Test cache separates different parameters."""
        cache = LLMCache(tmp_path)
        response1 = LLMResponse(text="response 1", model="test")
        response2 = LLMResponse(text="response 2", model="test")

        cache.set("test", "prompt", None, {"temperature": 0.0}, response1)
        cache.set("test", "prompt", None, {"temperature": 1.0}, response2)

        cached1 = cache.get("test", "prompt", None, {"temperature": 0.0})
        cached2 = cache.get("test", "prompt", None, {"temperature": 1.0})

        assert cached1.text == "response 1"
        assert cached2.text == "response 2"

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache = LLMCache(tmp_path)
        response = LLMResponse(text="test", model="test")

        cache.get("test", "prompt1", None, {})  # miss
        cache.get("test", "prompt2", None, {})  # miss
        cache.set("test", "prompt1", None, {}, response)
        cache.get("test", "prompt1", None, {})  # hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1 / 3


class TestLLMClient:
    """Tests for the LLM client."""

    def test_client_with_stub(self):
        """Test client with stub provider."""
        client = create_stub_client({"test": "stub response"})
        response = client.complete("test")
        assert response.text == "stub response"
        assert response.model == "stub"

    def test_client_caching(self, tmp_path):
        """Test client caching works."""
        cache = LLMCache(tmp_path)
        provider = StubProvider({"cached prompt": "cached response"})
        client = LLMClient(provider=provider, cache=cache)

        # First call
        response1 = client.complete("cached prompt")
        assert response1.cached is False

        # Second call - should hit cache
        response2 = client.complete("cached prompt")
        assert response2.cached is True
        assert response2.text == "cached response"

    def test_client_no_cache(self, tmp_path):
        """Test client with caching disabled."""
        cache = LLMCache(tmp_path)
        provider = StubProvider({"test": "response"})
        client = LLMClient(provider=provider, cache=cache)

        response1 = client.complete("test", use_cache=False)
        response2 = client.complete("test", use_cache=False)

        assert response1.cached is False
        assert response2.cached is False
        # When use_cache=False, cache is not accessed, so misses/hits stay at 0
        assert cache.misses == 0
        assert cache.hits == 0

    def test_client_cache_stats(self, tmp_path):
        """Test client cache stats."""
        cache = LLMCache(tmp_path)
        provider = StubProvider({"test": "response"})
        client = LLMClient(provider=provider, cache=cache)

        client.complete("test")  # miss
        client.complete("test")  # hit

        stats = client.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
