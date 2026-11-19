"""Tests for retrieval metrics."""

from bioevidence.eval.retrieval_metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    bootstrap_ci,
    compute_metrics,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from bioevidence.retrieval.pipeline import Hit


class TestRecallAtK:
    """Tests for recall@k."""

    def test_perfect_recall(self):
        """Test recall when all relevant are retrieved."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_recall(self):
        """Test recall when none relevant are retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self):
        """Test partial recall."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, 5) == 0.75  # 3/4

    def test_recall_at_1(self):
        """Test recall@1."""
        # Only 1 of 2 relevant items in top-1
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 1) == 0.5

        # 1 of 1 relevant in top-1
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert recall_at_k(retrieved, relevant, 1) == 1.0

    def test_empty_relevant(self):
        """Test recall with empty relevant set."""
        retrieved = ["a", "b", "c"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 5) == 0.0

    def test_k_larger_than_retrieved(self):
        """Test recall when k > len(retrieved)."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 5) == 2.0 / 3.0


class TestMRRAtK:
    """Tests for MRR@k."""

    def test_first_relevant(self):
        """Test MRR when first result is relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mrr_at_k(retrieved, relevant, 3) == 1.0

    def test_second_relevant(self):
        """Test MRR when second result is relevant."""
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert mrr_at_k(retrieved, relevant, 3) == 0.5

    def test_no_relevant(self):
        """Test MRR when no relevant in top-k."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mrr_at_k(retrieved, relevant, 3) == 0.0

    def test_multiple_relevant(self):
        """Test MRR with multiple relevant (should use first)."""
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert mrr_at_k(retrieved, relevant, 3) == 0.5


class TestNDCGAtK:
    """Tests for nDCG@k."""

    def test_perfect_ndcg(self):
        """Test nDCG when all relevant at top."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_ndcg(self):
        """Test nDCG with partial relevance."""
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        ndcg = ndcg_at_k(retrieved, relevant, 3)
        assert 0.9 < ndcg < 1.0

    def test_empty_relevant(self):
        """Test nDCG with empty relevant."""
        retrieved = ["a", "b"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0


class TestHitAtK:
    """Tests for Hit@k."""

    def test_hit(self):
        """Test hit when relevant found."""
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        assert hit_at_k(retrieved, relevant, 3) == 1.0

    def test_no_hit(self):
        """Test no hit."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert hit_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self):
        """Test hit with empty relevant."""
        retrieved = ["a", "b"]
        relevant = set()
        assert hit_at_k(retrieved, relevant, 2) == 0.0


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_all_metrics(self):
        """Test computing all metrics."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}

        metrics = compute_metrics(retrieved, relevant, ks=[1, 5, 10])

        # 1 of 3 relevant in top-1
        assert metrics.recall_at_1 == 1.0 / 3.0
        # All 3 of 3 relevant in top-5
        assert metrics.recall_at_5 == 1.0
        assert metrics.recall_at_10 == 1.0
        assert metrics.mrr_at_10 == 1.0  # "a" at position 1
        assert metrics.hit_at_5 == 1.0
        assert metrics.recall_at_5 == 1.0  # all 3 in top 5
        assert metrics.recall_at_10 == 1.0
        assert metrics.mrr_at_10 == 1.0
        assert metrics.hit_at_5 == 1.0


class TestAggregateMetrics:
    """Tests for aggregate_metrics."""

    def test_aggregate(self):
        """Test aggregating multiple metrics."""
        m1 = RetrievalMetrics(recall_at_5=1.0, mrr_at_10=1.0)
        m2 = RetrievalMetrics(recall_at_5=0.5, mrr_at_10=0.5)
        m3 = RetrievalMetrics(recall_at_5=0.0, mrr_at_10=0.0)

        agg = aggregate_metrics([m1, m2, m3])

        assert agg.recall_at_5 == (1.0 + 0.5 + 0.0) / 3
        assert agg.mrr_at_10 == (1.0 + 0.5 + 0.0) / 3

    def test_empty(self):
        """Test aggregate with empty list."""
        agg = aggregate_metrics([])
        assert agg.recall_at_5 == 0.0
        assert agg.mrr_at_10 == 0.0


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        metrics_list = [
            RetrievalMetrics(recall_at_5=1.0),
            RetrievalMetrics(recall_at_5=0.8),
            RetrievalMetrics(recall_at_5=0.6),
            RetrievalMetrics(recall_at_5=0.4),
            RetrievalMetrics(recall_at_5=0.2),
        ]

        lower, upper = bootstrap_ci(metrics_list, "recall_at_5", n_bootstrap=100)

        assert 0.0 <= lower <= 0.6
        assert 0.6 <= upper <= 1.0


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval (integration test with mock retriever)."""

    def test_evaluate_with_mock_retriever(self):
        """Test evaluate_retrieval with a mock retriever."""

        class MockRetriever:
            def search(self, query, k=5):
                # Return fixed results based on query
                if "cancer" in query.lower():
                    return [Hit(i, 1.0, "", {}, "trial", f"doc{i}") for i in [1, 2, 3]]
                return [Hit(i, 1.0, "", {}, "trial", f"doc{i}") for i in [4, 5, 6]]

        from bioevidence.eval.retrieval_metrics import evaluate_retrieval

        queries = [
            {"query": "cancer treatment", "gold_chunk_ids": {1, 2}},
            {"query": "heart disease", "gold_chunk_ids": {4, 5}},
        ]

        retriever = MockRetriever()
        result = evaluate_retrieval(queries, retriever, ks=[5])

        assert result["n_queries"] == 2
        assert "aggregate" in result
        assert "cis" in result
