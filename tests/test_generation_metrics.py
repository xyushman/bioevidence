"""Tests for generation/QA evaluation metrics."""

from bioevidence.eval.generation_metrics import (
    GenerationMetrics,
    LLMJudge,
    QAEvalResult,
    aggregate_qa_metrics,
    evaluate_qa,
)
from bioevidence.llm import create_stub_client


class TestLLMJudge:
    """Tests for LLMJudge."""

    def test_faithfulness_evaluation(self):
        """Test faithfulness evaluation with stub."""
        stub_responses = {
            "default": '{"score": 0.8, "reasoning": "Most claims supported"}',
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        chunks = {1: "Drug A treats cancer.", 2: "Drug A has side effects."}
        score, reason = judge.evaluate_faithfulness(
            query="What does Drug A treat?",
            answer="Drug A treats cancer.",
            citations=[1],
            chunks=chunks,
        )
        assert score == 0.8
        assert "supported" in reason

    def test_citation_accuracy(self):
        """Test citation accuracy evaluation."""
        stub_responses = {
            "default": '{"score": 1.0, "reasoning": "All citations accurate"}',
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        chunks = {1: "Drug A treats cancer."}
        score, reason = judge.evaluate_citation_accuracy(
            query="What does Drug A treat?",
            answer="Drug A treats cancer.",
            citations=[1],
            chunks=chunks,
        )
        assert score == 1.0

    def test_abstention_evaluation_answerable(self):
        """Test abstention evaluation for answerable question."""
        stub_responses = {
            "default": '{"score": 1.0, "reasoning": "Provided answer"}',
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        chunks = {1: "Drug A treats cancer."}
        score, reason = judge.evaluate_abstention(
            query="What does Drug A treat?",
            answer="Drug A treats cancer.",
            citations=[1],
            chunks=chunks,
            answerable=True,
        )
        assert score == 1.0

    def test_abstention_evaluation_unanswerable(self):
        """Test abstention evaluation for unanswerable question."""
        stub_responses = {
            "default": '{"score": 1.0, "reasoning": "Correctly abstained"}',
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        chunks = {}
        score, reason = judge.evaluate_abstention(
            query="What is the cure for everything?",
            answer="Insufficient evidence in the retrieved sources.",
            citations=[],
            chunks=chunks,
            answerable=False,
        )
        assert score == 1.0

    def test_answer_relevance(self):
        """Test answer relevance evaluation."""
        stub_responses = {
            "default": '{"score": 1.0, "reasoning": "Fully addresses question"}',
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        score, reason = judge.evaluate_answer_relevance(
            query="What does Drug A treat?",
            answer="Drug A treats cancer.",
        )
        assert score == 1.0

    def test_json_parse_failure(self):
        """Test handling of invalid JSON from judge."""
        stub_responses = {
            "default": "not valid json",
        }
        llm = create_stub_client(stub_responses)
        judge = LLMJudge(llm)

        chunks = {1: "Drug A treats cancer."}
        score, reason = judge.evaluate_faithfulness(
            query="What does Drug A treat?",
            answer="Drug A treats cancer.",
            citations=[1],
            chunks=chunks,
        )
        assert score == 0.0
        assert "Failed to parse" in reason


class TestEvaluateQA:
    """Tests for evaluate_qa function."""

    def test_evaluate_qa_full(self):
        """Test full QA evaluation."""
        stub_responses = {
            "default": '{"score": 0.9, "reasoning": "Good"}',
        }
        llm = create_stub_client(stub_responses)

        chunks = {1: "Drug A treats cancer.", 2: "Drug A has side effects."}
        result = evaluate_qa(
            query="What does Drug A treat?",
            gold_answer="Drug A treats cancer.",
            pred_answer='{"answer": "Drug A treats cancer", "citations": [1]}',
            gold_citations=[1],
            pred_citations=[1],
            chunks=chunks,
            answerable=True,
            judge=LLMJudge(llm),
        )

        assert isinstance(result, QAEvalResult)
        assert result.metrics.faithfulness == 0.9
        assert result.metrics.citation_accuracy == 0.9
        assert result.metrics.abstention_f1 == 0.9
        assert result.metrics.answer_relevance == 0.9
        assert result.metrics.valid_json is True
        assert result.answerable is True

    def test_valid_json_check(self):
        """Test valid JSON detection."""
        stub_responses = {
            "default": '{"score": 0.5, "reasoning": "OK"}',
        }
        llm = create_stub_client(stub_responses)

        chunks = {1: "Drug A treats cancer."}
        result = evaluate_qa(
            query="What does Drug A treat?",
            gold_answer="Drug A treats cancer.",
            pred_answer='{"answer": "Drug A treats cancer", "citations": [1]}',
            gold_citations=[1],
            pred_citations=[1],
            chunks=chunks,
            answerable=True,
            judge=LLMJudge(llm),
        )
        assert result.metrics.valid_json is True

    def test_invalid_json(self):
        """Test invalid JSON detection."""
        stub_responses = {
            "default": '{"score": 0.5, "reasoning": "OK"}',
        }
        llm = create_stub_client(stub_responses)

        chunks = {1: "Drug A treats cancer."}
        result = evaluate_qa(
            query="What does Drug A treat?",
            gold_answer="Drug A treats cancer.",
            pred_answer="not json",
            gold_citations=[1],
            pred_citations=[1],
            chunks=chunks,
            answerable=True,
            judge=LLMJudge(llm),
        )
        assert result.metrics.valid_json is False


class TestAggregateQAMetrics:
    """Tests for aggregate_qa_metrics."""

    def test_aggregate(self):
        """Test aggregating QA metrics."""
        results = [
            QAEvalResult(
                query="q1",
                gold_answer="a1",
                pred_answer="p1",
                gold_citations=[1],
                pred_citations=[1],
                answerable=True,
                metrics=GenerationMetrics(
                    faithfulness=1.0,
                    citation_accuracy=1.0,
                    abstention_f1=1.0,
                    answer_relevance=1.0,
                ),
            ),
            QAEvalResult(
                query="q2",
                gold_answer="a2",
                pred_answer="p2",
                gold_citations=[2],
                pred_citations=[2],
                answerable=False,
                metrics=GenerationMetrics(
                    faithfulness=0.0,
                    citation_accuracy=0.0,
                    abstention_f1=1.0,
                    answer_relevance=0.0,
                ),
            ),
        ]

        agg = aggregate_qa_metrics(results)

        assert agg["faithfulness"] == 0.5
        assert agg["citation_accuracy"] == 0.5
        assert agg["abstention_f1"] == 1.0
        assert agg["answer_relevance"] == 0.5
        assert agg["answerable_count"] == 1
        assert agg["unanswerable_count"] == 1
        assert agg["total"] == 2

    def test_empty(self):
        """Test aggregate with empty list."""
        agg = aggregate_qa_metrics([])
        assert agg == {}
