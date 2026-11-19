"""Generation/QA evaluation metrics for BioEvidence."""

import json
from dataclasses import dataclass

import structlog

from bioevidence.llm import LLMClient, create_stub_client

logger = structlog.get_logger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for generated answer."""

    faithfulness: float = 0.0  # Claims supported by citations
    citation_accuracy: float = 0.0  # Citations actually support claims
    abstention_f1: float = 0.0  # Correctly abstains when no evidence
    answer_relevance: float = 0.0  # Answer addresses the question
    valid_json: bool = True  # Response is valid JSON
    latency_ms: float = 0.0


@dataclass
class QAEvalResult:
    """Result of QA evaluation."""

    query: str
    gold_answer: str
    pred_answer: str
    gold_citations: list[int]
    pred_citations: list[int]
    answerable: bool
    metrics: GenerationMetrics
    judge_reasoning: str = ""


class LLMJudge:
    """LLM-based judge for evaluating generated answers."""

    FAITHFULNESS_PROMPT = """You are an evaluator assessing the FAITHFULNESS of a generated answer against its citations.

Question: {query}

Generated Answer: {answer}
Cited Chunk IDs: {citations}
Retrieved Chunks:
{chunk_texts}

A faithful answer ONLY makes claims that are directly supported by the cited chunks.
Score 1.0 if all claims are supported, 0.0 if any claim is unsupported or contradicted.

Return JSON: {{"score": float, "reasoning": "..."}}"""

    CITATION_ACCURACY_PROMPT = """You are an evaluator assessing CITATION ACCURACY.

Question: {query}

Generated Answer: {answer}
Cited Chunk IDs: {citations}
Retrieved Chunks:
{chunk_texts}

For each citation, does the chunk actually contain information supporting the claim it's cited for?
Score = (number of accurate citations) / (total citations). If no citations, score 0.

Return JSON: {{"score": float, "reasoning": "..."}}"""

    ABSTENTION_PROMPT = """You are an evaluator assessing whether the system correctly ABSTAINS when evidence is insufficient.

Question: {query}

Generated Answer: {answer}
Cited Chunk IDs: {citations}
Retrieved Chunks:
{chunk_texts}

Gold Standard: {answerable}

If unanswerable: score 1.0 if answer says "Insufficient evidence" or similar, 0.0 otherwise.
If answerable: score 1.0 if answer provides a substantive response, 0.0 if it incorrectly abstains.

Return JSON: {{"score": float, "reasoning": "..."}}"""

    ANSWER_RELEVANCE_PROMPT = """You are an evaluator assessing ANSWER RELEVANCE.

Question: {query}

Generated Answer: {answer}

Does the answer actually address the question asked?
Score 1.0 if fully addresses, 0.5 if partially addresses, 0.0 if irrelevant.

Return JSON: {{"score": float, "reasoning": "..."}}"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or create_stub_client()

    def _format_chunks(self, chunk_ids: list[int], all_chunks: dict[int, str]) -> str:
        """Format chunks for prompt."""
        lines = []
        for cid in chunk_ids:
            if cid in all_chunks:
                lines.append(f"[Chunk {cid}] {all_chunks[cid][:500]}")
        return "\n".join(lines) if lines else "No chunks provided."

    def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        citations: list[int],
        chunks: dict[int, str],
    ) -> tuple[float, str]:
        """Evaluate faithfulness of answer to citations."""
        prompt = self.FAITHFULNESS_PROMPT.format(
            query=query,
            answer=answer,
            citations=citations,
            chunk_texts=self._format_chunks(citations, chunks),
        )
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text)
            return float(result.get("score", 0.0)), result.get("reasoning", "")
        except json.JSONDecodeError:
            return 0.0, "Failed to parse judge response"

    def evaluate_citation_accuracy(
        self,
        query: str,
        answer: str,
        citations: list[int],
        chunks: dict[int, str],
    ) -> tuple[float, str]:
        """Evaluate accuracy of citations."""
        prompt = self.CITATION_ACCURACY_PROMPT.format(
            query=query,
            answer=answer,
            citations=citations,
            chunk_texts=self._format_chunks(citations, chunks),
        )
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text)
            return float(result.get("score", 0.0)), result.get("reasoning", "")
        except json.JSONDecodeError:
            return 0.0, "Failed to parse judge response"

    def evaluate_abstention(
        self,
        query: str,
        answer: str,
        citations: list[int],
        chunks: dict[int, str],
        answerable: bool,
    ) -> tuple[float, str]:
        """Evaluate abstention correctness."""
        prompt = self.ABSTENTION_PROMPT.format(
            query=query,
            answer=answer,
            citations=citations,
            chunk_texts=self._format_chunks(citations, chunks),
            answerable="Answerable" if answerable else "Unanswerable - should abstain",
        )
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text)
            return float(result.get("score", 0.0)), result.get("reasoning", "")
        except json.JSONDecodeError:
            return 0.0, "Failed to parse judge response"

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, str]:
        """Evaluate answer relevance to question."""
        prompt = self.ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text)
            return float(result.get("score", 0.0)), result.get("reasoning", "")
        except json.JSONDecodeError:
            return 0.0, "Failed to parse judge response"


def evaluate_qa(
    query: str,
    gold_answer: str,
    pred_answer: str,
    gold_citations: list[int],
    pred_citations: list[int],
    chunks: dict[int, str],
    answerable: bool,
    judge: LLMJudge | None = None,
) -> QAEvalResult:
    """Evaluate a QA prediction against gold standard."""
    judge = judge or LLMJudge()

    # Faithfulness
    faithfulness, f_reason = judge.evaluate_faithfulness(query, pred_answer, pred_citations, chunks)

    # Citation accuracy
    citation_acc, c_reason = judge.evaluate_citation_accuracy(
        query, pred_answer, pred_citations, chunks
    )

    # Abstention
    abstention, a_reason = judge.evaluate_abstention(
        query, pred_answer, pred_citations, chunks, answerable
    )

    # Answer relevance
    relevance, r_reason = judge.evaluate_answer_relevance(query, pred_answer)

    # Valid JSON check
    try:
        json.loads(pred_answer)
        valid_json = True
    except json.JSONDecodeError:
        valid_json = False

    metrics = GenerationMetrics(
        faithfulness=faithfulness,
        citation_accuracy=citation_acc,
        abstention_f1=abstention,
        answer_relevance=relevance,
        valid_json=valid_json,
    )

    return QAEvalResult(
        query=query,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        gold_citations=gold_citations,
        pred_citations=pred_citations,
        answerable=answerable,
        metrics=metrics,
        judge_reasoning=f"Faithfulness: {f_reason}\nCitation: {c_reason}\nAbstention: {a_reason}\nRelevance: {r_reason}",
    )


def aggregate_qa_metrics(results: list[QAEvalResult]) -> dict[str, float]:
    """Aggregate QA metrics across multiple questions."""
    if not results:
        return {}

    n = len(results)
    return {
        "faithfulness": sum(r.metrics.faithfulness for r in results) / n,
        "citation_accuracy": sum(r.metrics.citation_accuracy for r in results) / n,
        "abstention_f1": sum(r.metrics.abstention_f1 for r in results) / n,
        "answer_relevance": sum(r.metrics.answer_relevance for r in results) / n,
        "valid_json_rate": sum(1 for r in results if r.metrics.valid_json) / n,
        "answerable_count": sum(1 for r in results if r.answerable),
        "unanswerable_count": sum(1 for r in results if not r.answerable),
        "total": n,
    }
