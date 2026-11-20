"""Benchmark script for extraction latency and cost."""

import json
import statistics
import structlog
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, SplitType, Trial
from bioevidence.db.session import get_session
from bioevidence.scripts.prepare_extraction_data import build_trial_text, load_trial_data
from bioevidence.extraction import extract_trial_fields, evaluate_extraction, aggregate_extraction_results

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of extraction benchmark."""
    n_trials: int
    method: str
    model_path: str | None
    total_time_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    valid_json_rate: float
    macro_f1: float | None = None
    cost_per_1k_usd: float | None = None


def run_extraction_benchmark(
    method: str = "lora",
    model_path: str | None = None,
    num_trials: int = 100,
    split: SplitType = SplitType.TEST,
    seed: int = 42,
) -> BenchmarkResult:
    """Run extraction benchmark on TEST split trials."""
    import random
    random.seed(seed)

    with get_session() as session:
        trial_ids = session.query(Trial.nct_id).join(Chunk, Trial.nct_id == Chunk.doc_id).filter(
            Chunk.split == split,
            Chunk.doc_type == DocType.TRIAL,
        ).distinct().all()

        trial_ids = [t[0] for t in trial_ids]
        random.shuffle(trial_ids)
        trial_ids = trial_ids[:num_trials]

        logger.info("benchmark_trials_selected", count=len(trial_ids))

    latencies = []
    predictions = []
    ground_truths = []
    valid_json_count = 0

    for i, nct_id in enumerate(trial_ids):
        trial_data = load_trial_data(nct_id)
        if not trial_data:
            continue

        trial_text = build_trial_text(trial_data)
        target = trial_data  # For ground truth, we'd need the extracted target

        # Time the extraction
        start = time.perf_counter()
        try:
            result = extract_trial_fields(trial_text, method=method, model_path=model_path)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Check valid JSON
            try:
                json.dumps(result)
                valid_json_count += 1
            except (TypeError, ValueError):
                pass

            predictions.append(result)

        except Exception as e:
            logger.warning("extraction_failed", nct_id=nct_id, error=str(e))
            latencies.append((time.perf_counter() - start) * 1000)
            predictions.append({})

    # Calculate metrics
    total_time = sum(latencies) / 1000
    avg_latency = statistics.mean(latencies) if latencies else 0
    p50_latency = statistics.median(latencies) if latencies else 0
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies) if latencies else 0

    valid_json_rate = valid_json_count / len(trial_ids) if trial_ids else 0

    # Estimate cost per 1k trials
    # Rough estimate: 4-bit 3B model on T4 ~$0.50/hr, ~100 trials/hr = $5/1k
    # This is a placeholder - real cost depends on GPU, provider, etc.
    estimated_cost_per_1k = estimate_cost(method, model_path, avg_latency)

    result = BenchmarkResult(
        n_trials=len(trial_ids),
        method=method,
        model_path=model_path,
        total_time_s=total_time,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        valid_json_rate=valid_json_rate,
        cost_per_1k_usd=estimated_cost_per_1k,
    )

    logger.info("benchmark_complete", **result.__dict__)
    return result


def estimate_cost(method: str, model_path: str | None, avg_latency_ms: float) -> float:
    """Estimate cost per 1000 trials in USD."""
    # Rough estimates based on typical GPU pricing
    # T4 GPU: ~$0.35/hr (spot) to $0.50/hr (on-demand)
    # A10G: ~$0.75/hr
    # A100: ~$2.00/hr

    if method == "constrained":
        # Outlines with transformers - similar to base model
        gpu_hourly = 0.50
    elif method == "lora":
        # 4-bit 3B model - can run on T4
        gpu_hourly = 0.50
    else:
        gpu_hourly = 1.00

    # Trials per hour
    trials_per_hour = (3600 * 1000) / max(avg_latency_ms, 1)
    cost_per_1k = (gpu_hourly / trials_per_hour) * 1000

    return cost_per_1k


def run_full_benchmark_suite(output_dir: Path | None = None) -> list[BenchmarkResult]:
    """Run benchmark suite for all extraction methods."""
    config = get_config()
    output_dir = output_dir or Path(config.llm.cache_dir).parent.parent / "results" / "extraction_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Test LoRA method
    logger.info("benchmarking_lora")
    lora_result = run_extraction_benchmark(
        method="lora",
        model_path="models/qwen2.5-3b-lora-extractor",
        num_trials=50,
    )
    results.append(lora_result)

    # Test constrained method (if outlines available)
    try:
        import outlines
        logger.info("benchmarking_constrained")
        constrained_result = run_extraction_benchmark(
            method="constrained",
            model_path="Qwen/Qwen2.5-3B-Instruct",
            num_trials=50,
        )
        results.append(constrained_result)
    except ImportError:
        logger.info("outlines_not_available_skipping_constrained")

    # Save results
    output_file = output_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    logger.info("benchmark_suite_complete", output=str(output_file), results=len(results))
    return results


def main():
    """Main entry point for extraction benchmark."""
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        method: str = typer.Option("lora", "--method", help="Extraction method: lora, constrained"),
        model_path: str = typer.Option(None, "--model-path", help="Model path for LoRA"),
        num_trials: int = typer.Option(100, "--num-trials", help="Number of trials to benchmark"),
        output: str = typer.Option("results/extraction_benchmark.json", "--output", help="Output file"),
    ):
        config = get_config()
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = run_extraction_benchmark(
            method=method,
            model_path=model_path,
            num_trials=num_trials,
        )

        with open(output_path, "w") as f:
            json.dump(result.__dict__, f, indent=2)

        print(f"Benchmark complete:")
        print(f"  Trials: {result.n_trials}")
        print(f"  Method: {result.method}")
        print(f"  Avg latency: {result.avg_latency_ms:.1f} ms")
        print(f"  P50 latency: {result.p50_latency_ms:.1f} ms")
        print(f"  P95 latency: {result.p95_latency_ms:.1f} ms")
        print(f"  P99 latency: {result.p99_latency_ms:.1f} ms")
        print(f"  Valid JSON rate: {result.valid_json_rate:.2%}")
        print(f"  Est. cost/1k: ${result.cost_per_1k_usd:.2f}")

    app()


if __name__ == "__main__":
    main()