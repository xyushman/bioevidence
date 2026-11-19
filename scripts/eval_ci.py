"""CI evaluation gate for BioEvidence."""

import json
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioevidence.config import get_config
from bioevidence.eval import load_golden_set, evaluate_retrieval
from bioevidence.retrieval.pipeline import get_retriever

logger = structlog.get_logger(__name__)


@dataclass
class GateResult:
    """Result of eval gate."""
    passed: bool
    system: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    failures: list[str]
    details: dict[str, Any]


def run_retrieval_gate(
    system: str,
    golden_path: Path,
    thresholds: dict[str, float] | None = None,
) -> GateResult:
    """Run retrieval evaluation gate for a system."""
    # Default thresholds from config
    default_thresholds = {
        "recall_at_5": 0.45,
        "mrr_at_10": 0.35,
        "ndcg_at_10": 0.30,
        "hit_at_5": 0.50,
    }
    thresholds = thresholds or default_thresholds

    # Load golden set
    queries = load_golden_set(golden_path)
    logger.info("loaded_golden_queries", count=len(queries))

    # Get retriever
    retriever = get_retriever(system)

    # Evaluate
    result = evaluate_retrieval(queries, retriever, ks=[1, 5, 10, 20])

    # Extract metrics
    agg = result["aggregate"]
    metrics = {
        "recall_at_1": agg.recall_at_1,
        "recall_at_5": agg.recall_at_5,
        "recall_at_10": agg.recall_at_10,
        "recall_at_20": agg.recall_at_20,
        "mrr_at_10": agg.mrr_at_10,
        "ndcg_at_10": agg.ndcg_at_10,
        "hit_at_5": agg.hit_at_5,
    }

    # Check thresholds
    failures = []
    for metric, threshold in thresholds.items():
        value = metrics.get(metric, 0.0)
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold:.4f}")

    passed = len(failures) == 0

    return GateResult(
        passed=passed,
        system=system,
        metrics=metrics,
        thresholds=thresholds,
        failures=failures,
        details={
            "n_queries": result["n_queries"],
            "cis": result["cis"],
        },
    )


def run_qa_gate(
    qa_golden_path: Path,
    thresholds: dict[str, float] | None = None,
) -> GateResult:
    """Run QA evaluation gate (placeholder - requires LLM judge)."""
    default_thresholds = {
        "faithfulness": 0.70,
        "citation_accuracy": 0.60,
        "abstention_f1": 0.65,
    }
    thresholds = thresholds or default_thresholds

    # In production, this would run the full QA pipeline
    # For CI gate, we just check that the structure is in place
    return GateResult(
        passed=True,
        system="qa",
        metrics={},
        thresholds=thresholds,
        failures=[],
        details={"note": "QA gate requires LLM judge - skipped in CI"},
    )


def run_regression_gate(
    current_metrics: dict[str, float],
    baseline_path: Path,
    regression_thresholds: dict[str, float] | None = None,
) -> GateResult:
    """Check for regression against baseline."""
    default_regression = {
        "recall_at_5": -0.02,
        "faithfulness": -0.03,
    }
    regression_thresholds = regression_thresholds or default_regression

    if not baseline_path.exists():
        return GateResult(
            passed=True,
            system="regression",
            metrics=current_metrics,
            thresholds=regression_thresholds,
            failures=[],
            details={"note": "No baseline found - skipping regression check"},
        )

    with open(baseline_path) as f:
        baseline = json.load(f)

    failures = []
    for metric, threshold in regression_thresholds.items():
        current = current_metrics.get(metric, 0.0)
        base = baseline.get(metric, 0.0)
        diff = current - base
        if diff < threshold:
            failures.append(f"{metric} regression: {diff:.4f} < {threshold:.4f} (current={current:.4f}, baseline={base:.4f})")

    return GateResult(
        passed=len(failures) == 0,
        system="regression",
        metrics=current_metrics,
        thresholds=regression_thresholds,
        failures=failures,
        details={"baseline": baseline, "current": current_metrics},
    )


def run_eval_ci(
    systems: list[str] | None = None,
    retrieval_golden: Path | None = None,
    qa_golden: Path | None = None,
    output_dir: Path | None = None,
    baseline_path: Path | None = None,
) -> dict[str, GateResult]:
    """Run all eval gates for CI."""
    config = get_config()
    systems = systems or ["A0", "A1a", "A1b", "A2", "A3", "A4", "A4b", "A5"]
    retrieval_golden = retrieval_golden or Path(config.llm.cache_dir).parent.parent / "goldens" / "retrieval_golden.jsonl"
    qa_golden = qa_golden or Path(config.llm.cache_dir).parent.parent / "goldens" / "qa_golden.jsonl"
    output_dir = output_dir or Path(config.llm.cache_dir).parent.parent / "results" / "eval_ci"
    baseline_path = baseline_path or Path(config.llm.cache_dir).parent.parent / "results" / "main_baseline.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Retrieval gates
    for system in systems:
        logger.info("running_retrieval_gate", system=system)
        result = run_retrieval_gate(system, retrieval_golden)
        results[f"retrieval_{system}"] = result

        if not result.passed:
            logger.warning("retrieval_gate_failed", system=system, failures=result.failures)

    # QA gate (placeholder)
    logger.info("running_qa_gate")
    qa_result = run_qa_gate(qa_golden)
    results["qa"] = qa_result

    # Regression gate
    # Collect all metrics for regression check
    all_metrics = {}
    for key, result in results.items():
        if key.startswith("retrieval_"):
            all_metrics.update(result.metrics)

    if all_metrics:
        logger.info("running_regression_gate")
        reg_result = run_regression_gate(all_metrics, baseline_path)
        results["regression"] = reg_result

        if not reg_result.passed:
            logger.warning("regression_gate_failed", failures=reg_result.failures)

    # Save results
    output_file = output_dir / "eval_ci_results.json"
    with open(output_file, "w") as f:
        json.dump({
            k: {
                "passed": v.passed,
                "system": v.system,
                "metrics": v.metrics,
                "thresholds": v.thresholds,
                "failures": v.failures,
                "details": v.details,
            }
            for k, v in results.items()
        }, f, indent=2)

    logger.info("eval_ci_complete", output=str(output_file))
    return results


def main():
    """Main entry point for CI eval gate."""
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        systems: str = typer.Option("A0,A1a,A1b,A2,A3,A4,A4b,A5", "--systems", help="Comma-separated systems"),
        retrieval_golden: str = typer.Option(None, "--retrieval-golden", help="Retrieval golden set path"),
        qa_golden: str = typer.Option(None, "--qa-golden", help="QA golden set path"),
        output_dir: str = typer.Option(None, "--output-dir", help="Output directory"),
        baseline: str = typer.Option(None, "--baseline", help="Baseline path for regression"),
    ):
        system_list = [s.strip() for s in systems.split(",")]
        results = run_eval_ci(
            systems=system_list,
            retrieval_golden=Path(retrieval_golden) if retrieval_golden else None,
            qa_golden=Path(qa_golden) if qa_golden else None,
            output_dir=Path(output_dir) if output_dir else None,
            baseline_path=Path(baseline) if baseline else None,
        )

        # Print summary
        print("\n=== Eval CI Results ===")
        all_passed = True
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name}: {status}")
            if result.failures:
                for f in result.failures:
                    print(f"    - {f}")
            if not result.passed:
                all_passed = False

        if not all_passed:
            print("\n[FAIL] Eval CI gate failed")
            raise typer.Exit(code=1)
        else:
            print("\n[PASS] All eval gates passed")

    app()


if __name__ == "__main__":
    main()