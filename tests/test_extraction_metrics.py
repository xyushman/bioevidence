"""Tests for extraction metrics."""

from bioevidence.extraction.metrics import (
    ExtractionMetrics,
    ExtractionResult,
    aggregate_extraction_results,
    check_extraction_thresholds,
    compute_field_metrics,
    evaluate_extraction,
    evaluate_extraction_dataset,
    exact_match,
    set_f1,
    token_f1,
)


class TestTokenF1:
    """Tests for token_f1 function."""

    def test_perfect_match(self):
        assert token_f1("overall survival", "overall survival") == 1.0

    def test_no_match(self):
        assert token_f1("cancer", "heart failure") == 0.0

    def test_partial_match(self):
        # "overall survival" vs "progression free survival" - share "survival"
        score = token_f1("overall survival", "progression free survival")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert token_f1("", "") == 1.0
        assert token_f1("", "test") == 0.0
        assert token_f1("test", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("OVERALL SURVIVAL", "overall survival") == 1.0


class TestExactMatch:
    """Tests for exact_match function."""

    def test_strings_equal(self):
        assert exact_match("double", "double") == 1.0
        assert exact_match("double", "single") == 0.0

    def test_booleans(self):
        assert exact_match(True, True) == 1.0
        assert exact_match(True, False) == 0.0
        assert exact_match(False, False) == 1.0

    def test_mixed_types(self):
        assert exact_match("true", True) == 0.0


class TestSetF1:
    """Tests for set_f1 function."""

    def test_perfect_match(self):
        assert set_f1(["drug_a", "drug_b"], ["drug_a", "drug_b"]) == 1.0

    def test_no_match(self):
        assert set_f1(["drug_a"], ["drug_b"]) == 0.0

    def test_partial_match(self):
        # 1 out of 2 match
        score = set_f1(["drug_a", "drug_b"], ["drug_a", "drug_c"])
        assert score == 0.5  # precision=0.5, recall=0.5, f1=2*0.5*0.5/1.0=0.5

    def test_empty_lists(self):
        assert set_f1([], []) == 1.0
        assert set_f1([], ["drug_a"]) == 0.0
        assert set_f1(["drug_a"], []) == 0.0

    def test_case_insensitive(self):
        assert set_f1(["DRUG_A"], ["drug_a"]) == 1.0

    def test_extra_whitespace(self):
        assert set_f1([" drug_a "], ["drug_a"]) == 1.0


class TestComputeFieldMetrics:
    """Tests for compute_field_metrics function."""

    def test_primary_endpoint_token_f1(self):
        metrics = compute_field_metrics(
            "primary_endpoint",
            "Overall Survival",
            "Overall Survival",
        )
        assert metrics.token_f1 == 1.0
        assert metrics.exact_match == 1.0

    def test_endpoint_category_exact_match(self):
        metrics = compute_field_metrics(
            "endpoint_category",
            "overall_survival",
            "overall_survival",
        )
        assert metrics.exact_match == 1.0

    def test_endpoint_category_mismatch(self):
        metrics = compute_field_metrics(
            "endpoint_category",
            "overall_survival",
            "progression_free_survival",
        )
        assert metrics.exact_match == 0.0

    def test_randomized_boolean(self):
        metrics = compute_field_metrics("randomized", True, True)
        assert metrics.exact_match == 1.0

        metrics = compute_field_metrics("randomized", True, False)
        assert metrics.exact_match == 0.0

    def test_intervention_names_set_f1(self):
        metrics = compute_field_metrics(
            "intervention_names",
            ["pembrolizumab", "placebo"],
            ["pembrolizumab", "placebo"],
        )
        assert metrics.set_f1 == 1.0

    def test_key_inclusion_set_f1(self):
        metrics = compute_field_metrics(
            "key_inclusion",
            ["Age >= 18", "Cancer diagnosis"],
            ["Age >= 18", "Cancer diagnosis"],
        )
        assert metrics.set_f1 == 1.0


class TestEvaluateExtraction:
    """Tests for evaluate_extraction function."""

    def test_perfect_extraction(self):
        pred = {
            "primary_endpoint": "Overall Survival",
            "endpoint_category": "overall_survival",
            "intervention_names": ["pembrolizumab"],
            "intervention_class": "monoclonal_antibody",
            "randomized": True,
            "blinding": "double",
            "population_age_group": "adult",
            "key_inclusion": ["Age >= 18"],
            "key_exclusion": ["Pregnancy"],
        }
        gt = pred.copy()

        result = evaluate_extraction(pred, gt)

        assert result.macro_f1 == 1.0
        assert result.valid_json is True
        for field_metrics in result.field_metrics.values():
            # All fields should have perfect scores
            if field_metrics.token_f1 > 0:
                assert field_metrics.token_f1 == 1.0
            if field_metrics.exact_match > 0:
                assert field_metrics.exact_match == 1.0
            if field_metrics.set_f1 > 0:
                assert field_metrics.set_f1 == 1.0

    def test_partial_extraction(self):
        pred = {
            "primary_endpoint": "Overall Survival",
            "endpoint_category": "overall_survival",
            "intervention_names": ["pembrolizumab"],
            "intervention_class": "monoclonal_antibody",
            "randomized": True,
            "blinding": "double",
            "population_age_group": "adult",
            "key_inclusion": ["Age >= 18"],
            "key_exclusion": ["Pregnancy"],
        }
        gt = {
            "primary_endpoint": "Progression Free Survival",  # Different
            "endpoint_category": "progression_free_survival",  # Different
            "intervention_names": ["pembrolizumab"],  # Same
            "intervention_class": "monoclonal_antibody",  # Same
            "randomized": True,  # Same
            "blinding": "double",  # Same
            "population_age_group": "adult",  # Same
            "key_inclusion": ["Age >= 18"],  # Same
            "key_exclusion": ["Pregnancy"],  # Same
        }

        result = evaluate_extraction(pred, gt)

        # 7 out of 9 fields match perfectly, 2 differ
        assert result.macro_f1 < 1.0
        assert result.macro_f1 > 0.5

    def test_invalid_json(self):
        pred = {"primary_endpoint": "Overall Survival"}  # Missing required fields
        gt = pred.copy()

        result = evaluate_extraction(pred, gt)
        # Should still compute metrics for available fields
        assert result.macro_f1 >= 0.0


class TestAggregateExtractionResults:
    """Tests for aggregate_extraction_results function."""

    def test_aggregate_multiple(self):
        results = [
            ExtractionResult(
                nct_id="NCT001",
                field_metrics={
                    "primary_endpoint": ExtractionMetrics(
                        field="primary_endpoint", token_f1=1.0, exact_match=1.0
                    ),
                    "endpoint_category": ExtractionMetrics(
                        field="endpoint_category", exact_match=1.0
                    ),
                    "intervention_names": ExtractionMetrics(field="intervention_names", set_f1=1.0),
                },
                macro_f1=1.0,
                valid_json=True,
                latency_ms=100.0,
            ),
            ExtractionResult(
                nct_id="NCT002",
                field_metrics={
                    "primary_endpoint": ExtractionMetrics(
                        field="primary_endpoint", token_f1=0.5, exact_match=0.0
                    ),
                    "endpoint_category": ExtractionMetrics(
                        field="endpoint_category", exact_match=1.0
                    ),
                    "intervention_names": ExtractionMetrics(field="intervention_names", set_f1=0.5),
                },
                macro_f1=0.75,
                valid_json=True,
                latency_ms=150.0,
            ),
        ]

        agg = aggregate_extraction_results(results)

        assert agg["macro_f1"] == 0.875  # (1.0 + 0.75) / 2
        assert agg["valid_json_rate"] == 1.0
        assert agg["avg_latency_ms"] == 125.0
        assert agg["total_trials"] == 2

        # Check per-field
        assert agg["per_field"]["primary_endpoint"]["token_f1"] == 0.75  # (1.0 + 0.5) / 2
        assert agg["per_field"]["endpoint_category"]["exact_match"] == 1.0
        assert agg["per_field"]["intervention_names"]["set_f1"] == 0.75

    def test_aggregate_empty(self):
        agg = aggregate_extraction_results([])
        assert agg == {}


class TestCheckExtractionThresholds:
    """Tests for check_extraction_thresholds function."""

    def test_pass_thresholds(self):
        metrics = {
            "macro_f1": 0.65,
            "valid_json_rate": 0.98,
        }
        passed, failures = check_extraction_thresholds(metrics)
        assert passed is True
        assert failures == []

    def test_fail_macro_f1(self):
        metrics = {
            "macro_f1": 0.50,  # Below 0.60 threshold
            "valid_json_rate": 0.98,
        }
        passed, failures = check_extraction_thresholds(metrics)
        assert passed is False
        assert any("macro_f1" in f for f in failures)

    def test_fail_valid_json_rate(self):
        metrics = {
            "macro_f1": 0.65,
            "valid_json_rate": 0.90,  # Below 0.95 threshold
        }
        passed, failures = check_extraction_thresholds(metrics)
        assert passed is False
        assert any("valid_json_rate" in f for f in failures)

    def test_fail_both(self):
        metrics = {
            "macro_f1": 0.50,
            "valid_json_rate": 0.90,
        }
        passed, failures = check_extraction_thresholds(metrics)
        assert passed is False
        assert len(failures) == 2


class TestEvaluateExtractionDataset:
    """Tests for evaluate_extraction_dataset function."""

    def test_evaluate_dataset(self):
        predictions = [
            {
                "primary_endpoint": "Overall Survival",
                "endpoint_category": "overall_survival",
                "intervention_names": ["pembrolizumab"],
                "intervention_class": "monoclonal_antibody",
                "randomized": True,
                "blinding": "double",
                "population_age_group": "adult",
                "key_inclusion": ["Age >= 18"],
                "key_exclusion": ["Pregnancy"],
            },
            {
                "primary_endpoint": "Progression Free Survival",
                "endpoint_category": "progression_free_survival",
                "intervention_names": ["nivolumab"],
                "intervention_class": "monoclonal_antibody",
                "randomized": True,
                "blinding": "double",
                "population_age_group": "adult",
                "key_inclusion": ["Age >= 18"],
                "key_exclusion": ["Pregnancy"],
            },
        ]
        ground_truths = [
            {
                "primary_endpoint": "Overall Survival",
                "endpoint_category": "overall_survival",
                "intervention_names": ["pembrolizumab"],
                "intervention_class": "monoclonal_antibody",
                "randomized": True,
                "blinding": "double",
                "population_age_group": "adult",
                "key_inclusion": ["Age >= 18"],
                "key_exclusion": ["Pregnancy"],
            },
            {
                "primary_endpoint": "Progression Free Survival",
                "endpoint_category": "progression_free_survival",
                "intervention_names": ["nivolumab"],
                "intervention_class": "monoclonal_antibody",
                "randomized": True,
                "blinding": "double",
                "population_age_group": "adult",
                "key_inclusion": ["Age >= 18"],
                "key_exclusion": ["Pregnancy"],
            },
        ]

        result = evaluate_extraction_dataset(predictions, ground_truths)

        assert result["macro_f1"] == 1.0
        assert result["valid_json_rate"] == 1.0
        assert result["total_trials"] == 2
        assert "per_field" in result
