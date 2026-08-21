"""Quality gates owned exclusively by ragas-report.v1."""

from __future__ import annotations

import json

import pytest


def _report(*, mean: float = 0.8, coverage: float = 1.0) -> dict:
    return {
        "schema_version": "ragas-report.v1",
        "evaluation": {
            "ragas_version": "0.4.3",
            "judge_model": "judge",
            "embedding_model": "embedding",
            "golden_set_sha256": "golden",
            "corpus_manifest_sha256": "corpus",
        },
        "aggregate_metrics": {
            "faithfulness": {
                "mean": mean,
                "count": 1,
                "eligible_count": 1,
                "coverage": coverage,
            }
        },
        "query_results": [],
    }


def test_parse_metric_thresholds_rejects_unknown_or_out_of_range_values() -> None:
    from paper_rag.evaluation.ragas_gates import parse_metric_thresholds

    assert parse_metric_thresholds(["faithfulness=0.75"], {"faithfulness"}) == {
        "faithfulness": 0.75
    }
    with pytest.raises(ValueError, match="unknown RAGAS metric"):
        parse_metric_thresholds(["mrr=0.5"], {"faithfulness"})
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_metric_thresholds(["faithfulness=1.2"], {"faithfulness"})


def test_evaluate_ragas_gates_checks_score_coverage_and_regression(tmp_path) -> None:
    from paper_rag.evaluation.ragas_gates import evaluate_ragas_gates, load_ragas_baseline

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_report(mean=0.9)), encoding="utf-8")
    baseline = load_ragas_baseline(baseline_path)

    violations = evaluate_ragas_gates(
        _report(mean=0.7, coverage=0.8),
        fail_under={"faithfulness": 0.75},
        min_coverage={"faithfulness": 0.9},
        baseline=baseline,
        max_regression={"faithfulness": 0.1},
    )

    assert {item["gate"] for item in violations} == {
        "fail_under",
        "min_coverage",
        "max_regression",
    }


def test_evaluate_ragas_gates_rejects_incomparable_baseline() -> None:
    from paper_rag.evaluation.ragas_gates import evaluate_ragas_gates

    baseline = _report()
    baseline["evaluation"]["judge_model"] = "other-judge"

    with pytest.raises(ValueError, match="judge_model"):
        evaluate_ragas_gates(
            _report(),
            fail_under={},
            min_coverage={},
            baseline=baseline,
            max_regression={"faithfulness": 0.1},
        )
