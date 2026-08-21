"""Dashboard service routing for the isolated RAGAS runner."""

from __future__ import annotations


def test_dashboard_ragas_uses_its_own_golden_set_default() -> None:
    from paper_rag.dashboard.pages.evaluation import _default_test_set

    assert _default_test_set(False, "ragas") == "tests/fixtures/evaluation/ragas_golden.json"
    assert _default_test_set(False, "custom") == "tests/fixtures/evaluation/golden.json"
    assert _default_test_set(True, "custom") == ("tests/fixtures/evaluation/retrieval_golden.json")


def test_evaluation_service_routes_ragas_without_shared_eval_runner(tmp_path, monkeypatch) -> None:
    from paper_rag.dashboard.services.evaluation_service import EvaluationHistory, EvaluationService
    from paper_rag.evaluation import ragas_runner

    calls: dict = {}

    class Report:
        def to_dict(self):
            return {
                "schema_version": "ragas-report.v1",
                "run_id": "ragas-run",
                "created_at": "2026-08-14T00:00:00+00:00",
                "backend": "ragas",
                "evaluation": {"mode": "ragas"},
                "aggregate_metrics": {},
                "query_results": [],
            }

    class Runner:
        def __init__(self, answer_fn, evaluator, *, top_k, query_rewrite, max_concurrency):
            calls["evaluator"] = evaluator
            calls["top_k"] = top_k
            calls["query_rewrite"] = query_rewrite
            calls["max_concurrency"] = max_concurrency

        def run(self, test_set):
            calls["test_set"] = str(test_set)
            return Report()

    monkeypatch.setattr(ragas_runner, "RagasEvalRunner", Runner)
    history = EvaluationHistory(tmp_path / "history.jsonl")

    record = EvaluationService(history).run(
        test_set="ragas.json",
        backend="ragas",
        custom_metrics=[],
        ragas_metrics=["faithfulness"],
        top_k=13,
        query_rewrite=False,
        max_concurrency=3,
    )

    assert record["schema_version"] == "ragas-report.v1"
    assert calls["test_set"] == "ragas.json"
    assert calls["evaluator"].metrics == ["faithfulness"]
    assert calls["top_k"] == 13
    assert calls["query_rewrite"] is False
    assert calls["max_concurrency"] == 3
