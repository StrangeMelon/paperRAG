"""Dashboard evaluation history contracts."""

from __future__ import annotations

from types import SimpleNamespace


def test_evaluation_history_persists_recent_runs(tmp_path) -> None:
    from paper_rag.dashboard.services.evaluation_service import EvaluationHistory

    history = EvaluationHistory(tmp_path / "evaluation.jsonl")
    history.append(
        {
            "run_id": "run-1",
            "created_at": "2026-08-13T10:00:00+00:00",
            "backend": "custom",
            "aggregate_metrics": {"hit_rate": 0.8},
            "query_results": [],
        }
    )
    history.append(
        {
            "run_id": "run-2",
            "created_at": "2026-08-13T11:00:00+00:00",
            "backend": "composite",
            "aggregate_metrics": {"custom.mrr": 0.7, "ragas.faithfulness": 0.9},
            "query_results": [],
        }
    )

    assert [item["run_id"] for item in history.list()] == ["run-2", "run-1"]
    assert history.get("run-2")["backend"] == "composite"


def test_retrieval_service_passes_configured_question_concurrency(monkeypatch, tmp_path) -> None:
    from paper_rag import config as cfg
    from paper_rag.dashboard.services import evaluation_service as service_module

    captured: dict = {}

    class Runner:
        def __init__(self, retrieve_fn, **kwargs):
            captured.update(kwargs)

        def run(self, test_set):
            return {
                "evaluation": {"mode": "retrieval"},
                "aggregate_metrics": {},
                "query_results": [],
            }

    monkeypatch.setattr(service_module, "RetrievalEvalRunner", Runner)
    monkeypatch.setattr(
        cfg, "load", lambda: SimpleNamespace(evaluation=SimpleNamespace(max_concurrency=8))
    )
    history = service_module.EvaluationHistory(tmp_path / "evaluation.jsonl")

    service_module.EvaluationService(history).run(
        test_set="unused.json",
        backend="custom",
        custom_metrics=[],
        ragas_metrics=[],
        mode="retrieval",
        query_rewrite=True,
    )

    assert captured["max_concurrency"] == 8
