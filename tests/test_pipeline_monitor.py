"""Pipeline-only monitoring persistence contracts."""

from __future__ import annotations


def test_pipeline_monitor_store_filters_ingestion_and_retrieval(tmp_path) -> None:
    from paper_rag.dashboard.services.pipeline_monitor import PipelineMonitorStore

    store = PipelineMonitorStore(tmp_path / "pipeline.jsonl")
    store.append({"run_id": "i1", "pipeline": "ingestion", "created_at": "2026-08-13T10:00:00Z"})
    store.append({"run_id": "r1", "pipeline": "retrieval", "created_at": "2026-08-13T11:00:00Z"})

    assert [item["run_id"] for item in store.list(pipeline="ingestion")] == ["i1"]
    assert [item["run_id"] for item in store.list(pipeline="retrieval")] == ["r1"]


def test_pipeline_monitor_store_returns_module_timing_rows(tmp_path) -> None:
    from paper_rag.dashboard.services.pipeline_monitor import PipelineMonitorStore

    store = PipelineMonitorStore(tmp_path / "pipeline.jsonl")
    store.append(
        {
            "run_id": "r1",
            "pipeline": "retrieval",
            "timings_ms": {"dense": 12.5, "sparse": 4.0},
            "created_at": "2026-08-13T11:00:00Z",
        }
    )

    assert store.get("r1")["timings_ms"]["dense"] == 12.5
