#!/usr/bin/env python3
"""Real Streamlit render acceptance for the dashboard."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from paper_rag import config as cfg


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "src" / "paper_rag" / "dashboard" / "app.py"
    with tempfile.TemporaryDirectory(prefix="paper-rag-dashboard-") as directory:
        root = Path(directory)
        config_path = root / "dashboard.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "data_root": str(root / "data"),
                        "papers_dir": str(root / "data" / "papers"),
                        "parsed_dir": str(root / "data" / "parsed"),
                        "index_dir": str(root / "data" / "index"),
                        "sqlite_path": str(root / "data" / "index" / "papers.sqlite"),
                        "bm25_path": str(root / "data" / "index" / "bm25.pkl"),
                        "models_dir": str(root / "data" / "models"),
                    },
                    "qdrant": {"url": "", "local_path": str(root / "data" / "qdrant")},
                    "vision": {"enabled": False},
                    "wiki": {"enabled": False},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        cfg.load.cache_clear()
        import os

        previous = os.environ.get("PAPER_RAG_CONFIG")
        os.environ["PAPER_RAG_CONFIG"] = str(config_path)
        try:
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            assert not app.exception, app.exception
            titles = [item.value for item in app.title]
            assert "问答工作台" in titles, titles
            text = "\n".join(item.value for item in [*app.markdown, *app.caption])
            assert "PAPER RAG" in text
            assert "尚无查询结果" in text
            metrics = {item.label for item in app.metric}
            assert {"论文", "Chunks", "视觉内容", "最近入库", "最近检索", "监控运行"} <= metrics
            assert len(app.text_area) >= 1
            assert any(button.label == "运行问答" for button in app.button)

            from paper_rag.dashboard.app import DashboardPaths
            from paper_rag.dashboard.services.pipeline_monitor import PipelineMonitorStore

            paths = DashboardPaths.from_data_root(root / "data")
            monitor = PipelineMonitorStore(paths.pipeline_monitor)
            now = datetime.now(UTC).isoformat()
            monitor.append(
                {
                    "run_id": "accept-ingestion",
                    "pipeline": "ingestion",
                    "paper_id": "paper:test",
                    "status": "done",
                    "created_at": now,
                    "timings_ms": {
                        "parse_seconds": 120.0,
                        "chunk_seconds": 30.0,
                        "total_seconds": 150.0,
                    },
                    "metadata": {"chunks": 3},
                }
            )
            from paper_rag.dashboard.services.retrieval_history import RetrievalHistoryStore

            RetrievalHistoryStore().save(
                {
                    "query": "persisted diagnostic query",
                    "timings_ms": {
                        "query_rewrite_ms": 1.0,
                        "dense_ms": 2.0,
                        "sparse_ms": 3.0,
                        "rrf_ms": 1.0,
                        "rerank_ms": 2.0,
                        "diversify_ms": 1.0,
                        "retrieval_total_ms": 10.0,
                    },
                    "stages": {
                        "Query Rewrite": {
                            "timing_ms": 1.0,
                            "rewritten_queries": ["persisted rewritten query"],
                            "bm25_query": "persisted keywords",
                        },
                        "Dense": {"timing_ms": 2.0, "items": []},
                        "Sparse": {"timing_ms": 3.0, "items": []},
                        "RRF": {"timing_ms": 1.0, "items": []},
                        "Rerank": {"timing_ms": 2.0, "items": []},
                        "Diversify": {"timing_ms": 1.0, "items": []},
                    },
                    "chunks": [],
                },
                paper_ids=None,
                top_k=8,
            )
            monitor.append(
                {
                    "run_id": "accept-retrieval",
                    "pipeline": "retrieval",
                    "query": "test query",
                    "status": "ok",
                    "created_at": now,
                    "timings_ms": {
                        "dense_ms": 20.0,
                        "sparse_ms": 5.0,
                        "rrf_ms": 1.0,
                        "retrieval_total_ms": 30.0,
                    },
                    "metadata": {"iterations": []},
                }
            )
            monitor_app = AppTest.from_string(
                "import streamlit as st\n"
                "from paper_rag.dashboard.app import DashboardPaths\n"
                "from paper_rag.dashboard.pages.traces import render\n"
                f"st.session_state['dashboard_paths'] = DashboardPaths.from_data_root({str(root / 'data')!r})\n"
                "render()\n",
                default_timeout=20,
            ).run()
            assert not monitor_app.exception, monitor_app.exception
            assert "管道监控" in [item.value for item in monitor_app.title]
            assert [tab.label for tab in monitor_app.tabs] == ["入库监控", "检索监控"]
            assert any(metric.label == "Total" for metric in monitor_app.metric)
            retrieval_tab = monitor_app.tabs[1]
            retrieval_tab.run()
            assert any(item.label == "查询问题" for item in monitor_app.text_area)
            assert any(button.label == "运行检索诊断" for button in monitor_app.button)
            assert any(item.label == "搜索历史查询" for item in monitor_app.text_input)
            assert any(item.label == "历史运行" for item in monitor_app.selectbox)
            assert any(
                "persisted diagnostic query" in str(item.value)
                for item in [*monitor_app.markdown, *monitor_app.caption]
            )
        finally:
            if previous is None:
                os.environ.pop("PAPER_RAG_CONFIG", None)
            else:
                os.environ["PAPER_RAG_CONFIG"] = previous
            cfg.load.cache_clear()

    print("dashboard acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
