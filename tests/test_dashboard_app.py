"""Streamlit dashboard page registration and smoke contracts."""

from __future__ import annotations


def test_dashboard_registers_four_pages() -> None:
    from paper_rag.dashboard.app import PAGE_SPECS

    assert [(item.title, item.icon, item.default) for item in PAGE_SPECS] == [
        ("问答工作台", ":material/chat:", True),
        ("数据浏览", ":material/database:", False),
        ("管道监控", ":material/account_tree:", False),
        ("评测面板", ":material/analytics:", False),
    ]


def test_dashboard_runtime_paths_live_under_data_root(tmp_path) -> None:
    from paper_rag.dashboard.app import DashboardPaths

    paths = DashboardPaths.from_data_root(tmp_path)

    assert paths.query_traces == tmp_path / "dashboard" / "query_traces.jsonl"
    assert paths.evaluation_history == tmp_path / "dashboard" / "evaluation_history.jsonl"
    assert paths.pipeline_monitor == tmp_path / "dashboard" / "pipeline_monitor.jsonl"
