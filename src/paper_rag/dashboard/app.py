"""Four-page Streamlit dashboard for paper-rag."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageSpec:
    title: str
    icon: str
    default: bool = False


@dataclass(frozen=True)
class DashboardPaths:
    query_traces: Path
    evaluation_history: Path
    pipeline_monitor: Path

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> DashboardPaths:
        root = Path(data_root) / "dashboard"
        return cls(
            query_traces=root / "query_traces.jsonl",
            evaluation_history=root / "evaluation_history.jsonl",
            pipeline_monitor=root / "pipeline_monitor.jsonl",
        )


PAGE_SPECS = (
    PageSpec("问答工作台", ":material/chat:", True),
    PageSpec("数据浏览", ":material/database:"),
    PageSpec("管道监控", ":material/account_tree:"),
    PageSpec("评测面板", ":material/analytics:"),
)


def page_workbench() -> None:
    from paper_rag.dashboard.pages.workbench import render

    render()


def page_data_browser() -> None:
    from paper_rag.dashboard.pages.data_browser import render

    render()


def page_traces() -> None:
    from paper_rag.dashboard.pages.traces import render

    render()


def page_evaluation() -> None:
    from paper_rag.dashboard.pages.evaluation import render

    render()


def main() -> None:
    import streamlit as st

    from paper_rag import config as cfg
    from paper_rag.dashboard.style import apply_style

    st.set_page_config(
        page_title="Paper RAG Research Desk",
        page_icon=":material/science:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_style()
    paths = DashboardPaths.from_data_root(cfg.load().paths.data_root)
    st.session_state.setdefault("dashboard_paths", paths)

    pages = [
        st.Page(page_workbench, title=PAGE_SPECS[0].title, icon=PAGE_SPECS[0].icon, default=True),
        st.Page(page_data_browser, title=PAGE_SPECS[1].title, icon=PAGE_SPECS[1].icon),
        st.Page(page_traces, title=PAGE_SPECS[2].title, icon=PAGE_SPECS[2].icon),
        st.Page(page_evaluation, title=PAGE_SPECS[3].title, icon=PAGE_SPECS[3].icon),
    ]
    with st.sidebar:
        st.markdown('<div class="brand">PAPER RAG</div>', unsafe_allow_html=True)
        st.caption("Research desk / local")
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
