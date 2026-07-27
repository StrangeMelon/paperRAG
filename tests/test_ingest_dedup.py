"""论文采集去重逻辑的行为契约测试。"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _dedup_module() -> ModuleType:
    return importlib.import_module("paper_rag.ingest.dedup")


def _install_fake_sqlite_store(
    monkeypatch,
    rows: dict[str, SimpleNamespace],
) -> None:
    store_package = ModuleType("paper_rag.store")
    store_package.__path__ = []

    sqlite_store = ModuleType("paper_rag.store.sqlite_store")
    sqlite_store.get_paper = lambda paper_id: rows.get(paper_id)

    monkeypatch.setitem(sys.modules, "paper_rag.store", store_package)
    monkeypatch.setitem(
        sys.modules,
        "paper_rag.store.sqlite_store",
        sqlite_store,
    )


def test_normalize_title_removes_case_spacing_and_punctuation() -> None:
    dedup = _dedup_module()

    assert dedup.normalize_title("Self-RAG: Learning") == "selfraglearning"
    assert dedup.normalize_title("  Hello, World!  ") == "helloworld"
    assert dedup.normalize_title("Version_2") == "version2"
    assert dedup.normalize_title("中文 标题") == "中文标题"
    assert dedup.normalize_title("!!!") == ""


def test_is_done_returns_true_only_for_completed_papers(
    monkeypatch,
) -> None:
    dedup = _dedup_module()
    rows = {
        "paper:done": SimpleNamespace(status="done"),
        "paper:parsing": SimpleNamespace(status="parsing"),
        "paper:failed": SimpleNamespace(status="failed"),
    }
    _install_fake_sqlite_store(monkeypatch, rows)

    assert dedup.is_done("paper:done") is True
    assert dedup.is_done("paper:parsing") is False
    assert dedup.is_done("paper:failed") is False
    assert dedup.is_done("paper:missing") is False
