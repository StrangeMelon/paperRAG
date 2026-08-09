"""wiki/usage.py 消费记录契约(真实临时 SQLite)。

QA 每次实际使用 wiki 背景都要落一行事件: trace 可追溯"这次回答参考了哪些
词条", proactive/评测层可据此统计词条价值。空上下文不写行。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _isolate(monkeypatch, tmp_path):
    sqlite_store = importlib.import_module("paper_rag.store.sqlite_store")
    config = SimpleNamespace(paths=SimpleNamespace(sqlite_path=str(tmp_path / "wiki.sqlite")))
    monkeypatch.setattr(sqlite_store.cfg, "load", lambda: config)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    return importlib.import_module("paper_rag.wiki.usage")


def _ctx():
    return {
        "role": "background_not_evidence",
        "fingerprint": "concept:rl:3",
        "entries": [
            {
                "entry_id": "concept:rl",
                "name": "Reinforcement Learning",
                "key_papers": ["arxiv:1", "arxiv:2"],
            }
        ],
    }


def test_record_consumption_writes_entry_paper_rows(monkeypatch, tmp_path):
    usage = _isolate(monkeypatch, tmp_path)

    usage.record_consumption(
        question="什么是强化学习",
        paper_ids=["arxiv:1"],
        wiki_context=_ctx(),
        trace_id="t-1",
    )
    rows = usage.recent(limit=10)
    assert len(rows) == 1  # 显式 paper_ids 优先: 1 词条 x 1 论文
    assert rows[0]["entry_id"] == "concept:rl"
    assert rows[0]["paper_id"] == "arxiv:1"
    assert rows[0]["wiki_fingerprint"] == "concept:rl:3"


def test_paper_ids_fall_back_to_key_papers(monkeypatch, tmp_path):
    usage = _isolate(monkeypatch, tmp_path)
    usage.record_consumption(question="q", paper_ids=None, wiki_context=_ctx(), trace_id="t-2")
    assert usage.consumed_paper_ids() == {"arxiv:1", "arxiv:2"}


def test_empty_context_is_noop(monkeypatch, tmp_path):
    usage = _isolate(monkeypatch, tmp_path)
    usage.record_consumption(
        question="q", paper_ids=["arxiv:1"], wiki_context={"entries": []}, trace_id="t-3"
    )
    assert usage.recent(limit=10) == []
    assert usage.consumed_paper_ids() == set()
