"""scripts/ingest_one.py 单篇入库入口的行为契约测试(引擎全打桩, 不发网络)。

切片 0: 参数契约(--arxiv/--pdf 二选一必填、--title/--force 透传)。
切片 1: 来源分发(--arxiv 走 ArxivSource, --pdf 走 LocalSource(title), 均送入
        ingest(force=...))。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

ingest_one = importlib.import_module("scripts.ingest_one")

_RESULT = SimpleNamespace(meta=SimpleNamespace(paper_id="p1", title="T"), pdf_path="/tmp/x.pdf")


def _stub_engine(monkeypatch, calls: dict):
    class _FakeArxiv:
        def fetch(self, identifier):
            calls["arxiv"] = identifier
            return _RESULT

    class _FakeLocal:
        def __init__(self, title=None):
            calls["title"] = title

        def fetch(self, identifier):
            calls["pdf"] = identifier
            return _RESULT

    def _ingest(result, *, force=False):
        calls["ingest"] = {"paper_id": result.meta.paper_id, "force": force}
        return {"paper_id": result.meta.paper_id, "status": "done", "chunks": 3}

    monkeypatch.setattr("paper_rag.ingest.arxiv_source.ArxivSource", _FakeArxiv)
    monkeypatch.setattr("paper_rag.ingest.local_source.LocalSource", _FakeLocal)
    monkeypatch.setattr("paper_rag.store.ingest_pipeline.ingest", _ingest)


def test_requires_arxiv_or_pdf():
    assert ingest_one.main([]) == 2


def test_arxiv_dispatch(monkeypatch):
    calls: dict = {}
    _stub_engine(monkeypatch, calls)
    rc = ingest_one.main(["--arxiv", "2310.11511", "--force"])
    assert rc == 0
    assert calls["arxiv"] == "2310.11511"
    assert calls["ingest"] == {"paper_id": "p1", "force": True}


def test_pdf_dispatch_with_title(monkeypatch):
    calls: dict = {}
    _stub_engine(monkeypatch, calls)
    rc = ingest_one.main(["--pdf", "/abs/path.pdf", "--title", "My Paper"])
    assert rc == 0
    assert calls["title"] == "My Paper"
    assert calls["pdf"] == "/abs/path.pdf"
    assert calls["ingest"]["force"] is False
