"""scripts/wiki_worker.py 后台 worker 的行为契约(triggers/嵌入全打桩)。

切片:
- --once 领取一批并处理: 报告含 skipped -> 任务记 skipped; 正常报告 -> done;
- --concurrency 并发处理多篇论文, 且限制在安全范围内;
- 处理抛异常 -> fail_job 退避重试, --drain 下重试至 max_attempts 后 failed, rc 1;
- 启动先 requeue_stale, 崩溃残留的 processing 任务能被本轮续跑;
- 每轮结束做 Qdrant 补偿同步(脏词条 embed+mirror), 失败只记日志不炸 worker。
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

wiki_worker = importlib.import_module("scripts.wiki_worker")


def _isolate(monkeypatch, tmp_path, *, backoff=None):
    sqlite_store = importlib.import_module("paper_rag.store.sqlite_store")
    config = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "wiki.sqlite")),
        wiki=SimpleNamespace(
            enabled=True,
            worker=SimpleNamespace(
                batch_size=10,
                concurrency=1,
                max_attempts=3,
                retry_backoff_sec=backoff if backoff is not None else [0, 0, 0],
            ),
        ),
    )
    monkeypatch.setattr(sqlite_store.cfg, "load", lambda: config)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    # 补偿同步默认打空桩, 相关用例单独覆盖
    monkeypatch.setattr(wiki_worker, "_compensate_qdrant", lambda: 0)
    return importlib.import_module("paper_rag.wiki.queue")


def _stub_triggers(monkeypatch, handler):
    # triggers 属后续课次; 以假模块注入, worker 用 importlib 动态解析
    mod = ModuleType("paper_rag.wiki.triggers")
    mod.on_paper_indexed = handler
    monkeypatch.setitem(sys.modules, "paper_rag.wiki.triggers", mod)


def test_once_processes_batch_and_marks_done(monkeypatch, tmp_path):
    wq = _isolate(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="a")
    wq.submit_paper_indexed("arxiv:2", language="en", content_fingerprint="b")

    calls: list[tuple] = []

    def _handler(paper_id, *, language=None):
        calls.append((paper_id, language))
        return {"created": 1, "patched": 0, "review": 0}

    _stub_triggers(monkeypatch, _handler)
    rc = wiki_worker.main(["--once"])

    assert rc == 0
    assert calls == [("arxiv:1", "zh"), ("arxiv:2", "en")]  # 语言显式传给 triggers
    assert wq.stats()["done"] == 2


def test_once_processes_papers_concurrently(monkeypatch, tmp_path):
    wq = _isolate(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="en", content_fingerprint="a")
    wq.submit_paper_indexed("arxiv:2", language="en", content_fingerprint="b")

    barrier = threading.Barrier(2, timeout=2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def _handler(paper_id, *, language=None):
        with calls_lock:
            calls.append(paper_id)
        barrier.wait()
        time.sleep(0.01)
        return {"created": 1}

    _stub_triggers(monkeypatch, _handler)
    rc = wiki_worker.main(["--once", "--concurrency", "2"])

    assert rc == 0
    assert set(calls) == {"arxiv:1", "arxiv:2"}
    assert wq.stats()["done"] == 2


def test_rejects_excessive_concurrency(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    with pytest.raises(SystemExit, match="between 1 and 32"):
        wiki_worker.main(["--once", "--concurrency", "33"])


def test_skipped_report_marks_job_skipped(monkeypatch, tmp_path):
    wq = _isolate(monkeypatch, tmp_path)
    wq.submit_paper_indexed("sha1:notice", language="zh", content_fingerprint="a")

    _stub_triggers(monkeypatch, lambda pid, language=None: {"skipped": "parsed_with=mineru+broken"})
    rc = wiki_worker.main(["--once"])

    assert rc == 0
    stats = wq.stats()
    assert stats["skipped"] == 1 and stats["done"] == 0


def test_drain_retries_until_permanent_failure(monkeypatch, tmp_path):
    wq = _isolate(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:bad", language="en", content_fingerprint="a")
    wq.submit_paper_indexed("arxiv:good", language="en", content_fingerprint="b")

    def _handler(paper_id, *, language=None):
        if paper_id == "arxiv:bad":
            raise RuntimeError("llm down")
        return {"created": 1}

    _stub_triggers(monkeypatch, _handler)
    rc = wiki_worker.main(["--drain"])

    assert rc == 1  # 有永久失败
    stats = wq.stats()
    assert stats["failed"] == 1  # bad 重试 3 次后 failed
    assert stats["done"] == 1  # good 正常完成


def test_requeues_stale_processing_on_start(monkeypatch, tmp_path):
    wq = _isolate(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="a")
    wq.claim_jobs(limit=1)  # 模拟上一个 worker 崩溃时残留 processing

    _stub_triggers(monkeypatch, lambda pid, language=None: {"created": 1})
    rc = wiki_worker.main(["--once", "--stale-sec", "0"])

    assert rc == 0
    assert wq.stats()["done"] == 1


def test_compensate_qdrant_failures_do_not_crash(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    def _boom():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(wiki_worker, "_compensate_qdrant", _boom)
    _stub_triggers(monkeypatch, lambda pid, language=None: {"created": 1})
    # 队列为空, 只有补偿同步在跑; 异常必须被吞掉
    rc = wiki_worker.main(["--once"])
    assert rc == 0
