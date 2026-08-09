"""wiki/queue.py 持久化任务队列契约(不照搬基准的进程内 daemon queue)。

钉死的关键行为:
- 任务落 SQLite wiki_jobs 表, CLI 进程退出不丢任务(基准 daemon 线程的真实缺陷);
- 同一 paper_id + content_fingerprint 幂等, force 重建产生新指纹 -> 新任务;
- wiki.enabled=false 时 submit 不入队(kill switch 在入口生效);
- claim 原子置 processing, 不会双领; fail 走退避重试, 超过 max_attempts 终为 failed;
- 语言随任务显式传递, worker 不再靠标题猜语言;
- requeue_stale 把崩溃残留的 processing 任务放回 pending(断点续跑)。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _isolated_queue(
    monkeypatch, tmp_path: Path, *, enabled: bool = True, backoff=None
) -> ModuleType:
    sqlite_store = importlib.import_module("paper_rag.store.sqlite_store")
    wq = importlib.import_module("paper_rag.wiki.queue")
    config = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "wiki.sqlite")),
        wiki=SimpleNamespace(
            enabled=enabled,
            worker=SimpleNamespace(
                batch_size=10,
                max_attempts=3,
                retry_backoff_sec=backoff if backoff is not None else [0, 0, 0],
            ),
        ),
    )
    monkeypatch.setattr(sqlite_store.cfg, "load", lambda: config)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    return wq


def test_submit_is_idempotent_per_fingerprint(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path)

    first = wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="fp-a")
    assert first["queued"] is True and first["created"] is True

    dup = wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="fp-a")
    assert dup["queued"] is True and dup["created"] is False
    assert dup["job_id"] == first["job_id"]

    # force 重建 -> 内容指纹变化 -> 新任务
    forced = wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="fp-b")
    assert forced["created"] is True
    assert forced["job_id"] != first["job_id"]
    assert wq.pending_count() == 2


def test_submit_respects_kill_switch(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path, enabled=False)
    res = wq.submit_paper_indexed("arxiv:1", language="en", content_fingerprint="fp")
    assert res["queued"] is False
    assert wq.pending_count() == 0


def test_claim_is_exclusive_and_carries_language(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="fp")

    jobs = wq.claim_jobs(limit=5)
    assert len(jobs) == 1
    assert jobs[0]["paper_id"] == "arxiv:1"
    assert jobs[0]["language"] == "zh"  # 语言显式随任务传递
    # 已被领取, 不会双领
    assert wq.claim_jobs(limit=5) == []


def test_complete_and_skip(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="en", content_fingerprint="fp")
    (job,) = wq.claim_jobs(limit=1)

    wq.complete_job(job["job_id"], report={"created": 2, "patched": 1})
    assert wq.stats()["done"] == 1

    wq.submit_paper_indexed("sha1:notice", language="zh", content_fingerprint="fp2")
    (job2,) = wq.claim_jobs(limit=1)
    wq.mark_skipped(job2["job_id"], reason="parsed_with=mineru+broken")
    stats = wq.stats()
    assert stats["skipped"] == 1
    assert wq.claim_jobs(limit=5) == []


def test_fail_retries_with_backoff_then_fails_permanently(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path)  # 退避 [0,0,0]: 立即可重试
    wq.submit_paper_indexed("arxiv:1", language="en", content_fingerprint="fp")

    for attempt in range(1, 4):  # max_attempts=3
        (job,) = wq.claim_jobs(limit=1)
        wq.fail_job(job["job_id"], error=f"boom {attempt}")

    # 三次失败后永久 failed, 不再可领取
    assert wq.claim_jobs(limit=5) == []
    stats = wq.stats()
    assert stats["failed"] == 1
    assert stats["pending"] == 0


def test_fail_backoff_delays_next_claim(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path, backoff=[3600, 3600, 3600])
    wq.submit_paper_indexed("arxiv:1", language="en", content_fingerprint="fp")
    (job,) = wq.claim_jobs(limit=1)
    wq.fail_job(job["job_id"], error="transient")

    # 一小时退避内不可领取
    assert wq.claim_jobs(limit=5) == []
    assert wq.stats()["pending"] == 1  # 仍在队列里, 只是未到重试时间


def test_requeue_stale_processing(monkeypatch, tmp_path):
    wq = _isolated_queue(monkeypatch, tmp_path)
    wq.submit_paper_indexed("arxiv:1", language="zh", content_fingerprint="fp")
    wq.claim_jobs(limit=1)

    # 模拟 worker 崩溃: processing 残留, 重启后归还
    requeued = wq.requeue_stale(older_than_sec=0)
    assert requeued == 1
    (job,) = wq.claim_jobs(limit=1)
    assert job["paper_id"] == "arxiv:1"
