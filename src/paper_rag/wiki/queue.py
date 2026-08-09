"""Wiki 持久化任务队列(SQLite wiki_jobs)。

不照搬基准的进程内 daemon 线程队列——那个设计在 CLI 进程退出时会静默丢弃
未消费的任务, 批量入库场景不可接受。这里 ingest 侧只做一次幂等 INSERT
(成本可忽略, 不阻塞主链路), 由独立的 scripts/wiki_worker.py 进程按自己的
节奏消费, 支持断点续跑与失败退避。

幂等键: (paper_id, content_fingerprint)。指纹由 ingest 对排序后的 chunk_id
集合取哈希, force 重建自然产生新指纹 -> 新任务。语言随任务显式传递
(zh | en | None), worker 不再靠标题临时猜语言。

状态机: pending -> processing -> done | skipped | failed
  - fail: attempts+1, 未超 max_attempts 回 pending 并按 retry_backoff_sec 退避;
  - skipped: 质量门槛拦下的文档(如 mineru+broken), 记原因不产词条;
  - requeue_stale: 崩溃残留的 processing 归还 pending。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, UniqueConstraint, select

from .. import config as cfg
from ..utils.logger import get_logger

log = get_logger(__name__)


class WikiJobRow(SQLModel, table=True):
    __tablename__ = "wiki_jobs"
    __table_args__ = (
        UniqueConstraint("paper_id", "content_fingerprint", name="uq_wiki_job_fingerprint"),
    )

    id: int | None = SQLField(default=None, primary_key=True)
    paper_id: str = SQLField(index=True)
    content_fingerprint: str = ""
    language: str | None = None  # zh | en | None, ingest 显式传入
    status: str = SQLField(default="pending", index=True)
    attempts: int = 0
    next_retry_at: datetime | None = None
    error: str | None = None
    report_json: str = "{}"
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


def _engine():
    from ..store.sqlite_store import get_engine

    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def _job_dict(row: WikiJobRow) -> dict[str, Any]:
    return {
        "job_id": row.id,
        "paper_id": row.paper_id,
        "content_fingerprint": row.content_fingerprint,
        "language": row.language,
        "status": row.status,
        "attempts": row.attempts,
        "error": row.error,
    }


def submit_paper_indexed(
    paper_id: str,
    *,
    language: str | None = None,
    content_fingerprint: str = "",
) -> dict[str, Any]:
    """幂等入队。wiki.enabled=false 时不入队(kill switch 在入口生效)。"""
    if not cfg.load().wiki.enabled:
        return {"queued": False, "reason": "wiki_disabled"}
    with Session(_engine()) as s:
        existing = s.exec(
            select(WikiJobRow).where(
                WikiJobRow.paper_id == paper_id,
                WikiJobRow.content_fingerprint == content_fingerprint,
            )
        ).first()
        if existing is not None:
            return {"queued": True, "created": False, "job_id": existing.id}
        row = WikiJobRow(
            paper_id=paper_id,
            content_fingerprint=content_fingerprint,
            language=language,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
    log.info(f"wiki job queued: paper={paper_id} lang={language} fp={content_fingerprint[:12]}")
    return {"queued": True, "created": True, "job_id": row.id}


def claim_jobs(limit: int | None = None) -> list[dict[str, Any]]:
    """领取可执行任务(pending 且未在退避期), 原子置 processing。"""
    limit = limit or cfg.load().wiki.worker.batch_size
    now = datetime.now(UTC)
    claimed: list[dict[str, Any]] = []
    with Session(_engine()) as s:
        rows = list(
            s.exec(
                select(WikiJobRow)
                .where(WikiJobRow.status == "pending")
                .order_by(WikiJobRow.id)
                .limit(limit * 2)  # 余量过滤退避期任务
            )
        )
        for row in rows:
            if len(claimed) >= limit:
                break
            retry_at = row.next_retry_at
            if retry_at is not None:
                if retry_at.tzinfo is None:  # SQLite 读回可能丢时区, 按 UTC 补
                    retry_at = retry_at.replace(tzinfo=UTC)
                if retry_at > now:
                    continue
            row.status = "processing"
            row.updated_at = now
            s.add(row)
            claimed.append(_job_dict(row))
        s.commit()
    return claimed


def _finish(job_id: int, *, status: str, error: str | None = None, report_json: str = "{}") -> None:
    with Session(_engine()) as s:
        row = s.get(WikiJobRow, job_id)
        if row is None:
            return
        row.status = status
        row.error = error
        row.report_json = report_json
        row.updated_at = datetime.now(UTC)
        s.add(row)
        s.commit()


def complete_job(job_id: int, *, report: dict[str, Any] | None = None) -> None:
    import json

    _finish(job_id, status="done", report_json=json.dumps(report or {}, ensure_ascii=False))


def mark_skipped(job_id: int, *, reason: str) -> None:
    """质量门槛拦下的任务: 如实记 skipped 与原因, 不伪装成功也不静默丢弃。"""
    _finish(job_id, status="skipped", error=reason)


def fail_job(job_id: int, *, error: str) -> None:
    worker_cfg = cfg.load().wiki.worker
    with Session(_engine()) as s:
        row = s.get(WikiJobRow, job_id)
        if row is None:
            return
        row.attempts += 1
        row.error = error
        now = datetime.now(UTC)
        row.updated_at = now
        if row.attempts >= worker_cfg.max_attempts:
            row.status = "failed"
            log.warning(f"wiki job {row.id} ({row.paper_id}) failed permanently: {error}")
        else:
            backoff = worker_cfg.retry_backoff_sec
            delay = backoff[min(row.attempts - 1, len(backoff) - 1)] if backoff else 0
            row.status = "pending"
            row.next_retry_at = now + timedelta(seconds=delay)
            log.info(f"wiki job {row.id} retry in {delay}s (attempt {row.attempts}): {error}")
        s.add(row)
        s.commit()


def requeue_stale(*, older_than_sec: int = 3600) -> int:
    """崩溃残留的 processing 任务归还 pending(worker 启动时调用, 断点续跑)。"""
    cutoff = datetime.now(UTC) - timedelta(seconds=older_than_sec)
    count = 0
    with Session(_engine()) as s:
        rows = s.exec(select(WikiJobRow).where(WikiJobRow.status == "processing"))
        for row in rows:
            updated = row.updated_at
            if updated is not None and updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if updated is not None and updated > cutoff:
                continue
            row.status = "pending"
            row.updated_at = datetime.now(UTC)
            s.add(row)
            count += 1
        s.commit()
    if count:
        log.info(f"wiki queue requeued {count} stale processing jobs")
    return count


def pending_count() -> int:
    with Session(_engine()) as s:
        rows = s.exec(select(WikiJobRow.id).where(WikiJobRow.status == "pending"))
        return len(list(rows))


def stats() -> dict[str, int]:
    out = {"pending": 0, "processing": 0, "done": 0, "failed": 0, "skipped": 0}
    with Session(_engine()) as s:
        for row in s.exec(select(WikiJobRow.status)):
            out[row] = out.get(row, 0) + 1
    return out
