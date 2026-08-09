"""人工复核队列: 存放三级解析判不准的合并疑问, 并提供闭环执行动作。

与基准的差别: 基准只有"记一条待复核", 没有任何修正手段——判定为同一概念
之后没有代码能把两条词条并起来, 队列只能看不能动。这里补上 resolve_merge
(调 store.merge_entries 完成关系吸收 + tombstone 重定向) 与 dismiss
(确认是不同概念), 复核才真正闭环。

去重: 同 (event_type, concept_norm, paper_id, reason) 在窗口内只留一行,
避免批量入库把同一疑问刷成几百行。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, select

from ..utils.logger import get_logger
from .schema import WikiEntry, normalize_label

log = get_logger(__name__)

_DEDUPE_WINDOW_HOURS = 24


class WikiReviewRow(SQLModel, table=True):
    __tablename__ = "wiki_review_queue"

    id: int | None = SQLField(default=None, primary_key=True)
    event_type: str = SQLField(index=True)  # resolve_review | consistency
    concept: str = ""
    concept_norm: str = SQLField(default="", index=True)
    paper_id: str | None = SQLField(default=None, index=True)
    reason: str = ""
    payload_json: str = "{}"
    status: str = SQLField(default="pending", index=True)  # pending|resolved|dismissed
    note: str = ""
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None


def _engine():
    from ..store.sqlite_store import get_engine

    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def _row_to_dict(row: WikiReviewRow) -> dict[str, Any]:
    try:
        payload = json.loads(row.payload_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "id": row.id,
        "event_type": row.event_type,
        "concept": row.concept,
        "paper_id": row.paper_id,
        "reason": row.reason,
        "payload": payload,
        "status": row.status,
        "note": row.note,
        "created_at": row.created_at,
    }


def enqueue(
    event_type: str,
    *,
    concept: str | None = None,
    paper_id: str | None = None,
    reason: str = "",
    payload: dict[str, Any] | None = None,
) -> int | None:
    """入队一条复核项, 窗口内同疑问去重, 返回行 id。"""
    concept = (concept or "").strip()
    norm = normalize_label(concept)
    cutoff = datetime.now(UTC) - timedelta(hours=_DEDUPE_WINDOW_HOURS)
    with Session(_engine()) as s:
        dup = s.exec(
            select(WikiReviewRow)
            .where(WikiReviewRow.event_type == event_type)
            .where(WikiReviewRow.concept_norm == norm)
            .where(WikiReviewRow.paper_id == paper_id)
            .where(WikiReviewRow.reason == reason)
            .where(WikiReviewRow.status == "pending")
        ).first()
        if dup is not None and _as_aware(dup.created_at) >= cutoff:
            return dup.id

        row = WikiReviewRow(
            event_type=event_type,
            concept=concept,
            concept_norm=norm,
            paper_id=paper_id,
            reason=reason,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        log.info(f"wiki review queued: {event_type} concept={concept!r} reason={reason}")
        return row.id


def _as_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def recent(limit: int = 50, *, status: str | None = None) -> list[dict[str, Any]]:
    with Session(_engine()) as s:
        stmt = select(WikiReviewRow)
        if status:
            stmt = stmt.where(WikiReviewRow.status == status)
        stmt = stmt.order_by(WikiReviewRow.id.desc()).limit(limit)  # type: ignore[union-attr]
        return [_row_to_dict(r) for r in s.exec(stmt)]


def count_pending() -> int:
    with Session(_engine()) as s:
        return len(list(s.exec(select(WikiReviewRow.id).where(WikiReviewRow.status == "pending"))))


def resolve_merge(review_id: int, *, source_id: str, target_id: str, note: str = "") -> WikiEntry:
    """复核判定"两条是同一概念": 执行词条合并并闭环该复核行。"""
    from . import store as wstore

    merged = wstore.merge_entries(source_id, target_id, reason=f"review#{review_id} {note}".strip())
    _close(review_id, status="resolved", note=note or f"merged {source_id} -> {target_id}")
    return merged


def dismiss(review_id: int, *, note: str = "") -> None:
    """复核判定"确实是不同概念": 只记账, 不动词条。"""
    _close(review_id, status="dismissed", note=note)


def _close(review_id: int, *, status: str, note: str) -> None:
    with Session(_engine()) as s:
        row = s.get(WikiReviewRow, review_id)
        if row is None:
            log.warning(f"wiki review row not found: {review_id}")
            return
        row.status = status
        row.note = note
        row.resolved_at = datetime.now(UTC)
        s.add(row)
        s.commit()


__all__ = ["count_pending", "dismiss", "enqueue", "recent", "resolve_merge"]
