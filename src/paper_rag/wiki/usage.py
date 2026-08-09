"""wiki 消费记录: QA 每次实际用到 wiki 背景就落一行事件。

用途: trace 可追溯"这次回答参考了哪些词条", 后续评测/proactive 层据此统计
词条的真实价值(哪些词条从未被消费 -> 抽取策略需调整)。

写入永不阻塞 QA: 任何异常只记 warning。空上下文不写行。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, select

from ..utils.logger import get_logger

log = get_logger(__name__)


class WikiUsageRow(SQLModel, table=True):
    __tablename__ = "wiki_usage"

    id: int | None = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True)
    paper_id: str | None = SQLField(default=None, index=True)
    question: str = ""
    trace_id: str | None = SQLField(default=None, index=True)
    wiki_fingerprint: str = ""
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


def _engine():
    from ..store.sqlite_store import get_engine

    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def record_consumption(
    *,
    question: str,
    paper_ids: list[str] | None,
    wiki_context: dict[str, Any] | None,
    trace_id: str | None = None,
) -> int:
    """记录一次 wiki 背景消费, 返回写入行数。失败非致命返回 0。"""
    entries = list((wiki_context or {}).get("entries") or [])
    if not entries:
        return 0
    fingerprint = str((wiki_context or {}).get("fingerprint") or "")
    explicit = [str(p) for p in (paper_ids or []) if p]

    rows: list[WikiUsageRow] = []
    for entry in entries:
        entry_id = str(entry.get("entry_id") or "")
        if not entry_id:
            continue
        # 显式 paper_ids(问答被限定在某几篇)优先; 否则记词条自身的关键论文
        targets = explicit or [str(p) for p in (entry.get("key_papers") or []) if p] or [None]
        for paper_id in targets:
            rows.append(
                WikiUsageRow(
                    entry_id=entry_id,
                    paper_id=paper_id,
                    question=(question or "")[:500],
                    trace_id=trace_id,
                    wiki_fingerprint=fingerprint,
                )
            )
    if not rows:
        return 0
    try:
        with Session(_engine()) as s:
            for row in rows:
                s.add(row)
            s.commit()
        return len(rows)
    except Exception as e:  # pragma: no cover - 防御: 记账永不阻塞 QA
        log.warning(f"wiki usage record failed (non-fatal): {e}")
        return 0


def recent(limit: int = 50) -> list[dict[str, Any]]:
    with Session(_engine()) as s:
        rows = s.exec(
            select(WikiUsageRow).order_by(WikiUsageRow.id.desc()).limit(limit)  # type: ignore[union-attr]
        )
        return [
            {
                "id": r.id,
                "entry_id": r.entry_id,
                "paper_id": r.paper_id,
                "question": r.question,
                "trace_id": r.trace_id,
                "wiki_fingerprint": r.wiki_fingerprint,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def consumed_paper_ids() -> set[str]:
    """被 wiki 背景带出的论文集合(供 proactive/评测层分析路由效果)。"""
    with Session(_engine()) as s:
        return {p for p in s.exec(select(WikiUsageRow.paper_id)) if p}


__all__ = ["consumed_paper_ids", "recent", "record_consumption"]
