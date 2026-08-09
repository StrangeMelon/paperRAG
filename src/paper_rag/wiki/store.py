"""Wiki 存储层: SQLite 为真相源, Qdrant 只是可重建的向量镜像。

表职责(与 ADR-0003 一致):
- wiki_entries        词条当前快照(定义、类别、版本、merged_into 重定向、qdrant 脏标)
- wiki_labels         可索引的中英文名字/别名(text_norm 索引, 查找唯一入口)
- wiki_entry_papers   概念-关键论文关系(可反查"这篇论文支撑哪些概念")
- wiki_entry_evidence 概念-chunk 证据关系
- wiki_versions       追加式历史版本(自动流程永不删除旧事实)

设计要点:
- 所有名字查找走 wiki_labels.text_norm 索引, 不做 list_all 全表扫描
  (基准 find_match 的已知性能缺陷, 20k 篇规模下不可接受)。
- merged_into 非空表示 tombstone; 读路径默认跟随重定向(带环保护)。
- upsert/合并只追加关系, 不删除; 版本史每次写快照。
- Qdrant 同步失败不回滚 SQLite: upsert 标 qdrant_dirty, 补偿同步后清除。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, UniqueConstraint, select

from .. import config as cfg
from ..utils.logger import get_logger
from .schema import Variant, WikiEntry, WikiLabel, normalize_label

log = get_logger(__name__)

_REDIRECT_MAX_HOPS = 10


class WikiEntryRow(SQLModel, table=True):
    __tablename__ = "wiki_entries"

    entry_id: str = SQLField(primary_key=True)
    name: str
    category: str = "concept"
    definition: str = ""
    definition_language: str | None = None  # zh | en | None
    variants_json: str = "[]"
    related_json: str = "[]"
    open_problems_json: str = "[]"
    version: int = 1
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    definition_lock_until: datetime | None = None  # 只锁定义重写
    merged_into: str | None = SQLField(default=None, index=True)  # 非空即 tombstone
    qdrant_dirty: bool = SQLField(default=True, index=True)  # 待镜像补偿标记


class WikiLabelRow(SQLModel, table=True):
    __tablename__ = "wiki_labels"
    __table_args__ = (UniqueConstraint("entry_id", "text_norm", name="uq_wiki_label"),)

    id: int | None = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True)
    text: str
    text_norm: str = SQLField(index=True)
    language: str | None = None  # zh | en | None
    kind: str = "variant"  # primary | translation | acronym | variant
    source_paper_id: str | None = None
    confidence: float = 1.0
    verified: bool = False
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class WikiEntryPaperRow(SQLModel, table=True):
    __tablename__ = "wiki_entry_papers"
    __table_args__ = (UniqueConstraint("entry_id", "paper_id", name="uq_wiki_entry_paper"),)

    id: int | None = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True)
    paper_id: str = SQLField(index=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class WikiEntryEvidenceRow(SQLModel, table=True):
    __tablename__ = "wiki_entry_evidence"
    __table_args__ = (UniqueConstraint("entry_id", "chunk_id", name="uq_wiki_entry_evidence"),)

    id: int | None = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True)
    chunk_id: str = SQLField(index=True)
    paper_id: str | None = None
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class WikiVersionRow(SQLModel, table=True):
    __tablename__ = "wiki_versions"

    id: int | None = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True)
    version: int
    content_json: str = ""
    reason: str = ""
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


def _engine():
    from ..store.sqlite_store import get_engine

    engine = get_engine()
    # wiki 表注册于本模块; 若 get_engine 先于本模块 import 已建过表, 这里幂等补建
    SQLModel.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------- 内部工具


def _compose(s: Session, row: WikiEntryRow) -> WikiEntry:
    """由快照行 + 关系表组装完整 WikiEntry(消费端一次读取)。"""
    labels = list(
        s.exec(
            select(WikiLabelRow)
            .where(WikiLabelRow.entry_id == row.entry_id)
            .order_by(WikiLabelRow.id)
        )
    )
    papers = list(
        s.exec(
            select(WikiEntryPaperRow)
            .where(WikiEntryPaperRow.entry_id == row.entry_id)
            .order_by(WikiEntryPaperRow.id)
        )
    )
    evidence = list(
        s.exec(
            select(WikiEntryEvidenceRow)
            .where(WikiEntryEvidenceRow.entry_id == row.entry_id)
            .order_by(WikiEntryEvidenceRow.id)
        )
    )
    return WikiEntry(
        entry_id=row.entry_id,
        name=row.name,
        category=row.category,  # type: ignore[arg-type]
        definition=row.definition,
        definition_language=row.definition_language,  # type: ignore[arg-type]
        labels=[
            WikiLabel(
                text=lb.text,
                language=lb.language,  # type: ignore[arg-type]
                kind=lb.kind,  # type: ignore[arg-type]
                source_paper_id=lb.source_paper_id,
                confidence=lb.confidence,
                verified=lb.verified,
            )
            for lb in labels
        ],
        key_papers=[p.paper_id for p in papers],
        variants=[Variant(**v) for v in json.loads(row.variants_json or "[]")],
        related=json.loads(row.related_json or "[]"),
        open_problems=json.loads(row.open_problems_json or "[]"),
        evidence_chunks=[e.chunk_id for e in evidence],
        version=row.version,
        updated_at=row.updated_at,
        definition_lock_until=row.definition_lock_until,
        merged_into=row.merged_into,
    )


def _existing_label_norms(s: Session, entry_id: str) -> set[str]:
    rows = s.exec(select(WikiLabelRow.text_norm).where(WikiLabelRow.entry_id == entry_id))
    return set(rows)


def _add_labels(s: Session, entry_id: str, labels: list[WikiLabel]) -> int:
    seen = _existing_label_norms(s, entry_id)
    added = 0
    for lb in labels:
        norm = normalize_label(lb.text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        s.add(
            WikiLabelRow(
                entry_id=entry_id,
                text=lb.text,
                text_norm=norm,
                language=lb.language,
                kind=lb.kind,
                source_paper_id=lb.source_paper_id,
                confidence=lb.confidence,
                verified=lb.verified,
            )
        )
        added += 1
    return added


def _add_papers(s: Session, entry_id: str, paper_ids: list[str]) -> int:
    existing = set(
        s.exec(select(WikiEntryPaperRow.paper_id).where(WikiEntryPaperRow.entry_id == entry_id))
    )
    added = 0
    for pid in paper_ids:
        if not pid or pid in existing:
            continue
        existing.add(pid)
        s.add(WikiEntryPaperRow(entry_id=entry_id, paper_id=pid))
        added += 1
    return added


def _add_evidence(s: Session, entry_id: str, items: list[dict[str, Any]]) -> int:
    existing = set(
        s.exec(
            select(WikiEntryEvidenceRow.chunk_id).where(WikiEntryEvidenceRow.entry_id == entry_id)
        )
    )
    added = 0
    for item in items:
        chunk_id = item.get("chunk_id")
        if not chunk_id or chunk_id in existing:
            continue
        existing.add(chunk_id)
        s.add(
            WikiEntryEvidenceRow(
                entry_id=entry_id,
                chunk_id=chunk_id,
                paper_id=item.get("paper_id"),
            )
        )
        added += 1
    return added


def _snapshot_version(s: Session, entry: WikiEntry, reason: str) -> None:
    s.add(
        WikiVersionRow(
            entry_id=entry.entry_id,
            version=entry.version,
            content_json=json.dumps(entry.model_dump(mode="json"), ensure_ascii=False),
            reason=reason,
        )
    )


def _resolve_redirect_in_session(s: Session, entry_id: str) -> str:
    current = entry_id
    for _ in range(_REDIRECT_MAX_HOPS):
        row = s.get(WikiEntryRow, current)
        if row is None or not row.merged_into:
            return current
        current = row.merged_into
    log.warning(f"wiki redirect chain too long from {entry_id}, stop at {current}")
    return current


# ---------------------------------------------------------------- 公开 API


def upsert_entry(entry: WikiEntry, *, reason: str = "") -> WikiEntry:
    """插入或更新词条快照。标量字段覆盖, 关系(标签/论文/证据)只追加不删除;
    每次写入追加版本快照并标记 qdrant_dirty。"""
    with Session(_engine()) as s:
        row = s.get(WikiEntryRow, entry.entry_id)
        now = datetime.now(UTC)
        if row is None:
            row = WikiEntryRow(
                entry_id=entry.entry_id,
                name=entry.name,
                category=entry.category,
                definition=entry.definition,
                definition_language=entry.definition_language,
                variants_json=json.dumps(
                    [v.model_dump() for v in entry.variants], ensure_ascii=False
                ),
                related_json=json.dumps(entry.related, ensure_ascii=False),
                open_problems_json=json.dumps(entry.open_problems, ensure_ascii=False),
                version=1,
                updated_at=now,
                definition_lock_until=entry.definition_lock_until,
                qdrant_dirty=True,
            )
            s.add(row)
        else:
            row.name = entry.name
            row.category = entry.category
            row.definition = entry.definition
            row.definition_language = entry.definition_language
            row.variants_json = json.dumps(
                [v.model_dump() for v in entry.variants], ensure_ascii=False
            )
            row.related_json = json.dumps(entry.related, ensure_ascii=False)
            row.open_problems_json = json.dumps(entry.open_problems, ensure_ascii=False)
            row.version += 1
            row.updated_at = now
            row.definition_lock_until = entry.definition_lock_until
            row.qdrant_dirty = True
            s.add(row)
        _add_labels(s, entry.entry_id, entry.labels)
        _add_papers(s, entry.entry_id, entry.key_papers)
        _add_evidence(s, entry.entry_id, [{"chunk_id": c} for c in entry.evidence_chunks])
        s.flush()
        composed = _compose(s, row)
        _snapshot_version(s, composed, reason)
        s.commit()
    log.info(f"wiki upsert {entry.entry_id} v{composed.version} ({reason})")
    return composed


def add_labels(entry_id: str, labels: list[WikiLabel]) -> int:
    with Session(_engine()) as s:
        added = _add_labels(s, entry_id, labels)
        s.commit()
    return added


def add_key_papers(entry_id: str, paper_ids: list[str]) -> int:
    with Session(_engine()) as s:
        added = _add_papers(s, entry_id, paper_ids)
        s.commit()
    return added


def add_evidence(entry_id: str, items: list[dict[str, Any]]) -> int:
    with Session(_engine()) as s:
        added = _add_evidence(s, entry_id, items)
        s.commit()
    return added


def get_entry(entry_id: str, *, follow_redirect: bool = True) -> WikiEntry | None:
    with Session(_engine()) as s:
        target = _resolve_redirect_in_session(s, entry_id) if follow_redirect else entry_id
        row = s.get(WikiEntryRow, target)
        return _compose(s, row) if row else None


def find_by_label(text: str) -> list[str]:
    """按规范化词面查词条 ID(走 text_norm 索引), 重定向解析后去重。"""
    norm = normalize_label(text)
    if not norm:
        return []
    with Session(_engine()) as s:
        rows = s.exec(select(WikiLabelRow.entry_id).where(WikiLabelRow.text_norm == norm))
        out: list[str] = []
        for entry_id in rows:
            resolved = _resolve_redirect_in_session(s, entry_id)
            if resolved not in out:
                out.append(resolved)
        return out


def list_entries(*, include_merged: bool = False) -> list[WikiEntry]:
    with Session(_engine()) as s:
        stmt = select(WikiEntryRow)
        if not include_merged:
            stmt = stmt.where(WikiEntryRow.merged_into.is_(None))  # type: ignore[union-attr]
        return [_compose(s, row) for row in s.exec(stmt)]


def version_count(entry_id: str) -> int:
    with Session(_engine()) as s:
        rows = s.exec(select(WikiVersionRow.id).where(WikiVersionRow.entry_id == entry_id))
        return len(list(rows))


def merge_entries(source_id: str, target_id: str, *, reason: str = "") -> WikiEntry:
    """把 source 并入 target: 关系吸收、source 置 tombstone、target 版本+1。
    只追加不删除——source 的标签/论文/证据行保留在原处供追溯。"""
    with Session(_engine()) as s:
        final_target = _resolve_redirect_in_session(s, target_id)
        if final_target == source_id:
            raise ValueError(f"merge would create a cycle: {source_id} -> {target_id}")
        source = s.get(WikiEntryRow, source_id)
        target = s.get(WikiEntryRow, final_target)
        if source is None or target is None:
            raise ValueError(f"merge needs both entries: {source_id}, {final_target}")

        src = _compose(s, source)
        _add_labels(s, final_target, src.labels)
        _add_papers(s, final_target, src.key_papers)
        _add_evidence(s, final_target, [{"chunk_id": c} for c in src.evidence_chunks])

        source.merged_into = final_target
        source.updated_at = datetime.now(UTC)
        s.add(source)
        target.version += 1
        target.updated_at = datetime.now(UTC)
        target.qdrant_dirty = True
        s.add(target)
        s.flush()
        composed = _compose(s, target)
        _snapshot_version(s, composed, reason or f"merged {source_id} into {final_target}")
        s.commit()
    log.info(f"wiki merge {source_id} -> {final_target} ({reason})")
    return composed


# ---------------------------------------------------------------- Qdrant 镜像


def _stable_point_id(entry_id: str) -> int:
    return int(hashlib.sha1(entry_id.encode("utf-8")).hexdigest()[:16], 16)


def pending_qdrant_entries() -> list[WikiEntry]:
    """qdrant_dirty 且非 tombstone 的词条, 供补偿同步。"""
    with Session(_engine()) as s:
        stmt = select(WikiEntryRow).where(
            WikiEntryRow.qdrant_dirty == True,  # noqa: E712 - sqlalchemy 布尔比较
            WikiEntryRow.merged_into.is_(None),  # type: ignore[union-attr]
        )
        return [_compose(s, row) for row in s.exec(stmt)]


def mark_qdrant_synced(entry_id: str) -> None:
    with Session(_engine()) as s:
        row = s.get(WikiEntryRow, entry_id)
        if row is not None:
            row.qdrant_dirty = False
            s.add(row)
            s.commit()


def mirror_entry(entry: WikiEntry, vector: list[float]) -> None:
    """镜像单词条到 Qdrant 并清除脏标。失败向上抛出, 由调用方决定重试;
    SQLite 侧不回滚, 脏标存续即重试凭据。"""
    from qdrant_client.http import models as qm

    from ..store.qdrant_store import get_client

    client = get_client()
    coll = cfg.load().qdrant.collection_wiki
    payload: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "name": entry.name,
        "category": entry.category,
        "definition_language": entry.definition_language,
        "version": entry.version,
        "definition_excerpt": (entry.definition or "")[:500],
        "labels": [lb.text for lb in entry.labels][:10],
    }
    client.upsert(
        collection_name=coll,
        points=[
            qm.PointStruct(id=_stable_point_id(entry.entry_id), vector=vector, payload=payload)
        ],
        wait=True,
    )
    mark_qdrant_synced(entry.entry_id)


def search_qdrant(query_vec: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    from ..store.qdrant_store import get_client

    client = get_client()
    coll = cfg.load().qdrant.collection_wiki
    if hasattr(client, "query_points"):
        qres = client.query_points(
            collection_name=coll, query=query_vec, limit=top_k, with_payload=True
        )
        hits = qres.points if hasattr(qres, "points") else qres
    else:
        hits = client.search(
            collection_name=coll, query_vector=query_vec, limit=top_k, with_payload=True
        )
    out: list[dict[str, Any]] = []
    for hit in hits:
        d = dict(hit.payload or {})
        d["score"] = float(hit.score)
        out.append(d)
    return out
