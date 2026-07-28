"""基于 SQLModel 的 SQLite 论文元数据存储。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .. import config as cfg
from ..utils.logger import get_logger

log = get_logger(__name__)


class Paper(SQLModel, table=True):
    """论文元数据与 ingest 状态记录。"""

    # 论文元数据
    paper_id: str = Field(primary_key=True) # 根据论文身份标识生成的唯一主键, 例如 arxiv:2310.11511, doi:10.1145/1234567, pmid:12345678
    title: str = "" # 论文的原始标题
    authors_json: str = "[]"    # 保存作者列表。authors 被保存成 JSON 字符串, 是因为 SQLite 没有原生的字符串列表类型
    year: int | None = None     # 发表年份
    venue: str | None = None    # 发表的会议、期刊或者平台
    doi: str | None = None      # 正式出版论文常用的永久标识
    arxiv_id: str | None = None # arXiv 论文 ID
    arxiv_version: str | None = None # arXiv 版本号
    abstract: str | None = None # 论文的摘要
    title_norm: str | None = Field(default=None, index=True)    # 用于去重的规范化标题, 去掉大小写、空格、标点符号等, 只保留字母和数字
    # 入库状态字段
    status: str = Field(default="created", index=True)  # 论文当前处于哪个处理阶段: created、fetched、parsed、chunked、embedded、indexed、done, 任意步骤失败后变为failed
    parsed_with: str | None = None  # 记录使用的PDF解析器和解析质量
    error: str | None = None    # 入库失败时的错误信息
    # 用户与时间字段
    user_id: str | None = Field(default="system", index=True)   # 论文所属的用户, 默认system, 所有人都能看到的公共论文
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))   # 记录创建时间, default_factory 确保在每次创建对象的时候都重新执行datetime.now(UTC)来获取当前时间, 而不是只在程序启动的时候计算一次
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))   # 记录最近修改时间

class Section(SQLModel, table=True):
    """论文中的逻辑章节, 例如 Introduction、Method、Experiment 等, 用于存储论文的结构化内容。"""

    section_id: str = Field(primary_key=True) # 章节的唯一标识, 例如 arxiv:2310.11511:introduction
    paper_id : str = Field(index=True) # 章节所属的论文 ID
    idx: int    # 章节在论文中的顺序, 用于排序章节
    name: str   # 章节的标题, 例如: Abstract、Introduction、Method、Experiment 等
    page_start: int | None = None   # 章节的起始页码
    page_end: int | None = None     # 章节的结束页码

class Chunk(SQLModel, table=True):
    """实际可检索的论文内容片段。论文的文本块, 用于存储论文的分段内容, 便于后续的向量化和检索。"""

    # 归属与定位字段
    chunk_id: str = Field(primary_key=True) # Chunk的唯一ID, 由论文ID、章节序号、内容类型和块序号生成。例如 arxiv:2310.11511:introduction:0
    paper_id: str = Field(index=True)      # 片段所属的论文 ID
    section_id: str | None = Field(default=None, index=True)    # 片段所属的章节 ID
    section: str | None = None      # 片段所属的章节名称, 例如 Introduction、Method、Experiment 等
    section_idx: int | None = None  # 片段所属的章节在论文中的顺序, 用于排序章节
    page: int | None = None # 片段所在的PDF页码, 用于定位片段在论文中的位置
    char_start: int | None = None
    char_end: int | None = None

    # 内容字段
    modality: str = "text" # 片段的模态类型, 内容类型有: text、figure、table、formula 或者 metadata
    text: str = "" # 片段的文本内容, 用于展示、回答和引用的正文
    context_text: str = ""  # 在正文前加入论文标题和章节名称后的文本, 主要用于生成向量
    title: str | None = None    # 论文标题
    raw_snippet: str | None = None  # 图、表格或公式在 markdown 中的原始片段
    # 文件字段
    source_path: str | None = None  # 解析后的paper.md 的绝对路径
    asset_path: str | None = None   # 资源解析狗的绝对路径
    asset_rel_path: str | None = None   # 图片、表格、公式等资源在解析目录中的相对路径, 用于在前端展示时拼接 URL

    # 扩展字段
    metadata_json: str = "{}"   # JSON 字符串, 保存 section_level、chunk_ordinal、element_type、视觉摘要等扩展信息。
    neighbors_json: str = "[]"  # 相邻 Chunk ID 的 JSON 数组; 当前构建流程默认写入空数组 []。


class IngestRun(SQLModel, table=True):
    """一次 ingest pipeline 的执行记录, 用于追踪每个论文的处理步骤。"""

    __tablename__ = "ingest_runs"

    id: int | None = Field(default=None, primary_key=True)
    paper_id: str = Field(index=True)
    step: str = ""          # parsed | chunked | embedded | indexed 记录论文的处理步骤
    status: str = "ok"      # ok | error
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    error: str | None = None

_ENGINE: Engine | None = None


_CHUNK_COLUMN_MIGRATIONS: dict[str, str] = {
    "section": "TEXT",
    "section_idx": "INTEGER",
    "title": "TEXT",
    "source_path": "TEXT",
    "asset_path": "TEXT",
    "asset_rel_path": "TEXT",
    "char_start": "INTEGER",
    "char_end": "INTEGER",
    "raw_snippet": "TEXT",
    "metadata_json": "TEXT DEFAULT '{}'",
}

def _apply_pragmas(
    dbapi_connection: Any,
    _connection_record: Any,
) -> None:
    """为每个 SQLite 连接设置并发与安全参数。"""
    cursor = dbapi_connection.cursor()

    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def get_engine() -> Engine:
    """创建并缓存 SQLite Engine。"""
    global _ENGINE

    if _ENGINE is None:
        config = cfg.load()
        database_path = Path(config.paths.sqlite_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        _ENGINE = create_engine(
            f"sqlite:///{database_path}",
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 5,
            },
        )
        event.listen(_ENGINE, "connect", _apply_pragmas)
        SQLModel.metadata.create_all(_ENGINE)
        _migrate_chunk_columns(_ENGINE)

        log.info(
            "sqlite engine ready at %s (WAL+busy_timeout)",
            database_path,
        )

    return _ENGINE

def _migrate_chunk_columns(engine: Engine) -> None:
    """为旧版本 chunk 表补充新增的可空字段。"""
    from sqlalchemy import text

    with engine.begin() as connection:
        rows = connection.exec_driver_sql(
            "PRAGMA table_info(chunk)"
        ).fetchall()
        existing_columns = {row[1] for row in rows}

        for column_name, column_type in (
            _CHUNK_COLUMN_MIGRATIONS.items()
        ):
            if column_name in existing_columns:
                continue

            connection.execute(
                text(
                    "ALTER TABLE chunk "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            )

def upsert_paper(
    metadata: dict[str, Any],
    status: str = "created",
) -> None:
    """新增论文, 或使用新元数据更新已有论文。"""
    engine = get_engine()

    with Session(engine) as session:
        paper_id = metadata["paper_id"]
        existing = session.get(Paper, paper_id)
        extra = metadata.get("extra") or {}

        if existing is None:
            from ..ingest.dedup import normalize_title

            title = metadata.get("title", "") or ""
            paper = Paper(
                paper_id=paper_id,
                title=title,
                authors_json=json.dumps(
                    metadata.get("authors", []),
                    ensure_ascii=False,
                ),
                year=metadata.get("year"),
                venue=metadata.get("venue"),
                doi=metadata.get("doi"),
                arxiv_id=metadata.get("arxiv_id"),
                arxiv_version=extra.get("arxiv_version"),
                abstract=metadata.get("abstract"),
                title_norm=normalize_title(title) if title else None,
                status=status,
            )
            session.add(paper)
        else:
            update_fields = (
                "title",
                "year",
                "venue",
                "doi",
                "arxiv_id",
                "abstract",
            )

            for field_name in update_fields:
                value = metadata.get(field_name)
                if value is not None:
                    setattr(existing, field_name, value)

            if "authors" in metadata:
                existing.authors_json = json.dumps(
                    metadata["authors"],
                    ensure_ascii=False,
                )

            if extra.get("arxiv_version"):
                existing.arxiv_version = extra["arxiv_version"]

            existing.status = status
            existing.updated_at = datetime.now(UTC)
            session.add(existing)

        session.commit()


def set_status(
    paper_id: str,
    status: str,
    error: str | None = None,
    parsed_with: str | None = None,
) -> None:
    """更新一篇论文的 ingest 状态。"""
    engine = get_engine()

    with Session(engine) as session:
        paper = session.get(Paper, paper_id)

        if paper is None:
            log.warning("set_status: paper not found %s", paper_id)
            return

        paper.status = status

        if error is not None:
            paper.error = error

        if parsed_with is not None:
            paper.parsed_with = parsed_with

        paper.updated_at = datetime.now(UTC)
        session.add(paper)
        session.commit()


def get_paper(paper_id: str) -> Paper | None:
    """按照主键读取论文记录。"""
    engine = get_engine()

    with Session(engine) as session:
        return session.get(Paper, paper_id)

def record_ingest_step(
    paper_id: str,
    step: str,
    *,
    status: str = "ok",
    error: str | None = None,
) -> int:
    """创建 ingest 步骤记录并返回数据库主键。"""
    engine = get_engine()

    with Session(engine) as session:
        ingest_run = IngestRun(
            paper_id=paper_id,
            step=step,
            status=status,
            error=error,
        )
        session.add(ingest_run)
        session.commit()
        session.refresh(ingest_run)

        # 这里抛出了一个异常, 为了保证服务不崩溃, 后续需要在调用这个函数的地方捕获异常并处理
        if ingest_run.id is None:
            raise RuntimeError("ingest run ID was not generated")

        return ingest_run.id


def finish_ingest_step(
    run_id: int,
    *,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """结束 ingest 步骤并记录最终状态。"""
    engine = get_engine()

    with Session(engine) as session:
        ingest_run = session.get(IngestRun, run_id)

        if ingest_run is None:
            return

        ingest_run.finished_at = datetime.now(UTC)
        ingest_run.status = status

        if error is not None:
            ingest_run.error = error

        session.add(ingest_run)
        session.commit()


def find_existing_paper(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    title_norm: str | None = None,
) -> Paper | None:
    """按照 DOI、arXiv ID、规范化标题的顺序查找已有论文。"""
    engine = get_engine()

    with Session(engine) as session:
        if doi:
            statement = select(Paper).where(Paper.doi == doi)
            paper = session.exec(statement).first()

            if paper is not None:
                return paper

        if arxiv_id:
            statement = select(Paper).where(
                Paper.arxiv_id == arxiv_id
            )
            paper = session.exec(statement).first()

            if paper is not None:
                return paper

        if title_norm:
            statement = select(Paper).where(
                Paper.title_norm == title_norm
            )
            paper = session.exec(statement).first()

            if paper is not None:
                return paper

    return None


def _chunk_payload_for_sqlite(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """将 pipeline chunk 转换成 SQLite 可保存的字段。"""
    payload = dict(chunk)
    payload["neighbors_json"] = json.dumps(
        payload.pop("neighbors", []),
        ensure_ascii=False,
    )
    payload["metadata_json"] = json.dumps(
        payload.pop("metadata", {}),
        ensure_ascii=False,
    )

    allowed_fields = set(Chunk.model_fields)

    return {
        key: value
        for key, value in payload.items()
        if key in allowed_fields
    }


def upsert_sections_and_chunks(
    paper_id: str,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """用最新解析结果替换一篇论文的 section 和 chunk 快照。"""
    engine = get_engine()
    incoming_section_ids = {
        section["section_id"]
        for section in sections
    }
    incoming_chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    with Session(engine) as session:
        existing_chunks = session.exec(
            select(Chunk).where(Chunk.paper_id == paper_id)
        ).all()

        for existing_chunk in existing_chunks:
            if existing_chunk.chunk_id not in incoming_chunk_ids:
                session.delete(existing_chunk)

        existing_sections = session.exec(
            select(Section).where(Section.paper_id == paper_id)
        ).all()

        for existing_section in existing_sections:
            if existing_section.section_id not in incoming_section_ids:
                session.delete(existing_section)

        for section_data in sections:
            section_id = section_data["section_id"]
            existing_section = session.get(Section, section_id)

            if existing_section is None:
                session.add(Section(**section_data))
                continue

            for key, value in section_data.items():
                setattr(existing_section, key, value)

            session.add(existing_section)

        for chunk_data in chunks:
            chunk_id = chunk_data["chunk_id"]
            payload = _chunk_payload_for_sqlite(chunk_data)
            existing_chunk = session.get(Chunk, chunk_id)

            if existing_chunk is None:
                session.add(Chunk(**payload))
                continue

            for key, value in payload.items():
                setattr(existing_chunk, key, value)

            session.add(existing_chunk)

        session.commit()


def list_chunks_for_papers(
    paper_ids: list[str],
) -> list[Chunk]:
    """读取指定论文集合的全部 chunk。"""
    engine = get_engine()

    with Session(engine) as session:
        statement = select(Chunk).where(
            Chunk.paper_id.in_(paper_ids)
        )
        return list(session.exec(statement))


def get_chunk(chunk_id: str) -> Chunk | None:
    """按照主键读取单个 chunk。"""
    engine = get_engine()

    with Session(engine) as session:
        return session.get(Chunk, chunk_id)
