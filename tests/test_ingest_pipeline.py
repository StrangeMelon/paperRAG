"""入库流水线 ingest 的行为契约测试(全下游打桩, 不碰真实服务)。

切片 0: 元数据卡片(_title_aliases 缩写词、英文/中文模板路由、chunk 字段)。
切片 1: 主流程(状态机顺序、卡片插在 chunks[0]、向量条数、Qdrant 替换语义、
        语言贯通到 grade_sections、wiki 钩子缺模块时非致命)。
切片 2: 幂等与去重(dedup 探测 merged_into、done 跳过、force 覆盖)。
切片 3: 失败隔离(步骤异常 -> failed + ingest_runs 记录 + 异常上抛;
        真空 chunks -> failed/no_chunks——基准该守卫在插卡之后是死代码,
        重建版挪到插卡之前, 2026-08-01 确认的偏离)。

接口约定(与基准一致):

    ingest(result: FetchResult, *, force: bool = False) -> dict
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper_rag.store.ingest_pipeline as ip
from paper_rag.ingest.schema import FetchResult, PaperMeta


class _FakeSqlite:
    def __init__(self, existing_status: str | None = None, existing_other_id: str | None = None):
        self.statuses: list[tuple[str, dict]] = []
        self.runs: list[list] = []  # [name, status, error]
        self.saved: tuple | None = None
        self.upserted: dict | None = None
        self._existing_status = existing_status
        self._existing_other_id = existing_other_id

    def record_ingest_step(self, paper_id: str, name: str) -> int:
        self.runs.append([name, "start", None])
        return len(self.runs) - 1

    def finish_ingest_step(self, run_id: int, status: str, error: str | None = None) -> None:
        self.runs[run_id][1] = status
        self.runs[run_id][2] = error

    def set_status(self, paper_id: str, status: str, **kw) -> None:
        self.statuses.append((status, kw))

    def upsert_paper(self, meta: dict, status: str) -> None:
        self.upserted = {"meta": meta, "status": status}

    def get_paper(self, paper_id: str):
        if self._existing_status is None:
            return None
        return SimpleNamespace(paper_id=paper_id, status=self._existing_status)

    def find_existing_paper(self, doi=None, arxiv_id=None, title_norm=None):
        if self._existing_other_id is None:
            return None
        return SimpleNamespace(paper_id=self._existing_other_id)

    def upsert_sections_and_chunks(self, paper_id: str, sections, chunks) -> None:
        self.saved = (sections, chunks)


class _FakeQdrant:
    def __init__(self):
        self.ops: list[tuple] = []

    def delete_chunks_for_paper(self, paper_id: str) -> None:
        self.ops.append(("delete", paper_id))

    def upsert_chunks(self, chunks, vectors) -> int:
        self.ops.append(("upsert", len(chunks), len(vectors)))
        return len(chunks)


def _meta(**kw) -> PaperMeta:
    base = {"paper_id": "p1", "title": "Graph-Mamba: Towards Long-Range Modeling", "source": "test"}
    base.update(kw)
    return PaperMeta(**base)


def _sections_chunks(names: list[str]):
    sections = [
        {"section_id": f"s{i}", "paper_id": "p1", "idx": i, "name": n} for i, n in enumerate(names)
    ]
    chunks = [
        {
            "chunk_id": f"c{i}",
            "paper_id": "p1",
            "section": n,
            "modality": "text",
            "text": n,
            "context_text": f"ctx {n}",
            "metadata": {},
        }
        for i, n in enumerate(names)
    ]
    return sections, chunks


def _wire(
    monkeypatch,
    tmp_path: Path,
    *,
    language: str | None = None,
    section_names: list[str] | None = None,
    sqlite: _FakeSqlite | None = None,
):
    """把 pipeline 的全部下游替换为桩; 返回 (fake_sqlite, fake_qdrant, parsed_dir)。"""
    parsed = tmp_path / "parsed"
    parsed.mkdir(exist_ok=True)
    if language is not None:
        (parsed / "language.json").write_text(
            json.dumps({"document_language": language}), encoding="utf-8"
        )
    fake_sql = sqlite or _FakeSqlite()
    fake_qd = _FakeQdrant()
    names = section_names or ["Introduction", "Method", "Experiments", "Conclusion"]
    monkeypatch.setattr(ip, "sqlite_store", fake_sql)
    monkeypatch.setattr(ip, "qdrant_store", fake_qd)
    monkeypatch.setattr(ip, "parse_pdf", lambda pid, path: (parsed, "mineru"))
    monkeypatch.setattr(ip, "build_chunks", lambda pid, d, title: _sections_chunks(names))
    monkeypatch.setattr(
        ip, "bge_m3", SimpleNamespace(encode=lambda texts: [[0.0] * 4 for _ in texts])
    )
    return fake_sql, fake_qd, parsed


# ---------------------------------------------------------------------------
# 切片 0: 元数据卡片
# ---------------------------------------------------------------------------


def test_title_aliases_acronym_and_stopwords() -> None:
    assert ip._title_aliases("Retrieval-Augmented Generation for Knowledge Tasks") == ["RAG"]
    assert ip._title_aliases("A Survey of the Field") == ["SF"]  # 停用词 a/of/the 不入缩写
    assert ip._title_aliases("综合能源服务区块链的网络架构") == []  # 中文标题优雅空集


def test_metadata_chunk_en_fields() -> None:
    meta = _meta(arxiv_id="2402.00789", authors=["A", "B"], year=2024, abstract="Long  range.")
    card = ip._paper_metadata_chunk(meta)

    assert card["chunk_id"] == hashlib.sha1(b"p1::paper-metadata").hexdigest()[:20]
    assert card["modality"] == "metadata"
    assert card["section"] == "Paper Metadata"
    assert card["section_idx"] == -1
    assert card["text"].startswith("Paper metadata record.\n")
    assert "Title: Graph-Mamba: Towards Long-Range Modeling" in card["text"]
    assert "arXiv id: 2402.00789" in card["text"]
    assert "Authors: A, B" in card["text"]
    assert "Abstract: Long range." in card["text"]  # 摘要空白归一
    assert card["context_text"] == card["text"]
    assert card["metadata"]["aliases"] == ["GM"]


def test_metadata_chunk_zh_template() -> None:
    meta = _meta(
        title="综合能源服务区块链的网络架构",
        authors=["张三"],
        year=2020,
        abstract="研究 综合能源。",
    )
    card = ip._paper_metadata_chunk(meta, language="zh")

    assert card["text"].startswith("论文元数据记录。\n")
    assert "标题: 综合能源服务区块链的网络架构" in card["text"]
    assert "作者: 张三" in card["text"]
    assert "年份: 2020" in card["text"]
    assert "摘要: 研究 综合能源。" in card["text"]
    assert "Title:" not in card["text"]


# ---------------------------------------------------------------------------
# 切片 1: 主流程
# ---------------------------------------------------------------------------


def test_happy_path_states_card_vectors_and_replacement(monkeypatch, tmp_path: Path) -> None:
    fake_sql, fake_qd, _ = _wire(monkeypatch, tmp_path)
    out = ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"))

    assert out["status"] == "done"
    assert out["chunks"] == 5  # 4 正文 + 1 元数据卡片
    assert "error" in out["wiki"]  # wiki 模块未重建, 非致命

    assert fake_sql.upserted["status"] == "fetched"
    status_names = [s for s, _ in fake_sql.statuses]
    assert status_names == ["parsed", "parsed", "chunked", "embedded", "indexed", "done"]
    assert fake_sql.statuses[1][1]["parsed_with"] == "mineru+complete"  # 打分拼接

    _, chunks = fake_sql.saved
    assert chunks[0]["modality"] == "metadata"  # 卡片在 chunks[0]
    assert [r[:2] for r in fake_sql.runs] == [
        ["parse", "ok"],
        ["chunk", "ok"],
        ["embed", "ok"],
        ["index", "ok"],
    ]
    assert fake_qd.ops == [("delete", "p1"), ("upsert", 5, 5)]  # 先删后插的替换语义


def test_zh_language_flows_to_grading_and_card(monkeypatch, tmp_path: Path) -> None:
    fake_sql, _, _ = _wire(
        monkeypatch, tmp_path, language="zh", section_names=["引言", "方法", "实验", "结论"]
    )
    ip.ingest(FetchResult(meta=_meta(title="中文论文"), pdf_path="/tmp/x.pdf"))

    # 语言若未贯通, zh 节名走 en 表会判 broken
    assert fake_sql.statuses[1][1]["parsed_with"] == "mineru+complete"
    _, chunks = fake_sql.saved
    assert chunks[0]["text"].startswith("论文元数据记录。")


# ---------------------------------------------------------------------------
# 切片 2: 幂等与去重
# ---------------------------------------------------------------------------


def test_dedup_probe_returns_merged_into(monkeypatch, tmp_path: Path) -> None:
    fake_sql = _FakeSqlite(existing_other_id="p0")
    _wire(monkeypatch, tmp_path, sqlite=fake_sql)
    out = ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"))
    assert out == {"paper_id": "p1", "status": "skipped", "merged_into": "p0", "reason": "dedup"}


def test_done_paper_skipped_unless_force(monkeypatch, tmp_path: Path) -> None:
    fake_sql = _FakeSqlite(existing_status="done")
    _wire(monkeypatch, tmp_path, sqlite=fake_sql)
    out = ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"))
    assert out == {"paper_id": "p1", "status": "skipped", "reason": "done"}

    fake_sql2 = _FakeSqlite(existing_status="done")
    _wire(monkeypatch, tmp_path, sqlite=fake_sql2)
    out2 = ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"), force=True)
    assert out2["status"] == "done"


# ---------------------------------------------------------------------------
# 切片 3: 失败隔离
# ---------------------------------------------------------------------------


def test_step_failure_marks_failed_and_raises(monkeypatch, tmp_path: Path) -> None:
    fake_sql, _, _ = _wire(monkeypatch, tmp_path)

    def _boom(pid, path):
        raise RuntimeError("gpu exploded")

    monkeypatch.setattr(ip, "parse_pdf", _boom)
    with pytest.raises(RuntimeError, match="gpu exploded"):
        ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"))

    assert fake_sql.runs[0][0] == "parse" and fake_sql.runs[0][1] == "error"
    failed = [kw for s, kw in fake_sql.statuses if s == "failed"]
    assert failed and failed[0]["error"].startswith("parse: gpu exploded")


def test_empty_build_chunks_fails_before_metadata_card(monkeypatch, tmp_path: Path) -> None:
    """基准把守卫放在插卡之后成了死代码; 重建版真空产物必须 failed/no_chunks。"""
    fake_sql, fake_qd, _ = _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(ip, "build_chunks", lambda pid, d, title: ([], []))

    out = ip.ingest(FetchResult(meta=_meta(), pdf_path="/tmp/x.pdf"))
    assert out["status"] == "failed" and out["reason"] == "no_chunks"
    assert fake_qd.ops == []  # 不得空转到嵌入/索引
    failed = [kw for s, kw in fake_sql.statuses if s == "failed"]
    assert failed and failed[0]["error"].startswith("chunk: empty")
