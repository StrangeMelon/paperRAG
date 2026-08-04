"""rank_bm25 后备稀疏检索的行为契约测试(真实临时 SQLite 库, 无外部服务)。

切片 0: _tokenize(英文整词小写、中文 bigram 与 fts5.segment_cjk 构造上一致、
        混排、标点丢弃)。
切片 1: 真实临时库(建索引、中英文检索、payload 含真实 section/title、
        rank/score 形状、空库降级)。
切片 2: paper_ids 先全库打分后过滤(top 名次全被其他论文占据时仍能命中)。
切片 3: 行数自愈(入库新 chunk 后自动重建)与 invalidate 兼容 API。
切片 4: 规模护栏(超出 bm25_max_chunks 拒绝建索引, 明确告警, 返回空)。

相对基准的偏离(2026-08-04 用户确认): 中文 unigram 统一为 bigram(复用
segment_cjk); 删除只写不读的 pickle 持久化; payload 填真实 section/title;
新增规模护栏 retrieve.bm25_max_chunks(知识库目标 20000 篇 ≈ 10^6 chunks,
纯内存 BM25 仅作小规模后备 + 评测对照)。

接口约定(与基准一致):

    search(query: str, top_k: int = 20, paper_ids: list[str] | None = None) -> list[dict]
    build_index(force: bool = False) -> _Index
    invalidate() -> None
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paper_rag.retrieve import fts5, sparse_bm25


def _isolated(monkeypatch, tmp_path: Path, *, max_chunks: int = 200000):
    import paper_rag.config as config
    from paper_rag.store import sqlite_store

    conf = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "papers.sqlite")),
        retrieve=SimpleNamespace(fts5_cjk_bigram=True, bm25_max_chunks=max_chunks),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    monkeypatch.setattr(sparse_bm25, "_INDEX", None)
    return sparse_bm25, sqlite_store


def _add_chunks(sqlite_store, rows: list[tuple[str, str, str]]) -> None:
    """rows: (chunk_id, paper_id, text)。"""
    from sqlmodel import Session

    with Session(sqlite_store.get_engine()) as s:
        for chunk_id, paper_id, text in rows:
            s.add(
                sqlite_store.Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=text,
                    section="Intro",
                    title="T",
                )
            )
        s.commit()


_CORPUS = [
    ("c-en-1", "p-en", "Graph neural networks struggle with long-range dependencies."),
    ("c-en-2", "p-en", "The GMB block combines Mamba with node prioritization."),
    ("c-zh-1", "p-zh", "图神经网络在长程依赖建模上的局限性分析。"),
    ("c-zh-2", "p-zh", "选择性状态空间模型可以高效处理长序列。"),
]


# ---------- 切片 0: _tokenize ----------


def test_tokenize_english_words_lowercased():
    assert sparse_bm25._tokenize("Graph-Mamba GMB") == ["graph", "mamba", "gmb"]


def test_tokenize_chinese_bigram():
    assert sparse_bm25._tokenize("长程依赖建模") == ["长程", "程依", "依赖", "赖建", "建模"]


def test_tokenize_matches_fts5_segmentation():
    """两个稀疏后端 zh 粒度构造上一致(ADR-0001 记账问题在本课关闭)。"""
    text = "综合能源服务区块链"
    assert sparse_bm25._tokenize(text) == fts5.segment_cjk(text).split()


def test_tokenize_mixed_and_punctuation():
    # 孤立单字 "用" 保留为 unigram(与 fts5 索引侧对孤立单字 run 的行为一致)
    assert sparse_bm25._tokenize("用 Mamba 建模长序列!") == [
        "用",
        "mamba",
        "建模",
        "模长",
        "长序",
        "序列",
    ]


def test_tokenize_empty():
    assert sparse_bm25._tokenize("") == []
    assert sparse_bm25._tokenize(None) == []


# ---------- 切片 1: 建索引与检索 ----------


def test_search_english_and_payload_fields(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = bm.search("long-range dependencies")
    assert hits and hits[0]["chunk_id"] == "c-en-1"
    top = hits[0]
    assert top["section"] == "Intro" and top["title"] == "T"  # 基准硬编码 None, 重建填真值
    assert {"chunk_id", "paper_id", "modality", "page", "text", "score_bm25", "rank_bm25"} <= set(
        top
    )


def test_search_chinese_bigram_recall(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = bm.search("长程依赖")
    assert hits and hits[0]["chunk_id"] == "c-zh-1"


def test_search_rank_increments_and_score_desc(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = bm.search("graph neural networks", top_k=3)
    assert [h["rank_bm25"] for h in hits] == list(range(len(hits)))
    scores = [h["score_bm25"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_corpus_returns_empty(monkeypatch, tmp_path):
    bm, _store = _isolated(monkeypatch, tmp_path)
    assert bm.search("anything") == []


def test_zero_score_hits_are_dropped(monkeypatch, tmp_path):
    """无词项交集的查询应返回空, 而非基准的 top_k 条 0 分噪声。"""
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)
    assert bm.search("quantum entanglement") == []


# ---------- 切片 2: 先全库打分后过滤 ----------


def test_paper_ids_filter_survives_crowded_top(monkeypatch, tmp_path):
    """构造 top 名次全被 p-noise 占据的场景: 后过滤截断会得 0, 先打分后过滤不会。"""
    bm, store = _isolated(monkeypatch, tmp_path)
    noise = [
        (f"c-n-{i}", "p-noise", "state spaces state spaces state spaces modeling")
        for i in range(30)
    ]
    target = [("c-t-1", "p-target", "Selective state spaces enable long sequence modeling.")]
    _add_chunks(store, noise + target + _CORPUS)

    # 无过滤时 top-5 全被高词频的 p-noise 占据, 目标块挤不进去
    unfiltered = bm.search("state spaces modeling", top_k=5)
    assert all(h["paper_id"] == "p-noise" for h in unfiltered)

    hits = bm.search("state spaces modeling", top_k=5, paper_ids=["p-target"])
    assert hits and hits[0]["chunk_id"] == "c-t-1" and hits[0]["score_bm25"] > 0
    assert bm.search("state spaces", top_k=5, paper_ids=["no-such"]) == []


# ---------- 切片 3: 自愈与 invalidate ----------


def test_search_self_heals_after_new_ingest(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)
    assert bm.search("dependencies")  # 建起缓存

    _add_chunks(store, [("c-zh-3", "p-zh2", "多头注意力机制的稀疏化改造。")])
    hits = bm.search("稀疏化")  # 行数 4->5, 应自动重建后命中
    assert hits and hits[0]["chunk_id"] == "c-zh-3"


def test_invalidate_forces_rebuild(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)
    idx1 = bm.build_index()
    bm.invalidate()
    idx2 = bm.build_index()
    assert idx1 is not idx2 and idx2.chunk_ids == idx1.chunk_ids


# ---------- 切片 4: 规模护栏 ----------


def test_scale_guard_refuses_oversized_corpus(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path, max_chunks=3)
    _add_chunks(store, _CORPUS)  # 4 条 > 上限 3

    warnings: list[str] = []
    monkeypatch.setattr(bm.log, "warning", warnings.append)
    hits = bm.search("dependencies")
    assert hits == []
    assert any("bm25_max_chunks" in w for w in warnings), "应有明确超限告警"


def test_scale_guard_boundary_allows_exact_limit(monkeypatch, tmp_path):
    bm, store = _isolated(monkeypatch, tmp_path, max_chunks=4)
    _add_chunks(store, _CORPUS)  # 恰好 4 条 = 上限, 应可用
    assert bm.search("dependencies")
