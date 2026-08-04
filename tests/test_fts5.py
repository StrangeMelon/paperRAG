"""FTS5 稀疏检索的行为契约测试(真实临时 SQLite 库, 无外部服务)。

切片 0: segment_cjk 纯函数(bigram 展开、单字退化、混排、标点断串、ASCII 不变)。
切片 1: _build_match_query(引号原子 OR 拼接、连字符拆分、CJK bigram 短语、
        单字前缀、空查询哨兵、bigram 开关关闭时退回基准整段原子)。
切片 2: 真实临时库端到端(search 自愈回填、中文子串召回、porter 词干召回、
        paper_ids 过滤、rank/score 序、返回原文而非分词镜像、sync_paper 增量)。

相对基准的三处结构性偏离见 docs/adrs/0001-fts5-cjk-bigram.md:
分词镜像 + JOIN 回原文、porter 词干器、去触发器改 Python 同步 + search 自愈。

接口约定:

    segment_cjk(text: str) -> str
    search(query: str, top_k: int = 20, paper_ids: list[str] | None = None) -> list[dict]
    reindex_all() -> int
    sync_paper(paper_id: str) -> int
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paper_rag.retrieve import fts5


def _isolated(monkeypatch, tmp_path: Path, *, bigram: bool = True):
    """真实临时 SQLite 库 + 配置隔离; 返回 (fts5, sqlite_store)。"""
    import paper_rag.config as config
    from paper_rag.store import sqlite_store

    conf = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "papers.sqlite")),
        retrieve=SimpleNamespace(fts5_cjk_bigram=bigram),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    monkeypatch.setattr(fts5, "_INITIALIZED", False)
    return fts5, sqlite_store


def _add_chunks(sqlite_store, rows: list[tuple[str, str, str]]) -> None:
    """rows: (chunk_id, paper_id, text)。"""
    from sqlmodel import Session

    with Session(sqlite_store.get_engine()) as s:
        for chunk_id, paper_id, text in rows:
            s.add(sqlite_store.Chunk(chunk_id=chunk_id, paper_id=paper_id, text=text))
        s.commit()


# ---------- 切片 0: segment_cjk ----------


def test_segment_cjk_bigram_expansion():
    assert fts5.segment_cjk("长程依赖建模") == "长程 程依 依赖 赖建 建模"


def test_segment_cjk_single_char_stays_unigram():
    assert fts5.segment_cjk("图") == "图"


def test_segment_cjk_punctuation_breaks_runs():
    # 全角逗号不在 CJK 统一表意区, 两侧各自成串; 二字串的 bigram 就是其本身
    assert fts5.segment_cjk("长程，依赖") == "长程 ， 依赖"  # noqa: RUF001  全角逗号是测试对象
    assert fts5.segment_cjk("长程依赖，建模分析") == "长程 程依 依赖 ， 建模 模分 分析"  # noqa: RUF001


def test_segment_cjk_mixed_text_keeps_ascii_verbatim():
    out = fts5.segment_cjk("Mamba 模块处理长序列 with SSM")
    assert "Mamba" in out and "with SSM" in out
    assert "模块 块处 处理 理长 长序 序列" in out


def test_segment_cjk_pure_ascii_unchanged():
    text = "Graph neural networks struggle with long-range dependencies."
    assert fts5.segment_cjk(text) == text


# ---------- 切片 1: _build_match_query ----------


def test_match_query_english_tokens_quoted_and_or_joined(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    assert f._build_match_query("graph mamba") == '"graph" OR "mamba"'


def test_match_query_hyphen_splits_instead_of_gluing(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    # 基准把 Graph-Mamba 清洗成单 token GraphMamba, 与索引侧 graph/mamba 永不相交
    assert f._build_match_query("Graph-Mamba") == '"Graph" OR "Mamba"'


def test_match_query_cjk_run_becomes_bigram_phrase(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    assert f._build_match_query("长程依赖") == '"长程 程依 依赖"'


def test_match_query_single_cjk_char_uses_prefix(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    assert f._build_match_query("图") == '"图"*'


def test_match_query_mixed_token_splits_scripts(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    assert f._build_match_query("Mamba模型") == '"Mamba" OR "模型"'


def test_match_query_empty_or_symbols_returns_sentinel(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path)
    assert f._build_match_query("") == '""'
    assert f._build_match_query("!?@#") == '""'


def test_match_query_bigram_disabled_falls_back_to_whole_run(monkeypatch, tmp_path):
    f, _ = _isolated(monkeypatch, tmp_path, bigram=False)
    assert f._build_match_query("长程依赖") == '"长程依赖"'


# ---------- 切片 2: 真实临时库端到端 ----------

_CORPUS = [
    ("c-en-1", "p-en", "Graph neural networks struggle with long-range dependencies."),
    ("c-en-2", "p-en", "The GMB block combines Mamba with node prioritization."),
    ("c-zh-1", "p-zh", "图神经网络在长程依赖建模上的局限性分析。"),
    ("c-zh-2", "p-zh", "选择性状态空间模型可以高效处理长序列。"),
]


def test_search_self_heals_when_index_never_built(monkeypatch, tmp_path):
    """先入库后首查(基准缺陷时序): 首次 search 自动回填, 不返回空。"""
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = f.search("dependencies")
    assert [h["chunk_id"] for h in hits] == ["c-en-1"]


def test_search_chinese_substring_recall(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = f.search("长程依赖")
    assert hits and hits[0]["chunk_id"] == "c-zh-1"


def test_search_porter_stem_recall(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = f.search("depend")
    assert [h["chunk_id"] for h in hits] == ["c-en-1"]


def test_search_returns_original_text_not_segmented_mirror(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = f.search("长程依赖")
    assert hits[0]["text"] == "图神经网络在长程依赖建模上的局限性分析。"


def test_search_paper_ids_filter(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    # "长" 单字前缀两篇都可能命中; 过滤后只剩指定论文
    hits = f.search("长序列", paper_ids=["p-zh"])
    assert hits and all(h["paper_id"] == "p-zh" for h in hits)
    assert f.search("dependencies", paper_ids=["p-zh"]) == []


def test_search_rank_and_score_shape(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    hits = f.search("graph neural networks")
    assert [h["rank_bm25"] for h in hits] == list(range(len(hits)))
    scores = [h["score_bm25"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert set(hits[0]) == {"chunk_id", "paper_id", "modality", "text", "score_bm25", "rank_bm25"}


def test_sync_paper_incremental_update(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)
    assert f.reindex_all() == 4

    # 新论文入库后行数不一致 -> 自愈; 也可显式 sync_paper 单篇增量
    _add_chunks(store, [("c-zh-3", "p-zh2", "多头注意力机制的稀疏化改造。")])
    n = f.sync_paper("p-zh2")
    assert n == 1
    hits = f.search("稀疏化")
    assert hits and hits[0]["chunk_id"] == "c-zh-3"


def test_reindex_all_counts_and_is_idempotent(monkeypatch, tmp_path):
    f, store = _isolated(monkeypatch, tmp_path)
    _add_chunks(store, _CORPUS)

    assert f.reindex_all() == 4
    assert f.reindex_all() == 4  # 重建不翻倍
    assert len(f.search("长程依赖")) == 1
