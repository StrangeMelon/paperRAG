"""hybrid RRF 融合检索的行为契约测试(两腿与配置全打桩)。

切片 0: rrf_fuse 纯函数(名次融合数学、共同块分数相加、字段合并、score_dense
        提升、输入不被改写、缺 chunk_id 跳过、排序)。
切片 1: _sparse_search 后端路由(fts5 主路、异常回退 rank_bm25、配置直选)。
切片 2: hybrid_search 组合(两腿参数透传、modality 对稀疏腿事后过滤、
        top_k*2 截断)。

相对基准的偏离: rrf_fuse 先拷贝再合并, 不原地改写调用方传入的 dict
(基准 setdefault 直接写进 dense_hits 的元素)。回退语义保持基准"仅异常回退"
(中文静默退化的根源已在 fts5 课修复, 合法空结果是诚实信号)。

接口约定(与基准一致):

    rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]
    hybrid_search(query, *, top_k=None, paper_ids=None, modality=None) -> list[dict]
"""

from __future__ import annotations

import pytest

from paper_rag.retrieve import hybrid


def _conf(monkeypatch, *, backend: str = "fts5"):
    from types import SimpleNamespace

    import paper_rag.config as config

    conf = SimpleNamespace(
        retrieve=SimpleNamespace(
            top_k_dense=20,
            top_k_bm25=20,
            rrf_k=60,
            rerank_top_k=8,
            sparse_backend=backend,
        )
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    return conf


# ---------- 切片 0: rrf_fuse ----------


def test_rrf_math_and_ordering():
    dense = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
    sparse = [{"chunk_id": "a", "score_bm25": 5.0}, {"chunk_id": "c", "score_bm25": 4.0}]

    fused = hybrid.rrf_fuse([dense, sparse], k=60)

    by_id = {d["chunk_id"]: d for d in fused}
    assert by_id["a"]["score_rrf"] == pytest.approx(1 / 61 + 1 / 61)  # 两条列表各第 1 名
    assert by_id["b"]["score_rrf"] == pytest.approx(1 / 62)
    assert by_id["c"]["score_rrf"] == pytest.approx(1 / 62)
    assert fused[0]["chunk_id"] == "a"  # 双列表共同靠前者胜出


def test_rrf_merges_fields_and_promotes_score_dense():
    dense = [{"chunk_id": "a", "score": 0.9, "text": "T"}]
    sparse = [{"chunk_id": "a", "score_bm25": 5.0, "rank_bm25": 0}]

    fused = hybrid.rrf_fuse([dense, sparse])

    top = fused[0]
    assert top["score_bm25"] == 5.0 and top["text"] == "T"  # 两侧字段都保留
    assert top["score_dense"] == 0.9  # 供 abstain 的绝对相似度信号


def test_rrf_does_not_mutate_inputs():
    dense_item = {"chunk_id": "a", "score": 0.9}
    sparse_item = {"chunk_id": "a", "score_bm25": 5.0}

    hybrid.rrf_fuse([[dense_item], [sparse_item]])

    assert dense_item == {"chunk_id": "a", "score": 0.9}  # 基准会被塞进 score_bm25
    assert sparse_item == {"chunk_id": "a", "score_bm25": 5.0}


def test_rrf_skips_items_without_chunk_id_and_empty_lists():
    assert hybrid.rrf_fuse([]) == []
    assert hybrid.rrf_fuse([[{"no_id": 1}], []]) == []


# ---------- 切片 1: _sparse_search 路由 ----------


def test_sparse_routes_to_fts5(monkeypatch):
    _conf(monkeypatch, backend="fts5")
    from paper_rag.retrieve import fts5, sparse_bm25

    monkeypatch.setattr(fts5, "search", lambda q, top_k, paper_ids: [{"chunk_id": "from-fts5"}])
    monkeypatch.setattr(
        sparse_bm25, "search", lambda q, top_k, paper_ids: pytest.fail("不应触达 bm25")
    )

    assert hybrid._sparse_search("q", 20, None) == [{"chunk_id": "from-fts5"}]


def test_sparse_falls_back_on_exception(monkeypatch):
    _conf(monkeypatch, backend="fts5")
    from paper_rag.retrieve import fts5, sparse_bm25

    def _boom(q, top_k, paper_ids):
        raise RuntimeError("fts5 down")

    monkeypatch.setattr(fts5, "search", _boom)
    monkeypatch.setattr(sparse_bm25, "search", lambda q, top_k, paper_ids: [{"chunk_id": "bm25"}])
    warnings: list[str] = []
    monkeypatch.setattr(hybrid.log, "warning", warnings.append)

    assert hybrid._sparse_search("q", 20, None) == [{"chunk_id": "bm25"}]
    assert any("falling back" in w for w in warnings)


def test_sparse_backend_config_selects_bm25_directly(monkeypatch):
    _conf(monkeypatch, backend="rank_bm25")
    from paper_rag.retrieve import fts5, sparse_bm25

    monkeypatch.setattr(fts5, "search", lambda *a, **kw: pytest.fail("不应触达 fts5"))
    monkeypatch.setattr(sparse_bm25, "search", lambda q, top_k, paper_ids: [{"chunk_id": "bm25"}])

    assert hybrid._sparse_search("q", 20, None) == [{"chunk_id": "bm25"}]


# ---------- 切片 2: hybrid_search 组合 ----------


def _stub_legs(monkeypatch, dense_hits, sparse_hits):
    calls: dict = {}

    def fake_dense(query, top_k, paper_ids, modality):
        calls["dense"] = {"top_k": top_k, "paper_ids": paper_ids, "modality": modality}
        return dense_hits

    def fake_sparse(query, top_k, paper_ids):
        calls["sparse"] = {"top_k": top_k, "paper_ids": paper_ids}
        return sparse_hits

    monkeypatch.setattr(hybrid, "dense", type("D", (), {"retrieve": staticmethod(fake_dense)}))
    monkeypatch.setattr(hybrid, "_sparse_search", fake_sparse)
    return calls


def test_hybrid_passes_candidate_pool_sizes_and_filters(monkeypatch):
    _conf(monkeypatch)
    calls = _stub_legs(monkeypatch, [], [])

    hybrid.hybrid_search("q", paper_ids=["p1"], modality="table")

    assert calls["dense"] == {"top_k": 20, "paper_ids": ["p1"], "modality": "table"}
    assert calls["sparse"] == {"top_k": 20, "paper_ids": ["p1"]}


def test_hybrid_filters_sparse_by_modality_post_hoc(monkeypatch):
    _conf(monkeypatch)
    sparse = [
        {"chunk_id": "t1", "modality": "table"},
        {"chunk_id": "x1", "modality": "text"},
    ]
    _stub_legs(monkeypatch, [], sparse)

    fused = hybrid.hybrid_search("q", modality="table")

    assert [h["chunk_id"] for h in fused] == ["t1"]


def test_hybrid_truncates_to_double_top_k(monkeypatch):
    _conf(monkeypatch)
    dense_hits = [{"chunk_id": f"d{i}", "score": 1 - i / 100} for i in range(20)]
    _stub_legs(monkeypatch, dense_hits, [])

    fused = hybrid.hybrid_search("q", top_k=3)
    assert len(fused) == 6  # top_k*2, 给 reranker 留余量

    fused_default = hybrid.hybrid_search("q")
    assert len(fused_default) == 16  # rerank_top_k(8)*2
