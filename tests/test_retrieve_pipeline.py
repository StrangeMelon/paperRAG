"""retrieve/pipeline 组合入口的行为契约测试(hybrid/rerank/config 全打桩)。

切片 0: infer_modalities(英文/中文线索、多模态、无线索)。
切片 1: retrieve_round_with_rewrite(多查询池化去重取高分、模态追加轮、
        top_k*3 精排窗口、rewrite_fn 注入优先、无 wiki_context 参数的
        TypeError 兼容、rag.query_rewrite 缺席时恒等回退 + warning)。
切片 2: _diversify_by_paper(单篇限额 50、溢出补位、top_k 截断)。

接口约定(与基准一致):

    infer_modalities(query) -> list[str]
    retrieve_round(query, paper_ids, top_k, *, wiki_context=None) -> list[dict]
    retrieve_round_with_rewrite(...) -> tuple[list[dict], dict]
"""

from __future__ import annotations

from paper_rag.retrieve import pipeline as pl

# ---------- 切片 0: infer_modalities ----------


def test_infer_modalities_en_zh_and_none():
    assert pl.infer_modalities("show me the equation derivation") == ["formula"]
    assert pl.infer_modalities("论文里的表格对比了什么") == ["table"]
    assert pl.infer_modalities("图 3 的示意图和公式") == ["formula", "figure"]
    assert pl.infer_modalities("what is self-rag") == []


# ---------- 切片 1: retrieve_round_with_rewrite ----------


def _stub(monkeypatch, *, hybrid_results, rerank_passthrough=True):
    """打桩 hybrid_search/_rerank; hybrid_results 是 (query, modality) -> hits 映射或函数。"""
    calls = {"hybrid": [], "rerank": []}

    def fake_hybrid(q, top_k, paper_ids, modality):
        calls["hybrid"].append(
            {"q": q, "top_k": top_k, "paper_ids": paper_ids, "modality": modality}
        )
        if callable(hybrid_results):
            return hybrid_results(q, modality)
        return hybrid_results.get((q, modality), [])

    def fake_rerank(query, candidates, top_k):
        calls["rerank"].append({"n": len(candidates), "top_k": top_k})
        return candidates if rerank_passthrough else []

    monkeypatch.setattr(pl, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(pl, "_rerank", fake_rerank)
    return calls


def _rw_two_queries(query, wiki_context=None):
    return {"dense_queries": [query, f"{query} rewritten"]}


def test_pooling_dedups_and_keeps_higher_rrf(monkeypatch):
    results = {
        ("q", None): [{"chunk_id": "a", "score_rrf": 0.1}, {"chunk_id": "b", "score_rrf": 0.3}],
        ("q rewritten", None): [{"chunk_id": "a", "score_rrf": 0.9}],
    }
    calls = _stub(monkeypatch, hybrid_results=results)

    chunks, rw = pl.retrieve_round_with_rewrite("q", None, 4, rewrite_fn=_rw_two_queries)

    assert rw == {"dense_queries": ["q", "q rewritten"]}
    assert len(calls["hybrid"]) == 2  # 每个改写查询一轮, 无模态追加
    by_id = {c["chunk_id"]: c for c in chunks}
    assert by_id["a"]["score_rrf"] == 0.9  # 同块保留高分版本
    assert chunks[0]["chunk_id"] == "a"  # 池化后按 score_rrf 降序


def test_retrieve_round_reports_module_timings(monkeypatch):
    _stub(monkeypatch, hybrid_results=lambda q, m: [{"chunk_id": "a", "score_rrf": 0.5}])
    timings: dict[str, float] = {}

    chunks, _rw, reported = pl.retrieve_round_with_rewrite(
        "q",
        None,
        4,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
        timings=timings,
    )

    assert chunks
    assert reported is timings
    assert {"query_rewrite_ms", "hybrid_ms", "rerank_ms", "diversify_ms"} <= set(timings)
    assert all(value >= 0 for value in timings.values())


def test_diagnostic_fallback_keeps_timings_and_stage_results(monkeypatch):
    calls: list[dict] = []

    def legacy_hybrid(query, *, top_k, paper_ids, modality, timings=None, diagnostics=None):
        calls.append({"timings": timings, "diagnostics": diagnostics})
        assert timings is not None
        assert diagnostics is not None
        timings.update({"dense_ms": 2.0, "sparse_ms": 3.0, "rrf_ms": 1.0})
        diagnostics.update(
            {
                "dense": [{"chunk_id": "dense-1", "score": 0.9}],
                "sparse": [{"chunk_id": "sparse-1", "score_bm25": 4.0}],
                "rrf": [{"chunk_id": "dense-1", "score_rrf": 0.1}],
            }
        )
        return diagnostics["rrf"]

    monkeypatch.setattr(pl, "hybrid_search", legacy_hybrid)
    monkeypatch.setattr(pl, "_rerank", lambda query, candidates, top_k: candidates)
    timings: dict[str, float] = {}
    diagnostics: dict[str, object] = {}

    chunks, _rewrite, _reported = pl.retrieve_round_with_rewrite(
        "q",
        None,
        4,
        rewrite_fn=lambda query, wiki_context=None: {
            "dense_queries": [query],
            "bm25_query": "keywords",
        },
        timings=timings,
        diagnostics=diagnostics,
    )

    assert chunks[0]["chunk_id"] == "dense-1"
    assert chunks[0]["score_rrf"] == 0.1
    assert chunks[0]["score_effective"] == 0.1
    assert chunks[0]["reference_penalized"] is False
    assert calls and calls[0]["timings"] is not None
    assert {"dense_ms", "sparse_ms", "rrf_ms"} <= timings.keys()
    assert diagnostics["dense"] == [{"chunk_id": "dense-1", "score": 0.9}]
    assert diagnostics["sparse"] == [{"chunk_id": "sparse-1", "score_bm25": 4.0}]
    assert diagnostics["rrf"] == [{"chunk_id": "dense-1", "score_rrf": 0.1}]


def test_modality_hint_adds_targeted_rounds(monkeypatch):
    calls = _stub(monkeypatch, hybrid_results=lambda q, m: [])

    pl.retrieve_round_with_rewrite(
        "论文里的表格对比了什么",
        None,
        4,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
    )

    assert [c["modality"] for c in calls["hybrid"]] == [None, "table"]


def test_rerank_window_is_triple_top_k(monkeypatch):
    many = [{"chunk_id": f"c{i}", "score_rrf": 1 - i / 100} for i in range(20)]
    calls = _stub(monkeypatch, hybrid_results=lambda q, m: many)

    pl.retrieve_round_with_rewrite(
        "q", None, 4, rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]}
    )

    assert calls["rerank"] == [{"n": 12, "top_k": 12}]  # top_k*3 窗口


def test_rewrite_fn_without_wiki_context_param(monkeypatch):
    _stub(monkeypatch, hybrid_results=lambda q, m: [])

    def legacy_rewrite(query):  # 无 wiki_context 形参
        return {"dense_queries": [query]}

    _chunks, rw = pl.retrieve_round_with_rewrite("q", None, 4, rewrite_fn=legacy_rewrite)
    assert rw == {"dense_queries": ["q"]}


def test_uses_real_query_rewrite_by_default(monkeypatch):
    """P7 落地后: 默认真的走 rag.query_rewrite.rewrite, 不再有 identity warning。

    (P6 期间此处断言的是"模块缺席 -> 恒等改写 + warning"的过渡契约;
    query_rewrite 重建后过渡结束, 契约翻转为"默认真接线"。)
    """
    calls = _stub(monkeypatch, hybrid_results=lambda q, m: [{"chunk_id": "a", "score_rrf": 0.5}])
    warnings: list[str] = []
    monkeypatch.setattr(pl.log, "warning", warnings.append)
    # 让 query_rewrite 走本地启发式, 不发网络。
    monkeypatch.setenv("PAPER_RAG_FORCE_LOCAL_REWRITE", "1")

    chunks, rw = pl.retrieve_round_with_rewrite("原问题", None, 4)

    assert rw["dense_queries"][0] == "原问题"  # 首项恒为原问题
    assert "bm25_query" in rw, "应拿到 query_rewrite 的完整载荷, 而非恒等改写"
    assert calls["hybrid"][0]["q"] == "原问题"
    assert not any("identity rewrite" in w for w in warnings)
    assert chunks and chunks[0]["chunk_id"] == "a"


def test_identity_fallback_when_query_rewrite_import_fails(monkeypatch):
    """query_rewrite 导入失败(如缺依赖)时仍降级为恒等改写 + warning, 检索照常。"""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if "query_rewrite" in name:
            raise ImportError("simulated missing query_rewrite")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    calls = _stub(monkeypatch, hybrid_results=lambda q, m: [{"chunk_id": "a", "score_rrf": 0.5}])
    warnings: list[str] = []
    monkeypatch.setattr(pl.log, "warning", warnings.append)

    chunks, rw = pl.retrieve_round_with_rewrite("原问题", None, 4)

    assert rw["dense_queries"] == ["原问题"]
    assert calls["hybrid"][0]["q"] == "原问题"
    assert any("identity rewrite" in w for w in warnings)
    assert chunks and chunks[0]["chunk_id"] == "a"


def test_retrieve_round_drops_payload(monkeypatch):
    _stub(monkeypatch, hybrid_results=lambda q, m: [{"chunk_id": "a", "score_rrf": 0.5}])
    out = pl.retrieve_round("q", None, 4)
    assert isinstance(out, list) and out[0]["chunk_id"] == "a"


def test_evaluation_parallel_reaches_hybrid_and_rerank(monkeypatch):
    calls: dict[str, list[bool]] = {"hybrid": [], "rerank": []}

    def hybrid(query, *, top_k, paper_ids, modality, allow_concurrent):
        calls["hybrid"].append(allow_concurrent)
        return [{"chunk_id": "a", "score_rrf": 0.5}]

    def rerank(query, candidates, *, top_k, allow_concurrent):
        calls["rerank"].append(allow_concurrent)
        return candidates

    monkeypatch.setattr(pl, "hybrid_search", hybrid)
    monkeypatch.setattr(pl, "_rerank", rerank)

    chunks, _rewrite = pl.retrieve_round_with_rewrite(
        "q",
        None,
        4,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
        evaluation_parallel=True,
    )

    assert chunks
    assert calls == {"hybrid": [True], "rerank": [True]}


def test_regular_query_demotes_reference_chunk_after_rerank(monkeypatch):
    hits = [
        {
            "chunk_id": "ref",
            "score_rrf": 0.09,
            "metadata": {"is_references": True},
        },
        {"chunk_id": "body", "score_rrf": 0.08, "section": "Methods"},
    ]
    _stub(monkeypatch, hybrid_results=lambda q, m: hits)

    chunks, _rewrite = pl.retrieve_round_with_rewrite(
        "How does the method work?",
        None,
        2,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
    )

    assert [chunk["chunk_id"] for chunk in chunks] == ["body", "ref"]
    assert chunks[1]["reference_penalized"] is True
    assert chunks[1]["score_effective"] < chunks[0]["score_effective"]


def test_reference_query_bypasses_reference_penalty(monkeypatch):
    hits = [
        {
            "chunk_id": "ref",
            "score_rrf": 0.09,
            "metadata": {"is_references": True},
        },
        {"chunk_id": "body", "score_rrf": 0.08},
    ]
    _stub(monkeypatch, hybrid_results=lambda q, m: hits)

    chunks, _rewrite = pl.retrieve_round_with_rewrite(
        "Show me the bibliography",
        None,
        2,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
    )

    assert [chunk["chunk_id"] for chunk in chunks] == ["ref", "body"]
    assert chunks[0]["reference_penalized"] is False


def test_reference_intent_override_survives_followup_query(monkeypatch):
    hits = [
        {
            "chunk_id": "ref",
            "score_rrf": 0.09,
            "metadata": {"is_references": True},
        },
        {"chunk_id": "body", "score_rrf": 0.08},
    ]
    _stub(monkeypatch, hybrid_results=lambda q, m: hits)

    chunks, _rewrite = pl.retrieve_round_with_rewrite(
        "Which specific works?",
        None,
        2,
        reference_intent=True,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
    )

    assert chunks[0]["chunk_id"] == "ref"
    assert chunks[0]["reference_penalized"] is False


def test_reference_policy_is_exposed_in_diagnostics(monkeypatch):
    hits = [
        {
            "chunk_id": "ref",
            "score_rrf": 0.09,
            "metadata": {"is_references": True},
        }
    ]
    _stub(monkeypatch, hybrid_results=lambda q, m: hits)
    timings: dict[str, float] = {}
    diagnostics: dict[str, object] = {}

    chunks, _rewrite, _reported = pl.retrieve_round_with_rewrite(
        "How does the method work?",
        None,
        1,
        rewrite_fn=lambda q, wiki_context=None: {"dense_queries": [q]},
        timings=timings,
        diagnostics=diagnostics,
    )

    assert diagnostics["reference_policy"] == {
        "enabled": True,
        "intent": False,
        "penalty": 0.15,
        "penalized_chunks": 1,
    }
    assert diagnostics["rerank"] == chunks


# ---------- 切片 2: _diversify_by_paper ----------


def _mk(paper: str, i: int) -> dict:
    return {"chunk_id": f"{paper}-{i}", "paper_id": paper}


def test_diversify_caps_fifty_per_paper():
    chunks = [_mk("p1", i) for i in range(51)] + [_mk("p2", 0)]
    out = pl._diversify_by_paper(chunks, top_k=51)
    assert [c["chunk_id"] for c in out] == [
        *[f"p1-{i}" for i in range(50)],
        "p2-0",
    ]


def test_diversify_overflow_backfills_when_quota_short():
    chunks = [_mk("p1", i) for i in range(55)]
    out = pl._diversify_by_paper(chunks, top_k=52)
    # 单篇限额 50, 但只有一篇论文时溢出块补位到 top_k
    assert [c["chunk_id"] for c in out] == [f"p1-{i}" for i in range(52)]


def test_diversify_truncates_at_top_k():
    chunks = [_mk(f"p{i}", 0) for i in range(6)]
    out = pl._diversify_by_paper(chunks, top_k=3)
    assert len(out) == 3
