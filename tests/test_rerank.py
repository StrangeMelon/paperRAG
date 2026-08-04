"""BGE reranker 重排的行为契约测试(模型与配置全打桩, 不碰真实权重)。

切片 0: 降级路径(空候选、enabled=false 直通截断、模型不可用回退原序、
        compute_score 异常回退、_LOAD_FAILED 闩锁不重试)。
切片 1: 重排语义(按分数降序、score_rerank 写入、单 float 兼容、top_k 截断)。
切片 2: 不改写输入(调用方列表顺序与元素保持原状——基准原地 sort + 写键)。

接口约定(与基准一致):

    rerank(query: str, candidates: list[dict], *, top_k: int | None = None) -> list[dict]
"""

from __future__ import annotations

from types import SimpleNamespace

from paper_rag.retrieve import rerank as rr


def _conf(monkeypatch, *, enabled: bool = True, top_k: int = 8):
    import paper_rag.config as config

    conf = SimpleNamespace(
        reranker=SimpleNamespace(
            enabled=enabled,
            model_name="stub",
            cache_dir=None,
            use_fp16=True,
            top_k=top_k,
        ),
        paths=SimpleNamespace(models_dir="/nonexistent"),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    monkeypatch.setattr(rr, "_MODEL", None)
    monkeypatch.setattr(rr, "_LOAD_FAILED", False)
    return conf


def _stub_model(monkeypatch, scores):
    model = SimpleNamespace(compute_score=lambda pairs, normalize: scores)
    monkeypatch.setattr(rr, "_model", lambda: model)
    return model


def _cands(n: int = 3) -> list[dict]:
    return [{"chunk_id": f"c{i}", "text": f"text {i}", "score_rrf": 1.0 - i / 10} for i in range(n)]


# ---------- 切片 0: 降级路径 ----------


def test_empty_candidates(monkeypatch):
    _conf(monkeypatch)
    assert rr.rerank("q", []) == []


def test_disabled_truncates_in_original_order(monkeypatch):
    _conf(monkeypatch, enabled=False, top_k=2)
    out = rr.rerank("q", _cands(3))
    assert [h["chunk_id"] for h in out] == ["c0", "c1"]
    assert all("score_rerank" not in h for h in out)


def test_model_unavailable_falls_back_to_original_order(monkeypatch):
    _conf(monkeypatch, top_k=2)
    monkeypatch.setattr(rr, "_model", lambda: None)
    out = rr.rerank("q", _cands(3))
    assert [h["chunk_id"] for h in out] == ["c0", "c1"]


def test_compute_score_failure_falls_back(monkeypatch):
    _conf(monkeypatch, top_k=3)

    def _boom(pairs, normalize):
        raise RuntimeError("cuda OOM")

    monkeypatch.setattr(rr, "_model", lambda: SimpleNamespace(compute_score=_boom))
    warnings: list[str] = []
    monkeypatch.setattr(rr.log, "warning", warnings.append)

    out = rr.rerank("q", _cands(3))
    assert [h["chunk_id"] for h in out] == ["c0", "c1", "c2"]
    assert any("compute_score failed" in w for w in warnings)


def test_load_failed_latch_never_retries(monkeypatch):
    _conf(monkeypatch)
    monkeypatch.setattr(rr, "_LOAD_FAILED", True)
    assert rr._model() is None  # 闩锁生效: 不再尝试 import/加载


# ---------- 切片 1: 重排语义 ----------


def test_reranks_by_score_desc_and_writes_score_rerank(monkeypatch):
    _conf(monkeypatch, top_k=3)
    _stub_model(monkeypatch, [0.1, 0.9, 0.5])  # c1 最相关

    out = rr.rerank("q", _cands(3))

    assert [h["chunk_id"] for h in out] == ["c1", "c2", "c0"]
    assert [h["score_rerank"] for h in out] == [0.9, 0.5, 0.1]
    assert out[0]["score_rrf"] == 0.9  # 原字段保留


def test_single_pair_float_score_compat(monkeypatch):
    _conf(monkeypatch)
    _stub_model(monkeypatch, 0.7)  # FlagReranker 单对时返回裸 float

    out = rr.rerank("q", _cands(1))
    assert out[0]["score_rerank"] == 0.7


def test_top_k_caps_output(monkeypatch):
    _conf(monkeypatch, top_k=2)
    _stub_model(monkeypatch, [0.1, 0.9, 0.5])

    assert [h["chunk_id"] for h in rr.rerank("q", _cands(3))] == ["c1", "c2"]
    assert len(rr.rerank("q", _cands(3), top_k=1)) == 1  # 显式参数覆盖配置


# ---------- 切片 2: 不改写输入 ----------


def test_inputs_not_mutated(monkeypatch):
    _conf(monkeypatch, top_k=3)
    _stub_model(monkeypatch, [0.1, 0.9, 0.5])
    cands = _cands(3)

    out = rr.rerank("q", cands)

    assert [h["chunk_id"] for h in cands] == ["c0", "c1", "c2"]  # 基准会被原地重排
    assert all("score_rerank" not in h for h in cands)  # 基准会被写入新键
    assert out is not cands
