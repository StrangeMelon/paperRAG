"""rag.abstain 三路证据充分性裁决的行为契约测试(纯函数, 无 LLM/IO)。

切片 0: 四态判定与阈值边界(no_chunks / no_evidence / weak / confident;
        恰好 == low 归 weak、恰好 == high 归 confident)。
切片 1: 字段优先级与单字段纪律(rerank > dense > score > bm25 > rrf; 首个
        "任一块携带"的字段服务全列表, 不跨字段混用)。
切片 2: 归一化(rerank/dense/score 裁剪 [0,1]; RRF 线性 x15 再裁剪; BM25
        sigmoid center=8)。
切片 3: fail open 三分支(enabled=False -> disabled; 字段全缺 -> missing;
        低质字段 bm25/rrf -> low_degraded, 分数再低也放行)。
切片 4: 坏值容错与 min_chunks(非数值/None 逐块跳过; top-N 均值; N<=0 取全部)。
切片 5: 返回 schema(四态逐键拉平, no_chunks 也带 signal_quality——基准该
        分支漏键, 重建版确认偏离补齐; 分数四舍五入 4 位; 配置回显)。
切片 6: 中文扩展(weak_evidence_hint(language) zh/en 路由——基准硬编码英文
        提示, 中文问题的 prompt 风格割裂)。

接口约定(与基准一致):

    decide(chunks, *, enabled=True, threshold_low=0.20, threshold_high=0.40,
           min_chunks=3, score_fields=(...)) -> dict
    evidence_score(chunks, *, score_fields=(...), min_chunks=3)
        -> (score, field_used, n_used)
"""

from __future__ import annotations

import pytest

from paper_rag.rag.abstain import (
    DECISION_CONFIDENT,
    DECISION_NO_CHUNKS,
    DECISION_NO_EVIDENCE,
    DECISION_WEAK,
    WEAK_EVIDENCE_HINT,
    WEAK_EVIDENCE_HINT_ZH,
    decide,
    evidence_score,
    weak_evidence_hint,
)


def _chunks(field: str, *scores) -> list[dict]:
    return [{"chunk_id": f"c{i}", field: s} for i, s in enumerate(scores)]


# ---------- 切片 0: 四态判定与阈值边界 ----------


def test_empty_chunks_is_no_chunks():
    result = decide([])
    assert result["decision"] == DECISION_NO_CHUNKS
    assert result["evidence_score"] == 0.0
    assert result["n_chunks"] == 0
    assert result["score_field"] is None


def test_empty_chunks_is_no_chunks_even_when_disabled():
    # 照抄基准: n_chunks==0 早退先于 enabled 检查(docstring 说 disabled 恒
    # confident, 代码实际以空表早退优先; 以代码行为为准)。
    assert decide([], enabled=False)["decision"] == DECISION_NO_CHUNKS


def test_low_score_is_no_evidence():
    result = decide(_chunks("score_rerank", 0.05, 0.04, 0.03))
    assert result["decision"] == DECISION_NO_EVIDENCE
    assert result["signal_quality"] == "high"


def test_mid_score_is_weak_evidence():
    result = decide(_chunks("score_rerank", 0.30, 0.30, 0.30))
    assert result["decision"] == DECISION_WEAK


def test_high_score_is_confident():
    result = decide(_chunks("score_rerank", 0.90, 0.85, 0.80))
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "high"


def test_score_exactly_low_threshold_is_weak():
    # score < low 才 no_evidence; == low 落入 weak 档。
    result = decide(_chunks("score_rerank", 0.20, 0.20, 0.20))
    assert result["decision"] == DECISION_WEAK


def test_score_exactly_high_threshold_is_confident():
    result = decide(_chunks("score_rerank", 0.40, 0.40, 0.40))
    assert result["decision"] == DECISION_CONFIDENT


def test_custom_thresholds_are_respected():
    result = decide(
        _chunks("score_rerank", 0.50, 0.50, 0.50), threshold_low=0.60, threshold_high=0.80
    )
    assert result["decision"] == DECISION_NO_EVIDENCE


# ---------- 切片 1: 字段优先级与单字段纪律 ----------


def test_rerank_preferred_over_dense():
    chunks = [{"chunk_id": "c0", "score_rerank": 0.9, "score_dense": 0.1}]
    score, field, _ = evidence_score(chunks)
    assert field == "score_rerank"
    assert score == pytest.approx(0.9)


def test_first_available_field_serves_whole_list():
    # 只有一块带 rerank, 其余只带 dense: 字段选 rerank, 其余块不参与均值,
    # 不允许跨字段混用(不同量纲的均值无意义)。
    chunks = [
        {"chunk_id": "c0", "score_rerank": 0.8},
        {"chunk_id": "c1", "score_dense": 0.9},
        {"chunk_id": "c2", "score_dense": 0.9},
    ]
    score, field, n_used = evidence_score(chunks)
    assert field == "score_rerank"
    assert n_used == 1
    assert score == pytest.approx(0.8)


def test_dense_used_when_no_rerank():
    score, field, _ = evidence_score(_chunks("score_dense", 0.7, 0.6, 0.5))
    assert field == "score_dense"
    assert score == pytest.approx((0.7 + 0.6 + 0.5) / 3)


def test_plain_score_fallback():
    _, field, _ = evidence_score(_chunks("score", 0.7, 0.6))
    assert field == "score"


# ---------- 切片 2: 归一化 ----------


def test_dense_scores_clipped_to_unit_interval():
    score, _, _ = evidence_score(_chunks("score_dense", 1.4, -0.2, 0.5))
    assert score == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_rrf_scores_scaled_linearly_then_clipped():
    # RRF 量级 ~(0, 0.05], x15 拉进 [0,1] 带; 0.1x15=1.5 裁到 1.0。
    score, field, _ = evidence_score(_chunks("score_rrf", 0.02, 0.02, 0.1))
    assert field == "score_rrf"
    assert score == pytest.approx((0.3 + 0.3 + 1.0) / 3)


def test_bm25_sigmoid_center_maps_to_half():
    # 无界 BM25 用 center=8 的 sigmoid 压扁: 恰好 8 分 -> 0.5。
    score, field, _ = evidence_score(_chunks("score_bm25", 8.0, 8.0, 8.0))
    assert field == "score_bm25"
    assert score == pytest.approx(0.5)


# ---------- 切片 3: fail open 三分支 ----------


def test_disabled_kill_switch_always_confident():
    result = decide(_chunks("score_rerank", 0.01), enabled=False)
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "disabled"


def test_no_usable_field_fails_open():
    result = decide([{"chunk_id": "c0", "text": "no scores at all"}])
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "missing"
    assert result["score_field"] is None


def test_bm25_only_fails_open_even_when_low():
    # 排名/词面型信号区分不了"无关块排前"与"相关块排前", 低分不触发拒答,
    # 降级态经 signal_quality 透出。
    result = decide(_chunks("score_bm25", 0.1, 0.1, 0.1))
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "low_degraded"


def test_rrf_only_fails_open_even_when_low():
    result = decide(_chunks("score_rrf", 0.001, 0.001))
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "low_degraded"


# ---------- 切片 4: 坏值容错与 min_chunks ----------


def test_non_numeric_values_skipped_per_chunk():
    chunks = [
        {"chunk_id": "c0", "score_rerank": "n/a"},
        {"chunk_id": "c1", "score_rerank": None},
        {"chunk_id": "c2", "score_rerank": 0.6},
    ]
    score, field, n_used = evidence_score(chunks)
    assert field == "score_rerank"
    assert n_used == 1
    assert score == pytest.approx(0.6)


def test_all_values_bad_falls_back_to_missing():
    result = decide([{"chunk_id": "c0", "score_rerank": "n/a"}])
    assert result["decision"] == DECISION_CONFIDENT
    assert result["signal_quality"] == "missing"


def test_numeric_string_is_accepted():
    score, _, _ = evidence_score([{"chunk_id": "c0", "score_rerank": "0.5"}])
    assert score == pytest.approx(0.5)


def test_min_chunks_takes_top_n_mean():
    # 5 块取 top-3 均值: 低分尾巴不稀释判定。
    score, _, n_used = evidence_score(_chunks("score_rerank", 1.0, 1.0, 1.0, 0.0, 0.0))
    assert n_used == 3
    assert score == pytest.approx(1.0)


def test_min_chunks_nonpositive_means_all():
    score, _, n_used = evidence_score(_chunks("score_rerank", 1.0, 0.0), min_chunks=0)
    assert n_used == 2
    assert score == pytest.approx(0.5)


def test_fewer_chunks_than_min_uses_what_exists():
    score, _, n_used = evidence_score(_chunks("score_rerank", 0.8), min_chunks=3)
    assert n_used == 1
    assert score == pytest.approx(0.8)


# ---------- 切片 5: 返回 schema ----------

_SCHEMA_KEYS = {
    "decision",
    "evidence_score",
    "top_chunk_score",
    "n_chunks",
    "score_field",
    "signal_quality",
    "threshold_low",
    "threshold_high",
    "enabled",
}


def test_schema_identical_across_all_four_decisions():
    # 确认偏离 b: 基准 no_chunks 早退分支漏 signal_quality 键, 重建版补齐
    # "no_chunks", 四态 schema 拉平, trace 消费方无需做缺键防御。
    results = [
        decide([]),
        decide(_chunks("score_rerank", 0.05, 0.05, 0.05)),
        decide(_chunks("score_rerank", 0.30, 0.30, 0.30)),
        decide(_chunks("score_rerank", 0.90, 0.90, 0.90)),
    ]
    for r in results:
        assert set(r.keys()) == _SCHEMA_KEYS
    assert results[0]["signal_quality"] == "no_chunks"


def test_scores_rounded_to_four_decimals():
    result = decide(_chunks("score_rerank", 0.123456, 0.123456, 0.123456))
    assert result["evidence_score"] == 0.1235
    assert result["top_chunk_score"] == 0.1235


def test_top_chunk_score_is_max_of_used_field():
    result = decide(_chunks("score_rerank", 0.2, 0.9, 0.5))
    assert result["top_chunk_score"] == pytest.approx(0.9)


def test_config_echoed_back():
    result = decide(
        _chunks("score_rerank", 0.5), threshold_low=0.21, threshold_high=0.48, min_chunks=5
    )
    assert result["threshold_low"] == 0.21
    assert result["threshold_high"] == 0.48
    assert result["n_chunks"] == 1
    assert result["enabled"] is True


# ---------- 切片 6: 中文扩展 weak hint 路由 ----------


def test_weak_hint_routes_zh():
    assert weak_evidence_hint("zh") == WEAK_EVIDENCE_HINT_ZH
    assert "证据" in WEAK_EVIDENCE_HINT_ZH


def test_weak_hint_defaults_to_english():
    # en / None / 未知语言值统一走基准英文常量(未知语言不猜)。
    assert weak_evidence_hint("en") == WEAK_EVIDENCE_HINT
    assert weak_evidence_hint(None) == WEAK_EVIDENCE_HINT
    assert weak_evidence_hint("fr") == WEAK_EVIDENCE_HINT
    assert WEAK_EVIDENCE_HINT.startswith("\n\n")
