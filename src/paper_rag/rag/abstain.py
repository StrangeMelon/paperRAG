"""证据充分性三路裁决——硬不变量"无证据宁拒答"的执行者。

为什么存在
----------
检索永远返回"最相似"的 top-k, 哪怕最相似的东西毫不相关: 拿域外问题(如
"上海明天天气")问论文库, RRF/重排照样排出一列噪声块, 下游 LLM 会一本正经
地引用它们作答。本模块在 **LLM 调用之前** 给 qa_agentic/qa_stream 一次
第一等的证据充分性裁决。

裁决协议
--------
输入为已融合 + 精排 + 截断后的最终块列表:

    no_chunks       — chunks == []
    no_evidence     — evidence_score < threshold_low       (跳过 LLM)
    weak_evidence   — low <= score < threshold_high        (LLM 照常调用,
                      prompt 注入"证据可能不足"提示)
    confident       — score >= threshold_high              (正常流程)

`evidence_score` 取块列表里最优可用分数字段(优先级可配, 缺省
rerank > dense > score > bm25 > rrf)的 top-`min_chunks` 归一化均值。RRF
分数量级 ~(0, 0.05], 线性放大进 [0,1] 带后再比阈值——阈值语义在 reranker
开/关时保持稳定。

设计纪律(与基准一致)
--------------------
1. **纯函数** — `decide()` 只吃 list[dict] + 阈值, 返回 typed dict; 无 I/O
   无日志, 打点由调用方做。
2. **逃生门** — `enabled=False` 时恒 confident, 行为等价于没有本模块。
3. **fail open** — 分数字段缺失/坏值时放行(confident)而不是阻塞流水线;
   拒答模块自己不可靠时宁可放行, 降级态用 `signal_quality` 透出给指标。
4. **可观测** — 每次裁决回显分数、所用字段与阈值, 调用方进 trace/metrics。
5. **可校准** — 阈值来自 `cfg.rag.abstain`(YAML 校准值 0.21/0.48), 代码里
   不藏 magic number; 签名缺省 0.20/0.40 只是保守的文档化缺省。

中文扩展(相对基准的确认偏离)
----------------------------
a) `weak_evidence_hint(language)` zh/en 路由——基准把英文提示硬编码进
   `WEAK_EVIDENCE_HINT`, 中文问题的 prompt 里插英文警告风格割裂; zh 返回
   中文文案, en/None/未知语言走基准英文(未知语言不猜)。
b) `no_chunks` 早退分支补 `signal_quality="no_chunks"` 键——基准该分支漏键,
   四态返回 schema 不一致, trace 消费方被迫做缺键防御。

核心打分主体语言无关: 分数来自 BGE-M3 / bge-reranker(双语模型)。BM25
sigmoid center=8 按英文语料校准, 中文 FTS5 bigram 分布不同——但低质字段
一律 fail open, 该常数从不影响裁决, 只影响 trace 展示数字(已记账)。
"""

from __future__ import annotations

from collections.abc import Iterable

# 决策取值别名, 便于调用方注解
Decision = str  # confident | weak_evidence | no_evidence | no_chunks

DECISION_CONFIDENT = "confident"
DECISION_WEAK = "weak_evidence"
DECISION_NO_EVIDENCE = "no_evidence"
DECISION_NO_CHUNKS = "no_chunks"

# 信号质量分级: 只有高质信号(真实相似度)能区分"无关块排前"与"相关块排前"。
# 排名型信号(RRF)只反映相对名次做不到; BM25 对域外问题也可能碰巧词面命中。
# 因此低质信号下 fail open(confident), 降级态作为独立指标透出。
HIGH_QUALITY_FIELDS = frozenset({"score_rerank", "score_dense", "score"})
LOW_QUALITY_FIELDS = frozenset({"score_bm25", "score_rrf"})

# RRF 分数是 1/(k+rank) 的和, k=60 时典型落在 (0, 0.05]。线性放大到 ~(0, 1]
# 再比阈值; 系数取 15 使单列表 rank-1 的 0.033 映射到 ~0.5。
_RRF_NORMALIZE_FACTOR = 15.0

# BM25 原始分无界(典型 0-30), 用 center=8(库内查询典型 rank-1 分)的软
# sigmoid 压进 [0,1]。仅在 dense 不可用的降级模式下作为展示兜底。
_BM25_SIGMOID_CENTER = 8.0
_BM25_SIGMOID_SLOPE = 0.5


def _normalize(score: float, field: str) -> float:
    """把单块分数拉进 [0,1] 带, 使阈值语义在 reranker 开/关配置间稳定。"""
    if field == "score_rrf":
        # RRF: 线性放大后裁剪到 [0, 1]
        return max(0.0, min(1.0, score * _RRF_NORMALIZE_FACTOR))
    if field == "score_bm25":
        # BM25 无界, sigmoid 压进 [0,1] 带(仅降级模式的展示兜底)。
        import math

        z = _BM25_SIGMOID_SLOPE * (score - _BM25_SIGMOID_CENTER)
        return 1.0 / (1.0 + math.exp(-z))
    # score_rerank: bge-reranker 的 sigmoid 输出, 天然 0..1
    # score / score_dense: bge-m3 cosine 理论 [-1,1] 实际相似对 ~[0,1], 裁剪保险
    return max(0.0, min(1.0, score))


def evidence_score(
    chunks: list[dict],
    *,
    score_fields: tuple[str, ...] = (
        "score_rerank",
        "score_dense",
        "score",
        "score_bm25",
        "score_rrf",
    ),
    min_chunks: int = 3,
) -> tuple[float, str | None, int]:
    """从块列表聚合出证据分。

    Returns
    -------
    (score, field_used, n_used)
        score      — top-`min_chunks` 块归一化分的均值; 无任何可用分数字段
                     时回退 0.0。
        field_used — 实际选中的分数字段(或 None)。
        n_used     — 参与均值的块数。
    """
    if not chunks:
        return 0.0, None, 0

    raw_scores: list[float] = []
    field_used = _best_available_field(chunks, score_fields)
    if field_used is None:
        return 0.0, None, 0
    for ch in chunks:
        value = ch.get(field_used)
        if value is None:
            continue
        try:
            raw_scores.append(_normalize(float(value), field_used))
        except (TypeError, ValueError):
            continue
    if not raw_scores:
        return 0.0, None, 0
    raw_scores.sort(reverse=True)
    take = raw_scores[:min_chunks] if min_chunks > 0 else raw_scores
    return sum(take) / len(take), field_used, len(take)


def _best_available_field(chunks: list[dict], score_fields: Iterable[str]) -> str | None:
    for field in score_fields:
        for ch in chunks:
            value = ch.get(field)
            if value is None:
                continue
            try:
                float(value)
                return field
            except (TypeError, ValueError):
                continue
    return None


def _classify(
    *,
    enabled: bool,
    field_used: str | None,
    score: float,
    threshold_low: float,
    threshold_high: float,
) -> tuple[str, str]:
    """纯裁决表: 给定校准输入, 返回 (decision, signal_quality)。

    从 evidence_score() 的取分代码里拆出来, 阈值表可独立单测。6 分支无 I/O。
    """
    if not enabled:
        return DECISION_CONFIDENT, "disabled"
    if field_used is None:
        # 无可用分数字段——放行而不是阻塞流水线。
        return DECISION_CONFIDENT, "missing"
    if field_used in LOW_QUALITY_FIELDS:
        # 排名型/无界分数(BM25/RRF)撑不起"均值低即域外"的假设。放行,
        # 降级态经 signal_quality 透出。
        return DECISION_CONFIDENT, "low_degraded"
    if score < threshold_low:
        return DECISION_NO_EVIDENCE, "high"
    if score < threshold_high:
        return DECISION_WEAK, "high"
    return DECISION_CONFIDENT, "high"


def _top_chunk_score(chunks: list[dict], field_used: str | None) -> float:
    if field_used is None or not chunks:
        return 0.0
    scores: list[float] = []
    for ch in chunks:
        try:
            scores.append(_normalize(float(ch.get(field_used, 0.0) or 0.0), field_used))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def decide(
    chunks: list[dict],
    *,
    enabled: bool = True,
    threshold_low: float = 0.20,
    threshold_high: float = 0.40,
    min_chunks: int = 3,
    score_fields: tuple[str, ...] = (
        "score_rerank",  # bge-reranker 输出(可用时的最佳信号)
        "score_dense",  # bge-m3 cosine(真实语义相似度)
        "score",  # 兜底别名(qdrant_store 出口设 `score`)
        "score_bm25",  # 降级兜底(dense 不可用)
        "score_rrf",  # 排名型, 最后手段(测不出"无证据")
    ),
) -> dict:
    """做一次拒答裁决。

    Parameters
    ----------
    chunks : 检索结果 dict 列表(已截断到 LLM 将看到的范围)。
    enabled : 逃生门。False 时恒 confident(历史行为)。
    threshold_low : 低于此值 -> no_evidence(跳过 LLM)。
    threshold_high : 达到此值 -> confident(正常流程)。
    min_chunks : 参与证据分均值的 top 块数。
    score_fields : 按优先级顺序查询的分数字段。

    Returns
    -------
    dict, 键: decision, evidence_score, top_chunk_score, n_chunks,
    score_field, signal_quality, threshold_low, threshold_high, enabled。
    """
    n_chunks = len(chunks)
    if n_chunks == 0:
        return {
            "decision": DECISION_NO_CHUNKS,
            "evidence_score": 0.0,
            "top_chunk_score": 0.0,
            "n_chunks": 0,
            "score_field": None,
            "signal_quality": "no_chunks",  # 确认偏离 b: 基准漏此键, 补齐拉平 schema
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "enabled": enabled,
        }

    score, field_used, _ = evidence_score(chunks, score_fields=score_fields, min_chunks=min_chunks)
    top_score = _top_chunk_score(chunks, field_used)
    decision, signal_quality = _classify(
        enabled=enabled,
        field_used=field_used,
        score=score,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )

    return {
        "decision": decision,
        "evidence_score": round(score, 4),
        "top_chunk_score": round(top_score, 4),
        "n_chunks": n_chunks,
        "score_field": field_used,
        "signal_quality": signal_quality,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "enabled": enabled,
    }


# decision == weak_evidence 时注入 prompt 的提示后缀(基准英文原文)
WEAK_EVIDENCE_HINT = (
    "\n\nNOTE: The retrieved evidence appears WEAK or only tangentially "
    "related to the question. If you cannot answer with high confidence "
    "using the evidence above, explicitly say so — do NOT compensate with "
    "general knowledge or fabricated citations."
)

# 确认偏离 a: 中文问题的 prompt 用中文提示, 避免风格割裂降低遵从度
WEAK_EVIDENCE_HINT_ZH = (
    "\n\n注意：检索到的证据与问题相关性较弱或仅间接相关。如果无法基于上述"  # noqa: RUF001
    "证据以高置信度回答，请明确说明——不要用通用知识补偿，也不要编造引用。"  # noqa: RUF001
)


def weak_evidence_hint(language: str | None = None) -> str:
    """按问题语言路由弱证据提示; zh 返回中文, 其余(en/None/未知)走基准英文。"""
    return WEAK_EVIDENCE_HINT_ZH if language == "zh" else WEAK_EVIDENCE_HINT


__all__ = [
    "DECISION_CONFIDENT",
    "DECISION_NO_CHUNKS",
    "DECISION_NO_EVIDENCE",
    "DECISION_WEAK",
    "WEAK_EVIDENCE_HINT",
    "WEAK_EVIDENCE_HINT_ZH",
    "decide",
    "evidence_score",
    "weak_evidence_hint",
]
