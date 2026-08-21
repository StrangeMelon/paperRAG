"""QA 生成前的确定性证据选择。

检索出口是一个宽检查窗(intent 三档 5/10/15 块); 交给 LLM 引用的证据集必须
更小——上下文纪律(相关块不被噪声稀释)与引用面控制(给得越多越容易乱引)。

选择器**确定性**: 不调 LLM, 纯打分排序。rerank/RRF 分数占大头, 词面重叠做
平局裁决, 章节提示微加分, 原始排名兜底锚。同一输入恒同一输出, trace 逐候选
记账四项得分, 可复盘"为什么选了这块没选那块"。

与基准的偏离(中文扩展, 逐条有测试钉死):
  1. 词面重叠 token 化从 `[a-z0-9]+` 改为"拉丁词 + CJK bigram"并集——基准对
     中文问题一个 token 都抽不出, overlap 恒 0, 平局裁决层对中文整体失明;
     bigram 口径与 FTS5 的 ADR-0001 一致, 问题侧与文本侧同一函数。
  2. `_SECTION_HINTS` 增中文条目(摘要/引言/方法/实验/结果/结论等)——基准全
     英文提示表, 中文块永远拿不到章节加分。
打分权重(0.2/0.03/0.001)与 `max_chunks=4, max_per_paper=2` 签名默认照抄基准;
是否配置化推迟到 qa_agentic 课由调用方决定。
"""

from __future__ import annotations

import re
from collections import Counter

from ..retrieve.reference_policy import detect_reference_intent, filter_answer_evidence

_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
_SECTION_HINTS = (
    "abstract",
    "introduction",
    "method",
    "approach",
    "experiment",
    "evaluation",
    "result",
    "conclusion",
    # 中文章节名(基准全英文, 中文块拿不到加分)
    "摘要",
    "引言",
    "绪论",
    "方法",
    "实验",
    "评估",
    "结果",
    "结论",
    "总结",
)


def select_evidence(
    question: str,
    chunks: list[dict],
    *,
    intent: str | None = None,
    max_chunks: int = 4,
    max_per_paper: int = 2,
    reference_intent: bool | None = None,
) -> tuple[list[dict], dict]:
    """从检索块中挑出紧凑、可引用的证据集。

    确定性选择: rerank/RRF 分数承担大部分权重, 词面重叠裁决平局,
    章节/标题提示提供微小推力。
    """
    input_chunks = list(chunks)
    resolved_reference_intent = (
        detect_reference_intent(question) if reference_intent is None else reference_intent
    )
    chunks = filter_answer_evidence(
        input_chunks,
        reference_intent=resolved_reference_intent,
    )
    eligible_ids = {id(chunk) for chunk in chunks}
    excluded_reference_chunk_ids = [
        chunk.get("chunk_id")
        for chunk in input_chunks
        if id(chunk) not in eligible_ids and chunk.get("chunk_id")
    ]
    if not chunks:
        return [], {
            "strategy": "deterministic_score_overlap",
            "selected_chunk_ids": [],
            "input_chunk_ids": [
                chunk.get("chunk_id") for chunk in input_chunks if chunk.get("chunk_id")
            ],
            "excluded_reference_chunk_ids": excluded_reference_chunk_ids,
            "max_chunks": max_chunks,
            "max_per_paper": max_per_paper,
            "candidates": [],
        }

    scored = []
    for rank, chunk in enumerate(chunks, 1):
        features = _score_chunk(question, chunk, rank)
        scored.append((features["selection_score"], rank, chunk, features))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for _, _, chunk, _ in scored:
        paper_id = str(chunk.get("paper_id") or "")
        if paper_id and counts[paper_id] >= max_per_paper:
            continue
        selected.append(chunk)
        if paper_id:
            counts[paper_id] += 1
        if len(selected) >= max_chunks:
            break

    trace = {
        "strategy": "deterministic_score_overlap",
        "intent": intent or "unknown",
        "max_chunks": max_chunks,
        "max_per_paper": max_per_paper,
        "input_chunk_ids": [c.get("chunk_id") for c in input_chunks if c.get("chunk_id")],
        "excluded_reference_chunk_ids": excluded_reference_chunk_ids,
        "selected_chunk_ids": [c.get("chunk_id") for c in selected if c.get("chunk_id")],
        "candidates": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "paper_id": chunk.get("paper_id"),
                "rank": rank,
                **features,
                "selected": chunk in selected,
            }
            for _, rank, chunk, features in scored
        ],
    }
    return selected, trace


def _score_chunk(question: str, chunk: dict, rank: int) -> dict:
    text = " ".join(
        str(chunk.get(key) or "") for key in ("title", "section", "text", "raw_snippet")
    )
    overlap = _lexical_overlap(question, text)
    model_score = _model_score(chunk)
    section_hint = _section_hint(chunk)
    rank_bonus = 1.0 / max(rank, 1)
    selection_score = model_score + 0.2 * overlap + 0.03 * section_hint + 0.001 * rank_bonus
    return {
        "selection_score": round(selection_score, 6),
        "model_score": round(model_score, 6),
        "lexical_overlap": round(overlap, 6),
        "section_hint": section_hint,
    }


def _model_score(chunk: dict) -> float:
    for key in ("score_rerank", "score_rrf", "score_dense", "score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _tokens(text: str) -> set[str]:
    """拉丁词 + CJK bigram 并集(单字 CJK 组退化为单字 token, 不空集)。"""
    lowered = text.lower()
    tokens = set(_LATIN_TOKEN_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _lexical_overlap(question: str, text: str) -> float:
    q_tokens = _tokens(question)
    if not q_tokens:
        return 0.0
    return len(q_tokens & _tokens(text)) / len(q_tokens)


def _section_hint(chunk: dict) -> int:
    section = str(chunk.get("section") or "").lower()
    title = str(chunk.get("title") or "").lower()
    haystack = f"{section} {title}"
    return int(any(hint in haystack for hint in _SECTION_HINTS))


__all__ = ["select_evidence"]
