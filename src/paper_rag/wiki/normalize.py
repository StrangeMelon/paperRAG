"""三级概念解析: 候选概念名 -> match(并入既有词条) / novel(新建) / review(人工复核)。

解析顺序与判定权分配(ADR-0003):
  1. 词面精确命中(wiki_labels.text_norm 索引): 非短标签且唯一 -> 直接 match;
     多命中或短标签(RL/CL 级缩写、中文单字) -> 降级为候选, 交 LLM 验证。
  2. BGE-M3 向量召回: 只负责召回(recall_floor), 不做合并判定;
     唯一例外是同语言且相似度 >= auto_merge_same_lang 的候选可免验证合并——
     跨语言(zh<->en)相似度分布不同, 不设独立阈值, 一律走验证。
  3. LLM 验证: 带定义/类别/证据上下文判 same/different/unsure;
     same -> match, different -> novel, unsure 或调用失败 -> review。
     review 宁可积压复核, 也不错误合并或制造重复词条。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import config as cfg
from ..utils.logger import get_logger
from . import store as wstore
from .schema import is_short_label, label_language, normalize_label

log = get_logger(__name__)

_RECALL_TOP_K = 5
_MAX_JUDGE_CANDIDATES = 3

_JUDGE_PROMPT_EN = """You judge whether a candidate research concept is the SAME \
concept as one of the existing wiki entries below.

Candidate concept: {name}
Candidate definition/context (may be empty): {hint}

Existing entries:
{candidates}

Rules:
- "same" only if the candidate and one entry clearly denote the same concept
  (translations across Chinese/English count as the same concept).
- A specific algorithm/variant is NOT the same as its broader field
  (e.g. "REINFORCE algorithm" is not "Reinforcement Learning").
- If evidence is insufficient, answer "unsure".

Return ONLY JSON: {{"decision": "same" | "different" | "unsure", "entry_id": "<id or null>"}}
"""

_JUDGE_PROMPT_ZH = """判断候选研究概念与下列已有 wiki 词条中的某一条是否为**同一概念**。

候选概念: {name}
候选定义/上下文(可能为空): {hint}

已有词条:
{candidates}

规则:
- 只有当候选与某词条明确指同一概念时才答 "same"(中英互译视为同一概念)。
- 具体算法/变体不等于其所属领域(如 "REINFORCE algorithm" 不是 "强化学习")。
- 证据不足时答 "unsure"。

只返回 JSON: {{"decision": "same" | "different" | "unsure", "entry_id": "<id 或 null>"}}
"""


def _embed(text: str) -> list[float]:
    from ..embed import bge_m3

    return bge_m3.encode_one(text)


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for c in candidates:
        rows.append(
            f"- entry_id={c.get('entry_id')} name={c.get('name')} "
            f"lang={c.get('definition_language')} "
            f"definition={str(c.get('definition_excerpt') or '')[:300]}"
        )
    return "\n".join(rows)


def _llm_judge(
    name: str,
    language: str | None,
    definition_hint: str | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """LLM 同一性验证。返回 {"decision": same|different|unsure, "entry_id": ...}。
    解析失败或返回未知 entry_id 一律降级 unsure。"""
    from .llm import chat

    template = _JUDGE_PROMPT_ZH if language == "zh" else _JUDGE_PROMPT_EN
    prompt = (
        template.replace("{name}", name)
        .replace("{hint}", definition_hint or "")
        .replace("{candidates}", _format_candidates(candidates))
    )
    raw = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(m.group(0)) if m else {}
    decision = data.get("decision")
    entry_id = data.get("entry_id")
    valid_ids = {c.get("entry_id") for c in candidates}
    if decision == "same" and entry_id in valid_ids:
        return {"decision": "same", "entry_id": entry_id}
    if decision == "different":
        return {"decision": "different", "entry_id": None}
    return {"decision": "unsure", "entry_id": None}


def _entry_as_candidate(entry_id: str, *, score: float | None = None) -> dict[str, Any] | None:
    entry = wstore.get_entry(entry_id)
    if entry is None:
        return None
    return {
        "entry_id": entry.entry_id,
        "name": entry.name,
        "definition_language": entry.definition_language,
        "definition_excerpt": (entry.definition or "")[:500],
        "score": score,
    }


def _judge_or_review(
    name: str,
    language: str | None,
    definition_hint: str | None,
    candidates: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    try:
        verdict = _llm_judge(name, language, definition_hint, candidates)
    except Exception as e:
        log.warning(f"wiki resolve judge failed for {name!r}: {e}")
        return {"decision": "review", "entry_id": None, "candidates": candidates, "reason": reason}
    if verdict["decision"] == "same":
        return {
            "decision": "match",
            "entry_id": verdict["entry_id"],
            "candidates": candidates,
            "reason": f"{reason}+llm_same",
        }
    if verdict["decision"] == "different":
        return {
            "decision": "novel",
            "entry_id": None,
            "candidates": candidates,
            "reason": f"{reason}+llm_different",
        }
    return {
        "decision": "review",
        "entry_id": None,
        "candidates": candidates,
        "reason": f"{reason}+llm_unsure",
    }


def resolve_concept(
    name: str,
    *,
    language: str | None = None,
    definition_hint: str | None = None,
) -> dict[str, Any]:
    """解析候选概念名。返回 {"decision", "entry_id", "candidates", "reason"}。"""
    norm = normalize_label(name)
    if not norm:
        return {
            "decision": "review",
            "entry_id": None,
            "candidates": [],
            "reason": "empty_normalized_name",
        }
    if language is None:
        language = label_language(name)
    resolve_cfg = cfg.load().wiki.resolve
    short = is_short_label(
        name,
        max_ascii_chars=resolve_cfg.short_label_max_ascii_chars,
        max_cjk_chars=resolve_cfg.short_label_max_cjk_chars,
    )

    # 一级: 词面精确命中
    exact_ids = wstore.find_by_label(name)
    if exact_ids:
        if not short and len(exact_ids) == 1:
            return {
                "decision": "match",
                "entry_id": exact_ids[0],
                "candidates": [],
                "reason": "exact_label",
            }
        candidates = [c for eid in exact_ids if (c := _entry_as_candidate(eid))]
        if candidates:
            return _judge_or_review(
                name,
                language,
                definition_hint,
                candidates[:_MAX_JUDGE_CANDIDATES],
                reason="exact_label_ambiguous" if len(exact_ids) > 1 else "exact_label_short",
            )

    # 二级: 向量召回
    try:
        query = f"{name}\n{definition_hint}" if definition_hint else name
        hits = wstore.search_qdrant(_embed(query), top_k=_RECALL_TOP_K)
    except Exception as e:
        log.warning(f"wiki resolve vector recall failed for {name!r}: {e}")
        hits = []
    candidates = [h for h in hits if float(h.get("score") or 0.0) >= resolve_cfg.recall_floor]
    if not candidates:
        return {"decision": "novel", "entry_id": None, "candidates": [], "reason": "no_candidates"}

    # 同语言高相似免验证合并(候选与概念名都非短标签)
    top = candidates[0]
    if (
        not short
        and float(top.get("score") or 0.0) >= resolve_cfg.auto_merge_same_lang
        and top.get("definition_language") == language
        and not is_short_label(
            str(top.get("name") or ""),
            max_ascii_chars=resolve_cfg.short_label_max_ascii_chars,
            max_cjk_chars=resolve_cfg.short_label_max_cjk_chars,
        )
    ):
        return {
            "decision": "match",
            "entry_id": top.get("entry_id"),
            "candidates": candidates,
            "reason": "auto_merge_same_lang",
        }

    # 三级: LLM 验证
    return _judge_or_review(
        name,
        language,
        definition_hint,
        candidates[:_MAX_JUDGE_CANDIDATES],
        reason="vector_recall",
    )
