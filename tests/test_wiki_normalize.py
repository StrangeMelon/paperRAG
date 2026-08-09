"""wiki/normalize.py 三级概念解析的行为契约。

解析结果只有三种: match(高置信同一概念) / novel(新概念) / review(歧义, 不建不并)。
钉死的关键规则:
- 词面精确命中且非短标签且唯一 -> 直接 match, 不触发向量与 LLM;
- 短标签(RL 等缩写)即使词面命中也不许单独合并, 必须经 LLM 验证;
- 向量只负责召回(recall_floor), 跨语言合并判定权在 LLM 验证;
  同语言且相似度 >= auto_merge_same_lang 才允许免验证合并;
- LLM 判 different -> novel; 判 unsure 或调用失败 -> review(宁可复核, 不错并);
- 召回无候选 -> novel。
"""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _normalize_module() -> ModuleType:
    return importlib.import_module("paper_rag.wiki.normalize")


def _fake_resolve_config():
    return SimpleNamespace(
        wiki=SimpleNamespace(
            resolve=SimpleNamespace(
                recall_floor=0.60,
                auto_merge_same_lang=0.90,
                short_label_max_ascii_chars=4,
                short_label_max_cjk_chars=1,
            ),
        ),
    )


def _rl_entry() -> WikiEntry:
    return WikiEntry(
        entry_id="concept:reinforcementlearning",
        name="Reinforcement Learning",
        category="method",
        definition="A learning paradigm driven by reward signals.",
        definition_language="en",
        labels=[WikiLabel(text="Reinforcement Learning", language="en", kind="primary")],
    )


def _setup(monkeypatch, *, find_by_label=None, search_hits=None, judge=None, entry=None):
    """注入三级解析的全部外部依赖, 返回调用记录。"""
    mod = _normalize_module()
    calls = {"embed": 0, "judge": 0}

    monkeypatch.setattr(mod.cfg, "load", _fake_resolve_config)
    monkeypatch.setattr(mod.wstore, "find_by_label", find_by_label or (lambda text: []))
    monkeypatch.setattr(mod.wstore, "get_entry", lambda entry_id, **kw: entry or _rl_entry())

    def _fake_embed(text):
        calls["embed"] += 1
        return [0.0] * 4

    monkeypatch.setattr(mod, "_embed", _fake_embed)
    monkeypatch.setattr(mod.wstore, "search_qdrant", lambda vec, top_k=5: search_hits or [])

    def _fake_judge(name, language, definition_hint, candidates):
        calls["judge"] += 1
        if judge is None:
            raise AssertionError("LLM judge should not be called in this scenario")
        return judge(name, language, definition_hint, candidates)

    monkeypatch.setattr(mod, "_llm_judge", _fake_judge)
    return mod, calls


def test_exact_unique_label_matches_without_vector_or_llm(monkeypatch):
    mod, calls = _setup(
        monkeypatch,
        find_by_label=lambda text: ["concept:reinforcementlearning"],
    )
    res = mod.resolve_concept("reinforcement-learning", language="en")
    assert res["decision"] == "match"
    assert res["entry_id"] == "concept:reinforcementlearning"
    assert calls["embed"] == 0
    assert calls["judge"] == 0


def test_exact_hit_with_short_label_requires_llm_verification(monkeypatch):
    mod, calls = _setup(
        monkeypatch,
        find_by_label=lambda text: ["concept:reinforcementlearning"],
        judge=lambda *a: {"decision": "same", "entry_id": "concept:reinforcementlearning"},
    )
    res = mod.resolve_concept("RL", language="en")
    assert res["decision"] == "match"
    assert res["entry_id"] == "concept:reinforcementlearning"
    assert calls["judge"] == 1


def test_same_language_high_similarity_auto_merges(monkeypatch):
    hits = [
        {
            "entry_id": "concept:reinforcementlearning",
            "name": "Reinforcement Learning",
            "definition_language": "en",
            "definition_excerpt": "A learning paradigm driven by reward signals.",
            "score": 0.95,
        }
    ]
    mod, calls = _setup(monkeypatch, search_hits=hits)
    res = mod.resolve_concept("Deep Reinforcement Learning Paradigm", language="en")
    assert res["decision"] == "match"
    assert res["entry_id"] == "concept:reinforcementlearning"
    assert calls["embed"] == 1
    assert calls["judge"] == 0


def test_cross_language_candidate_goes_through_llm(monkeypatch):
    # 中文概念命中英文词条: 即使 0.95 也不免验证 —— 跨语言没有独立阈值
    hits = [
        {
            "entry_id": "concept:reinforcementlearning",
            "name": "Reinforcement Learning",
            "definition_language": "en",
            "definition_excerpt": "A learning paradigm driven by reward signals.",
            "score": 0.95,
        }
    ]
    mod, calls = _setup(
        monkeypatch,
        search_hits=hits,
        judge=lambda *a: {"decision": "same", "entry_id": "concept:reinforcementlearning"},
    )
    res = mod.resolve_concept("强化学习", language="zh")
    assert res["decision"] == "match"
    assert res["entry_id"] == "concept:reinforcementlearning"
    assert calls["judge"] == 1


def test_llm_different_yields_novel(monkeypatch):
    # REINFORCE 召回强化学习词条但 LLM 判不同概念 -> novel, 不误并
    hits = [
        {
            "entry_id": "concept:reinforcementlearning",
            "name": "Reinforcement Learning",
            "definition_language": "en",
            "definition_excerpt": "A learning paradigm driven by reward signals.",
            "score": 0.78,
        }
    ]
    mod, _ = _setup(
        monkeypatch,
        search_hits=hits,
        judge=lambda *a: {"decision": "different", "entry_id": None},
    )
    res = mod.resolve_concept("REINFORCE algorithm", language="en")
    assert res["decision"] == "novel"


def test_llm_unsure_or_failure_yields_review(monkeypatch):
    hits = [
        {
            "entry_id": "concept:reinforcementlearning",
            "name": "Reinforcement Learning",
            "definition_language": "en",
            "definition_excerpt": "A learning paradigm driven by reward signals.",
            "score": 0.7,
        }
    ]
    mod, _ = _setup(
        monkeypatch,
        search_hits=hits,
        judge=lambda *a: {"decision": "unsure", "entry_id": None},
    )
    res = mod.resolve_concept("逆强化学习", language="zh")
    assert res["decision"] == "review"
    assert res["candidates"]

    def _boom(*a):
        raise RuntimeError("llm down")

    mod2, _ = _setup(monkeypatch, search_hits=hits, judge=_boom)
    res2 = mod2.resolve_concept("逆强化学习", language="zh")
    assert res2["decision"] == "review"


def test_no_candidates_above_floor_is_novel(monkeypatch):
    hits = [{"entry_id": "concept:x", "name": "x", "definition_language": "en", "score": 0.4}]
    mod, calls = _setup(monkeypatch, search_hits=hits)
    res = mod.resolve_concept("Novel Brand New Concept", language="en")
    assert res["decision"] == "novel"
    assert calls["judge"] == 0


def test_empty_normalized_name_is_review(monkeypatch):
    mod, calls = _setup(monkeypatch)
    res = mod.resolve_concept("---", language=None)
    assert res["decision"] == "review"
    assert calls["embed"] == 0
