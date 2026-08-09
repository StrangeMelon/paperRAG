"""wiki/concept_extractor.py 概念抽取契约(LLM 打桩)。

钉死的关键行为:
- 采样不是"前 30 个 chunk": 排除参考文献, 按摘要/引言/结论/方法优先级采样,
  预算按语言分档(zh 4000 / en 6000 字符);
- evidence_chunk_ids 必须属于输入白名单, 越界 ID 被剔除(论文正文不可信);
- prompt 按文档语言路由中英模板, 且都要求 canonical_name 优先用标准英文术语
  (跨语言合并的第一道保障), 中文名进 labels_zh;
- LLM 失败/解析失败返回 [], 不炸入库链路; 概念数不超过 max_concepts。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _mod():
    return importlib.import_module("paper_rag.wiki.concept_extractor")


def _fake_config():
    return SimpleNamespace(
        wiki=SimpleNamespace(
            extract=SimpleNamespace(
                max_concepts=2,
                char_budget_zh=200,
                char_budget_en=300,
                exclude_sections=["references", "bibliography", "参考文献"],
            ),
        ),
        llm=SimpleNamespace(temperatures=SimpleNamespace(wiki=0.2)),
    )


def _chunks():
    return [
        {"chunk_id": "c-ref", "section": "参考文献", "text": "[1] Sutton et al. 强化学习." * 3},
        {"chunk_id": "c-m1", "section": "方法", "text": "本文方法基于策略梯度." * 3},
        {"chunk_id": "c-abs", "section": "摘要", "text": "本文研究强化学习."},
        {"chunk_id": "c-i1", "section": "引言", "text": "强化学习是一种学习范式." * 2},
        {"chunk_id": "c-con", "section": "结论", "text": "实验证明了方法有效性."},
    ]


def test_sampling_excludes_references_and_prioritizes(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    sampled = mod._sample_chunks(_chunks(), language="zh")
    ids = [c["chunk_id"] for c in sampled]
    assert "c-ref" not in ids  # 参考文献剔除
    assert ids[0] == "c-abs"  # 摘要优先
    assert ids.index("c-i1") < ids.index("c-m1")  # 引言先于方法


def test_sampling_respects_language_budget(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    # zh 预算 200 字符, 装不下全部块; 至少保住摘要
    sampled = mod._sample_chunks(_chunks(), language="zh")
    total = sum(len(c["text"]) for c in sampled)
    assert total <= 200
    assert sampled[0]["chunk_id"] == "c-abs"


def test_extract_validates_evidence_whitelist_and_caps(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    raw = """{
      "concepts": [
        {"surface_name": "强化学习", "canonical_name": "Reinforcement Learning",
         "category": "method", "labels_zh": ["强化学习"], "labels_en": ["RL"],
         "evidence_chunk_ids": ["c-abs", "c-hacked-out-of-whitelist"], "confidence": 0.9},
        {"surface_name": "策略梯度", "canonical_name": "Policy Gradient",
         "category": "method", "labels_zh": ["策略梯度"], "labels_en": [],
         "evidence_chunk_ids": ["c-m1"], "confidence": 0.8},
        {"surface_name": "第三个概念", "canonical_name": "Third",
         "category": "concept", "labels_zh": [], "labels_en": [],
         "evidence_chunk_ids": [], "confidence": 0.7}
      ]
    }"""
    monkeypatch.setattr(mod, "_chat", lambda prompt: raw)

    out = mod.extract_concepts(title="某论文", chunks=_chunks(), language="zh")
    assert len(out) == 2  # max_concepts=2 截断
    first = out[0]
    assert first["name"] == "Reinforcement Learning"  # canonical 优先英文
    assert first["surface_name"] == "强化学习"
    # 白名单外的 evidence id 被剔除
    assert first["evidence_chunk_ids"] == ["c-abs"]
    assert first["labels_zh"] == ["强化学习"]


def test_prompt_language_routing_and_injection_guard(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    captured = {}

    def _fake_chat(prompt):
        captured["prompt"] = prompt
        return '{"concepts": []}'

    monkeypatch.setattr(mod, "_chat", _fake_chat)

    mod.extract_concepts(title="中文论文", chunks=_chunks(), language="zh")
    assert "只返回 JSON" in captured["prompt"]  # 中文模板
    assert "不可信" in captured["prompt"]  # 注入防护声明

    mod.extract_concepts(title="EN paper", chunks=_chunks(), language="en")
    assert "Return ONLY JSON" in captured["prompt"]  # 英文模板
    assert "untrusted" in captured["prompt"]


def test_llm_failure_returns_empty(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    def _boom(prompt):
        raise RuntimeError("llm down")

    monkeypatch.setattr(mod, "_chat", _boom)
    assert mod.extract_concepts(title="t", chunks=_chunks(), language="en") == []

    monkeypatch.setattr(mod, "_chat", lambda p: "not json at all")
    assert mod.extract_concepts(title="t", chunks=_chunks(), language="en") == []


def test_empty_chunks_short_circuits(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    monkeypatch.setattr(
        mod, "_chat", lambda p: (_ for _ in ()).throw(AssertionError("no LLM call"))
    )
    assert mod.extract_concepts(title="t", chunks=[], language="zh") == []
