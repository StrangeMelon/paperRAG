"""wiki/flow.py 词条创建/修补契约(LLM 打桩)。

钉死的关键行为:
- create: self_eval 低于门槛 -> 丢弃; 词条带 primary/translation 标签、
  定义语言跟随创建论文语言、定义重写锁生效;
- patch: LLM 只能返回白名单操作(add_label / add_key_paper / add_evidence /
  add_variant / propose_definition / add_open_problem), 未知操作忽略;
- 24h 锁只限制 propose_definition —— 锁内其余操作照常应用(基准锁整条词条
  会让批量入库丢关联, 这是修正点);
- 一切只追加: 旧标签/论文/开放问题不被删除; evidence 受输入白名单约束。
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _mod():
    return importlib.import_module("paper_rag.wiki.flow")


def _fake_config():
    return SimpleNamespace(
        wiki=SimpleNamespace(
            self_eval_threshold=0.7,
            definition_rewrite_lock_hours=24,
        ),
        llm=SimpleNamespace(temperatures=SimpleNamespace(wiki=0.2)),
    )


def _chunks():
    return [
        {"chunk_id": "c1", "section": "摘要", "text": "本文研究强化学习。"},
        {"chunk_id": "c2", "section": "方法", "text": "策略梯度方法。"},
    ]


def _existing() -> WikiEntry:
    return WikiEntry(
        entry_id="concept:reinforcementlearning",
        name="Reinforcement Learning",
        category="method",
        definition="A learning paradigm driven by reward signals.",
        definition_language="en",
        labels=[WikiLabel(text="Reinforcement Learning", language="en", kind="primary")],
        key_papers=["arxiv:1811.12560"],
        open_problems=["sample efficiency"],
    )


def test_create_entry_builds_bilingual_labels_and_lock(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    raw = """{
      "definition": "通过奖励信号驱动策略优化的学习范式 [chunk:c1]",
      "aliases_zh": ["强化学习"], "aliases_en": ["RL"],
      "open_problems": ["样本效率"], "self_eval": 0.85
    }"""
    monkeypatch.setattr(mod, "_chat", lambda prompt: raw)

    entry = mod.create_entry(
        name="Reinforcement Learning",
        category="method",
        language="zh",
        paper_id="arxiv:zh1",
        paper_title="某中文论文",
        chunks=_chunks(),
        labels_zh=["强化学习"],
        labels_en=["RL"],
    )
    assert entry is not None
    assert entry.entry_id == "concept:reinforcementlearning"
    assert entry.definition_language == "zh"  # 定义语言跟随创建论文
    texts = {lb.text for lb in entry.labels}
    assert {"Reinforcement Learning", "强化学习", "RL"} <= texts
    primary = [lb for lb in entry.labels if lb.kind == "primary"]
    assert primary and primary[0].text == "Reinforcement Learning"
    assert entry.key_papers == ["arxiv:zh1"]
    assert set(entry.evidence_chunks) == {"c1", "c2"}
    assert entry.definition_lock_until is not None  # 定义重写锁已生效


def test_create_entry_gated_by_self_eval(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    monkeypatch.setattr(mod, "_chat", lambda p: '{"definition": "x", "self_eval": 0.3}')
    assert (
        mod.create_entry(
            name="X",
            category="concept",
            language="en",
            paper_id="p",
            paper_title="t",
            chunks=_chunks(),
        )
        is None
    )
    monkeypatch.setattr(mod, "_chat", lambda p: "garbage not json")
    assert (
        mod.create_entry(
            name="X",
            category="concept",
            language="en",
            paper_id="p",
            paper_title="t",
            chunks=_chunks(),
        )
        is None
    )


def test_patch_applies_whitelisted_ops_only(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    raw = """{
      "ops": [
        {"op": "add_label", "text": "强化学习", "language": "zh", "kind": "translation"},
        {"op": "add_key_paper", "paper_id": "arxiv:2005.01643"},
        {"op": "add_evidence", "chunk_id": "c2"},
        {"op": "add_evidence", "chunk_id": "c-outside-whitelist"},
        {"op": "add_variant", "name": "Deep RL", "summary": "结合深度网络"},
        {"op": "add_open_problem", "text": "奖励设计"},
        {"op": "propose_definition", "definition": "新定义: 奖励驱动的策略优化范式"},
        {"op": "delete_everything", "target": "*"}
      ],
      "self_eval": 0.9
    }"""
    monkeypatch.setattr(mod, "_chat", lambda p: raw)

    merged = mod.patch_entry(
        existing=_existing(),
        paper_id="arxiv:zh1",
        paper_title="某中文论文",
        language="zh",
        chunks=_chunks(),
    )
    assert merged is not None
    assert {lb.text for lb in merged.labels} == {"Reinforcement Learning", "强化学习"}
    assert set(merged.key_papers) == {"arxiv:1811.12560", "arxiv:2005.01643", "arxiv:zh1"}
    assert "c2" in merged.evidence_chunks
    assert "c-outside-whitelist" not in merged.evidence_chunks  # 白名单外剔除
    assert merged.variants[0].name == "Deep RL"
    assert set(merged.open_problems) == {"sample efficiency", "奖励设计"}
    # 未锁定: 定义被更新且锁刷新
    assert merged.definition.startswith("新定义")
    assert merged.definition_lock_until is not None


def test_patch_lock_only_blocks_definition_rewrite(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    raw = """{
      "ops": [
        {"op": "add_key_paper", "paper_id": "arxiv:2005.01643"},
        {"op": "propose_definition", "definition": "试图重写定义"}
      ],
      "self_eval": 0.9
    }"""
    monkeypatch.setattr(mod, "_chat", lambda p: raw)

    locked = _existing()
    locked.definition_lock_until = datetime.now(UTC) + timedelta(hours=12)
    merged = mod.patch_entry(
        existing=locked,
        paper_id="arxiv:zh1",
        paper_title="t",
        language="zh",
        chunks=_chunks(),
    )
    assert merged is not None
    # 锁内: 定义不变, 但论文关联照常追加(基准锁整条词条的修正点)
    assert merged.definition == "A learning paradigm driven by reward signals."
    assert "arxiv:2005.01643" in merged.key_papers


def test_patch_gated_by_self_eval(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    monkeypatch.setattr(mod, "_chat", lambda p: '{"ops": [], "self_eval": 0.2}')
    assert (
        mod.patch_entry(
            existing=_existing(), paper_id="p", paper_title="t", language="en", chunks=_chunks()
        )
        is None
    )
