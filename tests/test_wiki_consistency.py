"""wiki/consistency.py 启发式一致性检查契约(无 LLM, 只标记不删除)。

关键的中文扩展: 定义长度门槛按语言分档 —— 英文 20 字符, 中文 12 字符
(中文信息密度高, "通过奖励信号优化策略的方法" 12 字已是完整定义)。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _mod():
    return importlib.import_module("paper_rag.wiki.consistency")


def _fake_config():
    return SimpleNamespace(
        wiki=SimpleNamespace(
            consistency=SimpleNamespace(min_definition_chars_en=20, min_definition_chars_zh=12),
        ),
    )


def _entry(**kw) -> WikiEntry:
    base = dict(
        entry_id="concept:reinforcementlearning",
        name="Reinforcement Learning",
        category="method",
        definition="A learning paradigm driven by reward signals.",
        definition_language="en",
        key_papers=["arxiv:1811.12560"],
        evidence_chunks=["c1"],
    )
    base.update(kw)
    return WikiEntry(**base)


def test_healthy_entry_has_no_issues(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)
    assert mod.check_entry(_entry()) == []


def test_definition_length_is_language_tiered(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    # 12 字中文定义: 达标
    zh_ok = _entry(definition="通过奖励信号优化策略的方法", definition_language="zh")
    assert "definition_too_short" not in mod.check_entry(zh_ok)
    # 同样 13 个字符的英文定义: 不达标(英文门槛 20)
    en_short = _entry(definition="reward method", definition_language="en")
    assert "definition_too_short" in mod.check_entry(en_short)
    # 过短中文
    zh_short = _entry(definition="奖励学习", definition_language="zh")
    assert "definition_too_short" in mod.check_entry(zh_short)
    # 语言未知: 按较宽松的中文档判, 避免误报
    unknown = _entry(definition="通过奖励信号优化策略的方法", definition_language=None)
    assert "definition_too_short" not in mod.check_entry(unknown)


def test_structural_issues(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    assert "no_key_papers" in mod.check_entry(_entry(key_papers=[]))
    assert "self_related" in mod.check_entry(_entry(related=["concept:reinforcementlearning"]))
    assert "high_version_no_evidence" in mod.check_entry(_entry(version=11, evidence_chunks=[]))
    # 规范化后为空的标签(纯符号)属垃圾标签
    bad_label = _entry(labels=[WikiLabel(text="--", kind="variant")])
    assert "empty_normalized_label" in mod.check_entry(bad_label)


def test_find_problematic_entries_reports_only_bad(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", _fake_config)

    good = _entry()
    bad = _entry(entry_id="concept:x", name="x", key_papers=[])
    out = mod.find_problematic_entries([good, bad])
    assert len(out) == 1
    assert out[0]["entry_id"] == "concept:x"
    assert "no_key_papers" in out[0]["issues"]
