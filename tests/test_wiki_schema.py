"""wiki/schema.py 纯逻辑契约。

钉死三件事:
1. normalize_label 的 NFKC + casefold 规范化对全角/大小写/标点等价, 且 CJK 原样保留
   (中文概念名是一等公民, 不是英文的降级路径)。
2. entry_id 只在创建时由当时的规范名生成一次, 之后是不可反推的稳定句柄——
   所有查找一律走 labels 表, 不存在"从名字重算 ID"的代码路径。
3. 短标签(RL/CL 这类缩写)必须被识别出来, 供解析层禁止其单独触发自动合并;
   中文按 CJK 字数单独分档, 4 字中文词(如"强化学习")不是短标签。
"""

from __future__ import annotations

from paper_rag.wiki.schema import (
    WikiEntry,
    WikiLabel,
    is_short_label,
    label_language,
    make_entry_id,
    normalize_label,
)


def test_normalize_label_nfkc_casefold_and_punctuation():
    # 全角字母 NFKC 折半角, casefold 折小写, 空白/连字符/标点全部剔除
    # (全角字符为被测输入, 故意保留 -> noqa RUF001)
    assert normalize_label("Ｒｅｉｎｆｏｒｃｅｍｅｎｔ－Learning") == "reinforcementlearning"  # noqa: RUF001
    assert normalize_label("Contrastive  Learning") == "contrastivelearning"
    assert normalize_label("self-supervised learning") == "selfsupervisedlearning"
    assert normalize_label("  RL  ") == "rl"


def test_normalize_label_preserves_cjk():
    assert normalize_label("强化学习") == "强化学习"
    assert normalize_label("强化 学习") == "强化学习"  # 中文词内空格视为噪声
    # 全角括号剔除, 内容保留(被测输入故意用全角 -> noqa RUF001)
    assert normalize_label("对比学习（CL）") == "对比学习cl"  # noqa: RUF001
    assert normalize_label("Ｑ学习") == "q学习"  # 全角字母与 CJK 混排


def test_normalize_label_empty_and_symbol_only():
    assert normalize_label("") == ""
    assert normalize_label("  --- ") == ""


def test_make_entry_id_from_creation_name():
    assert make_entry_id("Reinforcement Learning") == "concept:reinforcementlearning"
    assert make_entry_id("强化学习") == "concept:强化学习"
    # 规范化等价的名字生成相同 ID —— 创建前必须先经解析层查重, 这里只保证确定性
    assert make_entry_id("reinforcement-learning") == make_entry_id("Reinforcement Learning")


def test_is_short_label_ascii_acronyms():
    assert is_short_label("RL")
    assert is_short_label("CL")
    assert is_short_label("GAN")
    assert is_short_label("BERT")  # 恰在 4 字符阈值上: 只是不许单独自动合并, 仍可经 LLM 验证
    assert not is_short_label("REINFORCE")
    assert not is_short_label("Reinforcement")


def test_is_short_label_cjk_uses_char_count():
    # 中文信息密度高: 2 字词已是完整概念("蒸馏"), 只有单字才算短
    assert not is_short_label("强化学习")
    assert not is_short_label("蒸馏")
    assert is_short_label("图")
    # 混排含 >=2 个 CJK 字符即不算短
    assert not is_short_label("Q学习")


def test_label_language_heuristic():
    assert label_language("强化学习") == "zh"
    assert label_language("Q学习") == "zh"  # 含 CJK 即判中文
    assert label_language("Reinforcement Learning") == "en"
    assert label_language("RL") == "en"
    assert label_language("123") is None
    assert label_language("") is None


def test_wiki_label_defaults():
    lb = WikiLabel(text="强化学习", language="zh", kind="primary")
    assert lb.confidence == 1.0
    assert lb.verified is False
    assert lb.source_paper_id is None


def test_wiki_entry_defaults_and_roundtrip():
    entry = WikiEntry(
        entry_id=make_entry_id("Reinforcement Learning"),
        name="Reinforcement Learning",
        category="method",
        definition="A learning paradigm driven by reward signals.",
        definition_language="en",
        labels=[
            WikiLabel(text="Reinforcement Learning", language="en", kind="primary"),
            WikiLabel(text="强化学习", language="zh", kind="translation"),
        ],
    )
    assert entry.version == 1
    assert entry.merged_into is None
    assert entry.definition_lock_until is None
    assert entry.key_papers == []
    assert entry.evidence_chunks == []

    # dump -> validate 往返无损, 供 store 层 JSON 序列化与版本快照使用
    restored = WikiEntry.model_validate(entry.model_dump(mode="json"))
    assert restored == entry


def test_wiki_entry_rejects_unknown_category():
    import pytest

    with pytest.raises(ValueError):
        WikiEntry(entry_id="concept:x", name="x", category="paradigm")
