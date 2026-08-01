"""文本切块器的行为契约测试。

切片 0: 空输入与单段基线, 以及"偏移即切片"不变量 body[char_start:char_end] == text。
切片 1: 段落贪心打包、overlap 尾段携带与防重守卫、find 真实定位偏移(修正基准算术漂移)。
切片 2: token 计数——tiktoken 主路径与无 tiktoken 时的双语回退(CJK 逐字计 1, 其余 len//4)。
切片 3: 英文超长段落句子切分(小数点不切、贪心重打包)。
切片 4: 中文超长段落句子切分(。！？；…．与后随引号)、zh/en/None 语言路由、无标点硬切兜底。

接口约定(切块层已确认方案, 2026-08-01):

    chunk_text(body: str, *, language: str | None = None) -> list[TextChunk]

`body` 是 section_splitter 产出的单个章节正文; 配置仍走 cfg.load().chunk.text
(target_tokens / overlap_tokens / encoding), 不新增配置项。
"""  # noqa: RUF002

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

import paper_rag.config as config


def _chunker_module() -> ModuleType:
    return importlib.import_module("paper_rag.chunk.text_chunker")


def _patch_config(monkeypatch, *, target_tokens: int, overlap_tokens: int = 5) -> None:
    """把 chunk.text 配置压小, 让测试文本不必凑 500 token。"""
    conf = config.load()
    conf.chunk.text.target_tokens = target_tokens
    conf.chunk.text.overlap_tokens = overlap_tokens
    monkeypatch.setattr(config, "load", lambda path=None: conf)


def _force_fallback_tokens(monkeypatch, mod: ModuleType) -> None:
    """禁用 tiktoken, 走确定性的回退估算(ASCII len//4, CJK 逐字计 1)。"""
    monkeypatch.setattr(mod, "_ENC", None)
    monkeypatch.setattr(mod, "_USE_TIKTOKEN", False)


def _assert_slice_invariant(body: str, chunks) -> None:
    for c in chunks:
        assert body[c.char_start : c.char_end] == c.text, f"偏移不可回切原文: {c.text[:40]!r}"
        assert c.text == c.text.strip(), f"chunk 文本两端未去空白: {c.text[:40]!r}"


# ---------------------------------------------------------------------------
# 切片 0: 空输入与单段基线
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_body_returns_empty(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=100)
    assert mod.chunk_text("") == []
    assert mod.chunk_text("  \n\n \n ") == []


def test_single_short_paragraph_yields_one_exact_chunk(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=100)
    _force_fallback_tokens(monkeypatch, mod)

    body = "\nHello chunker world.\n"
    chunks = mod.chunk_text(body)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello chunker world."
    _assert_slice_invariant(body, chunks)


def test_small_paragraphs_under_target_stay_in_one_chunk(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=100)
    _force_fallback_tokens(monkeypatch, mod)

    body = "First tiny paragraph.\n\nSecond tiny paragraph."
    chunks = mod.chunk_text(body)

    assert len(chunks) == 1
    assert chunks[0].text == body
    _assert_slice_invariant(body, chunks)


# ---------------------------------------------------------------------------
# 切片 1: 贪心打包、overlap 与偏移
# ---------------------------------------------------------------------------


def test_greedy_packing_with_tail_paragraph_overlap(monkeypatch) -> None:
    """尾段 token*2 <= target 时按基准语义携带整段作为下一 chunk 开头。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=20, overlap_tokens=5)
    _force_fallback_tokens(monkeypatch, mod)

    pa, pb, pc = "a" * 32, "b" * 32, "c" * 32  # 各 8 token
    body = f"{pa}\n\n{pb}\n\n{pc}"
    chunks = mod.chunk_text(body)

    assert [c.text for c in chunks] == [f"{pa}\n\n{pb}", f"{pb}\n\n{pc}"]
    assert chunks[1].char_start == body.index(pb)
    _assert_slice_invariant(body, chunks)


def test_overlap_guard_drops_carry_when_tail_too_big(monkeypatch) -> None:
    """防重守卫: 尾段 token*2 > target 时放弃携带, 不产生重复 chunk。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=5)
    _force_fallback_tokens(monkeypatch, mod)

    pa, pb, pc = "a" * 24, "b" * 24, "c" * 24  # 各 6 token, 6*2 > 10
    body = f"{pa}\n\n{pb}\n\n{pc}"
    chunks = mod.chunk_text(body)

    assert [c.text for c in chunks] == [pa, pb, pc]
    _assert_slice_invariant(body, chunks)


def test_overlap_disabled_when_config_zero(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=20, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    pa, pb, pc = "a" * 32, "b" * 32, "c" * 32  # 各 8 token, 尾段本可携带
    body = f"{pa}\n\n{pb}\n\n{pc}"
    chunks = mod.chunk_text(body)

    assert [c.text for c in chunks] == [f"{pa}\n\n{pb}", pc]
    _assert_slice_invariant(body, chunks)


def test_offsets_survive_extra_blank_lines(monkeypatch) -> None:
    """基准的 cursor += len+2 算术在 4 个以上连续换行时漂移; 重建版用真实定位。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    pa, pb, pc = "a" * 24, "b" * 24, "c" * 24
    body = f"{pa}\n\n\n\n{pb}\n\n\n\n{pc}"
    chunks = mod.chunk_text(body)

    assert [c.text for c in chunks] == [pa, pb, pc]
    assert chunks[1].char_start == body.index(pb)
    assert chunks[2].char_start == body.index(pc)
    _assert_slice_invariant(body, chunks)


# ---------------------------------------------------------------------------
# 切片 2: token 计数
# ---------------------------------------------------------------------------


def test_fallback_token_count_is_cjk_aware(monkeypatch) -> None:
    """回退估算: CJK 码位(含全角标点)逐字计 1, 其余按 len//4, 下限 1。"""
    mod = _chunker_module()
    _force_fallback_tokens(monkeypatch, mod)

    assert mod._count_tokens("深度学习模型") == 6
    assert mod._count_tokens("abcdefgh") == 2
    assert mod._count_tokens("深度abcd") == 3
    assert mod._count_tokens("ab") == 1
    assert mod._count_tokens("检索增强。") == 5
    assert mod._count_tokens("摘要：") == 3  # noqa: RUF001  全角冒号也计 1


def test_tiktoken_path_counts_real_bpe_tokens(monkeypatch) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    mod = _chunker_module()
    monkeypatch.setattr(mod, "_ENC", None)
    monkeypatch.setattr(mod, "_USE_TIKTOKEN", None)

    enc = tiktoken.get_encoding(config.load().chunk.text.encoding)
    for s in ("Retrieval-augmented generation works.", "检索增强生成提升了引用准确率。"):
        assert mod._count_tokens(s) == len(enc.encode(s))


# ---------------------------------------------------------------------------
# 切片 3: 英文超长段落句子切分
# ---------------------------------------------------------------------------


def test_english_oversized_paragraph_splits_at_sentences(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    s1 = "The model works well fine."
    s2 = "It scores 3.5 in the eval."
    s3 = "Great results overall done."
    body = f"{s1} {s2} {s3}"  # 单段 ~20 token, 超过 target
    chunks = mod.chunk_text(body)

    assert [c.text for c in chunks] == [s1, s2, s3]
    _assert_slice_invariant(body, chunks)


def test_english_decimal_number_is_not_a_boundary(monkeypatch) -> None:
    """`3.5` 的点后无空白, 不得视为句子边界。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    s1 = "It reaches 3.51 exact points now."
    s2 = "The second sentence follows here."
    body = f"{s1} {s2}"
    chunks = mod.chunk_text(body)

    assert any("3.51" in c.text for c in chunks)
    assert all("3.5" not in c.text or "3.51" in c.text for c in chunks)
    assert [c.text for c in chunks] == [s1, s2]


def test_no_punctuation_paragraph_hard_cut_keeps_bound(monkeypatch) -> None:
    """完全无标点的病态段(OCR 碎块)按 token 等分硬切, 保证任何 chunk 有上界。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    body = "x" * 80  # 20 token, 无任何句读
    chunks = mod.chunk_text(body)

    assert len(chunks) == 2
    assert "".join(c.text for c in chunks) == body
    for c in chunks:
        assert mod._count_tokens(c.text) <= 10
    _assert_slice_invariant(body, chunks)


# ---------------------------------------------------------------------------
# 切片 4: 中文超长段落句子切分与语言路由
# ---------------------------------------------------------------------------


def test_chinese_oversized_paragraph_splits_at_sentences(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    z1 = "检索增强生成很有效。"
    z2 = "混合检索提升了精度！"  # noqa: RUF001
    z3 = "引用准确率继续上升？"  # noqa: RUF001
    body = f"{z1}{z2}{z3}"  # 30 token 单段, 各句 10 token
    chunks = mod.chunk_text(body, language="zh")

    assert [c.text for c in chunks] == [z1, z2, z3]
    _assert_slice_invariant(body, chunks)


def test_chinese_semicolon_is_a_boundary(monkeypatch) -> None:
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    z1 = "方法一简单有效；"  # noqa: RUF001
    z2 = "方法二精度更好；"  # noqa: RUF001
    z3 = "方法三代价最小。"
    body = f"{z1}{z2}{z3}"
    chunks = mod.chunk_text(body, language="zh")

    assert [c.text for c in chunks] == [z1, z2, z3]


def test_chinese_closing_quote_stays_with_sentence(monkeypatch) -> None:
    """句末引号归前句: 边界吞掉 `。”` 之后再切。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=12, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    z1 = "作者说：“模型确实有效。”"  # noqa: RUF001
    z2 = "后续分析进一步验证结论。"
    body = f"{z1}{z2}"
    chunks = mod.chunk_text(body, language="zh")

    assert chunks[0].text == z1
    assert chunks[0].text.endswith("。”")
    assert chunks[1].text == z2


def test_fullwidth_stop_is_a_zh_boundary(monkeypatch) -> None:
    """全角点 ．(U+FF0E)是中文参考文献常用条目结束符, 判为 zh 句子边界。

    修复前该形态段落无任何 zh 句读可用, 落到等分硬切(真实块检查决策点 b)。
    """  # noqa: RUF002
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=14, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    r1 = "王某．综合能源服务研究．"  # noqa: RUF001
    r2 = "李某．区块链交互模型综述．"  # noqa: RUF001
    body = f"{r1}{r2}"
    chunks = mod.chunk_text(body, language="zh")

    assert [c.text for c in chunks] == [r1, r2]
    _assert_slice_invariant(body, chunks)


def test_language_routing_zh_en_none(monkeypatch) -> None:
    """en 路由不识别中文句读(落到硬切); None 取并集, 对纯中文文本等价于 zh。"""
    mod = _chunker_module()
    _patch_config(monkeypatch, target_tokens=10, overlap_tokens=0)
    _force_fallback_tokens(monkeypatch, mod)

    # 三句长度刻意不等(6/10/10 token), 避免 en 路由的等分硬切碰巧落在句边界上
    z1 = "检索很有效。"
    z2 = "混合检索提升了精度。"
    z3 = "引用准确率继续上升。"
    zh_body = f"{z1}{z2}{z3}"

    zh_chunks = [c.text for c in mod.chunk_text(zh_body, language="zh")]
    en_chunks = [c.text for c in mod.chunk_text(zh_body, language="en")]
    none_chunks = [c.text for c in mod.chunk_text(zh_body)]

    assert zh_chunks == [z1, z2, z3]
    assert en_chunks != zh_chunks  # en 规则看不见中文句读
    assert none_chunks == zh_chunks  # 并集包含中文规则

    s1 = "The model works well fine."
    s2 = "It scores well in the eval."
    s3 = "Great results overall done."
    en_body = f"{s1} {s2} {s3}"
    assert [c.text for c in mod.chunk_text(en_body)] == [
        c.text for c in mod.chunk_text(en_body, language="en")
    ]
