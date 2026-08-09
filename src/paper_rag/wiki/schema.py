"""Wiki 词条与标签的纯逻辑模型(不依赖 sqlmodel, 供纯测试与所有上层复用)。

概念身份方案(ADR-0003):
- entry_id = "concept:<创建时规范名>", 创建后永不重算——它是稳定句柄, 不是查询
  入口; 所有名字查找一律走 wiki_labels 表, 不存在从名字反推 ID 的代码路径。
- 名字是可演化标签(WikiLabel): 携带语言(zh/en/None)、类型(primary/translation/
  acronym/variant)、来源论文与置信度。跨语言合并靠标签 + 语义解析, 不靠 ID。
- 规范化 = NFKC + casefold + 仅保留字母数字(CJK 属字母类, 原样保留), 使
  全角/大小写/连字符/空格差异折叠为同一词面; 与检索层 CJK bigram 策略互不干涉。
- 短标签(RL/CL 这类缩写)词面歧义大, 解析层禁止其单独触发自动合并; 中文按
  CJK 字数分档——两字词("蒸馏")已是完整概念, 只有单字才算短。
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

WikiCategory = Literal["concept", "method", "task", "dataset", "metric"]
LabelKind = Literal["primary", "translation", "acronym", "variant"]
LabelLanguage = Literal["zh", "en"]

_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # Extension A
    (0xF900, 0xFAFF),  # Compatibility Ideographs
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def normalize_label(text: str) -> str:
    """NFKC 折叠全角, casefold 折叠大小写, 只保留字母数字(含 CJK)。"""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(c for c in folded if c.isalnum())


def make_entry_id(name: str) -> str:
    """由创建时的名字生成 entry_id。只在创建时调用一次, 之后 ID 永不重算。"""
    return f"concept:{normalize_label(name)}"


def label_language(text: str) -> LabelLanguage | None:
    """标签语言启发式: 含 CJK 判 zh, 含拉丁字母判 en, 其余(纯数字/空)为 None。"""
    for ch in text or "":
        if _is_cjk(ch):
            return "zh"
    for ch in text or "":
        if ch.isascii() and ch.isalpha():
            return "en"
    return None


def is_short_label(text: str, *, max_ascii_chars: int = 4, max_cjk_chars: int = 1) -> bool:
    """短标签判定: 禁止单独触发自动合并(仍可经 LLM 验证后合并)。

    含 CJK 的标签按 CJK 字数分档(默认单字才算短); 纯 ASCII 按规范化长度分档
    (默认 <=4, 覆盖 RL/CL/GAN/BERT 这类缩写)。
    """
    norm = normalize_label(text)
    if not norm:
        return True
    cjk_count = sum(1 for c in norm if _is_cjk(c))
    if cjk_count:
        return cjk_count <= max_cjk_chars
    return len(norm) <= max_ascii_chars


class WikiLabel(BaseModel):
    """概念的一个名字(主名/翻译/缩写/变体), 可演化、可追溯。"""

    text: str
    language: LabelLanguage | None = None
    kind: LabelKind = "variant"
    source_paper_id: str | None = None
    confidence: float = 1.0
    verified: bool = False


class Variant(BaseModel):
    """概念的具名变种(如 Double DQN 之于 DQN), 只是词条内的轻量列表项。"""

    name: str
    summary: str
    paper_id: str | None = None


class WikiEntry(BaseModel):
    """词条当前快照。关系数据(标签/论文/证据)在 store 层另有关系表为真相源,
    快照内的冗余列表只为消费端(QA 背景)一次读取方便。"""

    entry_id: str
    name: str
    category: WikiCategory = "concept"
    definition: str = ""
    definition_language: LabelLanguage | None = None
    labels: list[WikiLabel] = Field(default_factory=list)
    key_papers: list[str] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    evidence_chunks: list[str] = Field(default_factory=list)
    version: int = 1
    updated_at: datetime | None = None
    definition_lock_until: datetime | None = None  # 只锁定义重写, 关系新增不受限
    merged_into: str | None = None  # 非空表示已并入目标词条, 本条只作重定向 tombstone
