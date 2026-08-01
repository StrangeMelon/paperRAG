"""文本切块器: 把章节正文贪心打包为 ~target_tokens 的文本块。

与基准的差异(切块层已确认方案, 2026-08-01):
- 偏移用 body.find 真实定位, 不变量 body[char_start:char_end] == text
  (基准的 cursor 算术在 4 个以上连续换行时漂移, 末块 char_end 多算 2)。
- 无 tiktoken 时的回退估算按码位区分: CJK(含全角标点)逐字计 1, 其余 len//4
  (基准的 len//4 对中文低估约 4-7 倍)。
- 超过 target 的段落先按句子切分再打包: zh 用 。！？；…． (后随引号归前句,
  全角点是中文参考文献条目结束符), en 用 [.!?] + 空白(小数点不切), None 取
  并集; 完全无句读的病态段按 token 等分硬切, 保证任何 chunk 有上界
  (基准从不切分超长段落)。
- overlap 携带尾段加防重守卫: 尾段 token*2 > target 时放弃携带
  (基准会把接近 target 的尾段重复输出成独立 chunk); overlap_tokens 仍只作
  布尔开关, 与基准一致。
"""  # noqa: RUF002

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .. import config as cfg


@dataclass
class TextChunk:
    text: str
    char_start: int
    char_end: int


@dataclass
class _Unit:
    """一个打包单元(段落、句子或硬切片段), 记录其在 body 中的真实跨度。"""

    start: int
    end: int
    tokens: int


# CJK 码位区间: 部首/康熙/CJK 标点/假名/注音/统一表意, 兼容表意, 全角形式
_CJK_RANGES = ((0x2E80, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))

# 中文句子边界: 句读(可连续, 如省略号/叹问连用)加零个或多个后随的收尾引号/括号;
# 全角点 ．(U+FF0E)是中文参考文献常用条目结束符, 切多了无害(贪心重打包兜底)  # noqa: RUF003
_ZH_BOUNDARY_RE = re.compile(r"[。！？；…．]+[”’』」）》〉】]*")  # noqa: RUF001
# 英文句子边界: 终结标点加收尾引号/括号, 且后随空白或段尾(小数点 3.5 不切)
_EN_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")
_UNION_BOUNDARY_RE = re.compile(f"(?:{_ZH_BOUNDARY_RE.pattern})|(?:{_EN_BOUNDARY_RE.pattern})")

_ENC = None
_USE_TIKTOKEN = None


def _encoder():
    global _ENC, _USE_TIKTOKEN
    if _USE_TIKTOKEN is False:
        return None
    if _ENC is None:
        try:
            import tiktoken

            _ENC = tiktoken.get_encoding(cfg.load().chunk.text.encoding)
            _USE_TIKTOKEN = True
        except ImportError:
            _USE_TIKTOKEN = False
            return None
    return _ENC


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _count_tokens(s: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(s))
    n_cjk = sum(1 for ch in s if _is_cjk(ch))
    return max(1, n_cjk + (len(s) - n_cjk) // 4)


def _boundary_pattern(language: str | None) -> re.Pattern[str]:
    if language == "zh":
        return _ZH_BOUNDARY_RE
    if language == "en":
        return _EN_BOUNDARY_RE
    return _UNION_BOUNDARY_RE


def _strip_span(body: str, start: int, end: int) -> tuple[int, int]:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def _hard_cut(body: str, start: int, end: int, tokens: int, target: int) -> list[_Unit]:
    """无句读可用时按 token 等分硬切, 每片 token 近似 <= target。"""
    pieces = math.ceil(tokens / target)
    step = math.ceil((end - start) / pieces)
    units: list[_Unit] = []
    for i in range(pieces):
        s, e = _strip_span(body, start + i * step, min(end, start + (i + 1) * step))
        if s < e:
            units.append(_Unit(s, e, _count_tokens(body[s:e])))
    return units


def _split_sentences(
    body: str, start: int, end: int, pattern: re.Pattern[str], target: int
) -> list[_Unit]:
    """把超长段落切成句子单元; 仍超长的单句(无句读)硬切兜底。"""
    spans: list[tuple[int, int]] = []
    pos = start
    for m in pattern.finditer(body, start, end):
        spans.append((pos, m.end()))
        pos = m.end()
    if pos < end:
        spans.append((pos, end))

    units: list[_Unit] = []
    for raw_s, raw_e in spans:
        s, e = _strip_span(body, raw_s, raw_e)
        if s >= e:
            continue
        tokens = _count_tokens(body[s:e])
        if tokens > target:
            units.extend(_hard_cut(body, s, e, tokens, target))
        else:
            units.append(_Unit(s, e, tokens))
    return units


def _build_units(body: str, language: str | None, target: int) -> list[_Unit]:
    pattern = _boundary_pattern(language)
    units: list[_Unit] = []
    cursor = 0
    for para in body.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        start = body.find(p, cursor)
        end = start + len(p)
        cursor = end
        tokens = _count_tokens(p)
        if tokens > target:
            units.extend(_split_sentences(body, start, end, pattern, target))
        else:
            units.append(_Unit(start, end, tokens))
    return units


def _make_chunk(body: str, buf: list[_Unit]) -> TextChunk:
    start, end = buf[0].start, buf[-1].end
    return TextChunk(text=body[start:end], char_start=start, char_end=end)


def chunk_text(body: str, *, language: str | None = None) -> list[TextChunk]:
    c = cfg.load().chunk.text
    if not body.strip():
        return []

    units = _build_units(body, language, c.target_tokens)
    if not units:
        return []

    chunks: list[TextChunk] = []
    buf: list[_Unit] = []
    buf_tokens = 0
    for unit in units:
        if buf and buf_tokens + unit.tokens > c.target_tokens:
            chunks.append(_make_chunk(body, buf))
            tail = buf[-1]
            if c.overlap_tokens > 0 and tail.tokens * 2 <= c.target_tokens:
                buf = [tail]
                buf_tokens = tail.tokens
            else:
                buf = []
                buf_tokens = 0
        buf.append(unit)
        buf_tokens += unit.tokens
    if buf:
        chunks.append(_make_chunk(body, buf))
    return chunks
