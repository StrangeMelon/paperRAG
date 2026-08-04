"""引用校验——硬不变量"答案只以 [chunk:<id>] 引用已检索块"的执行层。

prompt 再严令, LLM 仍有三种背叛方式, 三个纯函数各管一段(qa_simple/
qa_agentic/qa_stream 三条路径按 validate -> detect -> 有可疑才 strip 的固定
顺序消费):

  1. validate_citations — 删编造 id: [chunk:xxx] 的 id 不在检索集合里就是
     形式合规的内容幻觉, 从答案中移除并只收合法 id(保序去重);
  2. detect_suspicious_citations — 检测学术惯性形态: 数字引用 [1] 与
     作者-年份 (Vaswani et al., 2017)——预训练里的肌肉记忆, 出现即意味着
     模型在"引用"检索块之外的来源;
  3. strip_suspicious_citation_forms — 剥掉可疑形态并收拾标点前残留空格。

相对基准的确认偏离(中文答案的三个盲区):
a) 数字引用增加全角形态 【1】(中文排版惯用, Qwen 中文答案高频);
b) 新增 CJK 作者-年份形态 (张三等, 2020)——CJK 姓名 1-4 字 + 可选"等" +
   年份, 括号与逗号半/全角都认; 归入既有 author_year 键, 消费方 schema 不变;
c) strip 的标点收拾扩入全角 ，。；：、;
d) 不抄基准死常量 _CITE_RE(严格 hex 版定义后全仓库无人使用, validate 用的
   是宽松口径)。

照抄并记账的既有边界(与基准同款, 不扩): 区间/合并型 [1-3]、[1,2] 不识别;
非全括号形态 "Vaswani et al. (2017)"(作者在括号外)不识别, 全角括号包拉丁
作者名同理; validate 删无效引用会留双空格(仅当存在可疑形态触发 strip 时才
被顺手收拾)。
"""

from __future__ import annotations

import re

_ANY_CHUNK_CITE_RE = re.compile(r"\[chunk:([^\]\s]+)\]")

# 半角数字引用 [1]、[12]。纯数字要求天然排除 [chunk:..] 与 markdown 勾选框
# [x]; 上限 3 位排除 [1234] 类标识符。(基准还带一个永不生效的 (?<!chunk:)
# lookbehind——纯数字要求已排除该形态, 不抄。)
_NUM_CITE_RE = re.compile(r"\[(\d{1,3})\]")
# 全角数字引用 【1】(确认偏离 a)
_NUM_CITE_FW_RE = re.compile(r"【(\d{1,3})】")

# (Author Year) 与 (Author et al., Year) 形态(基准原版, 半角括号)。
_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*[A-Z][A-Za-zÀ-ſ\.\-]+(?:\s+et\s+al\.?)?(?:\s*,)?\s*(?:18|19|20)\d{2}[a-z]?\s*\)"
)
# CJK 作者-年份 (张三等, 2020) / （李四 等，2019）(确认偏离 b): CJK 姓名
# 1-4 字 + 可选"等"(= et al.) + 年份; 括号/逗号半全角都认。姓名后必须紧跟
# "等"或逗号分隔或空白再接年份, 裸 (见图 2)/（详见附录） 无年份不会误报。
_AUTHOR_YEAR_CJK_RE = re.compile(
    r"[（(]\s*[一-鿿]{1,4}\s*(?:等)?\s*[,，]?\s*(?:18|19|20)\d{2}[a-z]?\s*[)）]"
)


def validate_citations(answer: str, retrieved: list[dict]) -> tuple[str, list[str]]:
    """删掉 id 不在检索集合里的 chunk 引用。

    Returns (cleaned_answer, valid_chunk_ids)。valid 保序去重。
    """
    allowed = {c.get("chunk_id") for c in retrieved if c.get("chunk_id")}
    found = _ANY_CHUNK_CITE_RE.findall(answer)
    valid = []
    seen = set()
    for cid in found:
        if cid in allowed and cid not in seen:
            valid.append(cid)
            seen.add(cid)

    def _sub(m):
        return m.group(0) if m.group(1) in allowed else ""

    cleaned = _ANY_CHUNK_CITE_RE.sub(_sub, answer)
    return cleaned, valid


def detect_suspicious_citations(answer: str) -> dict:
    """报告答案中所有非 [chunk:] 引用形态。

    任何命中都意味着模型在引用检索块之外的"来源"——潜在的幻觉引用。

    Returns:
        {
            "numeric": ["[1]", "【3】", ...],
            "author_year": ["(Vaswani et al., 2017)", "（张三等，2020）", ...],
            "count": int,
        }
    """
    numeric = [f"[{n}]" for n in _NUM_CITE_RE.findall(answer)]
    numeric += [f"【{n}】" for n in _NUM_CITE_FW_RE.findall(answer)]
    author_year = _AUTHOR_YEAR_RE.findall(answer)
    author_year += _AUTHOR_YEAR_CJK_RE.findall(answer)
    return {
        "numeric": numeric,
        "author_year": author_year,
        "count": len(numeric) + len(author_year),
    }


def strip_suspicious_citation_forms(answer: str) -> str:
    """剥掉未锚定在检索块上的引用形态, 并收拾标点前的残留空格。"""
    cleaned = _NUM_CITE_RE.sub("", answer)
    cleaned = _NUM_CITE_FW_RE.sub("", cleaned)
    cleaned = _AUTHOR_YEAR_RE.sub("", cleaned)
    cleaned = _AUTHOR_YEAR_CJK_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+([,.;:，。；：、])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


__all__ = [
    "detect_suspicious_citations",
    "strip_suspicious_citation_forms",
    "validate_citations",
]
