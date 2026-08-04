"""检索后反思——反思式多轮检索的循环控制器。

单轮检索有天花板: 问题需要的证据未必能被一次查询召回(问 "A 与 B 的区别",
第一轮可能只召回 A)。本模块在每轮检索后让 LLM 回答三件事: **证据够了吗?
缺什么? 下一轮该搜什么?** qa_agentic/qa_stream 的循环据此决定继续或收敛:
sufficiency=="sufficient" 停轮; 否则拿 follow_up 作下一轮检索查询(reflect
自身每轮都评估**原始问题**, 只有检索查询在换); follow_up 为空也停(没有新
方向, 空转无意义)。最后一轮不调用(判定无法再触发下一轮, 调了白花钱)。

与 abstain 的分工: reflect 是循环中段的方向盘("还要不要再搜、往哪搜"),
abstain 是 LLM 作答前的最终闸门("到底许不许答")。失败姿态同向——reflect
任何异常返回 sufficient(宁停不空转, LLM 挂掉时不烧轮次), 都不阻塞主链路。

相对基准的确认偏离:
a) zh/en 双 prompt 模板路由(复用 query_rewrite._query_language; 未知语言走
   英文)。follow_up 不强制双语——它会喂回 retrieve_round, pipeline 内部再过
   query_rewrite, 中文查询在那里已做中英关键词混出, 这里不重复操心。
b) 修基准缺陷: 基准把 `float(data["score"])` 放在 try/except **之外**, LLM 回
   合法 JSON 但 score 非数值(如 "high")时 ValueError 炸穿整个 QA 请求——
   一个只该影响"要不要多搜一轮"的辅助判定打死了主链路。重建版输出净化全部
   走安全路径: score 强转失败落 0.5 并裁剪 [0,1]; sufficiency 大小写归一 +
   三值域校验, 域外值落 "sufficient"(与缺键缺省一致); missing/follow_up 非
   字符串(含显式 null)强转空串。

照抄基准: 签名与返回键、证据 6000 字符截断(中文 CJK 约 1 字 1 token, 反思
调用略贵但内容量更大, 非正确性问题已记账)、temperature=0/max_tokens=300
(判定类调用要确定性, 与 intent 课同口径不配置化)、chat() 走默认 chat_model。
"""

from __future__ import annotations

import json
import re

from ..utils.logger import get_logger
from .llm import chat
from .query_rewrite import _query_language

log = get_logger("rag.reflect")

_PROMPT = """You evaluate whether the retrieved evidence is sufficient to answer
a research question. Reply with a JSON object:

  "sufficiency": one of "sufficient" | "partial" | "insufficient"
  "missing":     short description of what is missing (empty string if sufficient)
  "follow_up":   a single follow-up search query (empty string if sufficient)
  "score":       float in [0,1]

Question: {q}

Evidence (truncated):
{evidence}

Return only JSON.
"""

_PROMPT_ZH = """你负责评估检索到的证据是否足以回答一个研究问题。只回复一个 JSON 对象:

  "sufficiency": "sufficient" | "partial" | "insufficient" 三选一
  "missing":     缺少什么证据的简短说明(充分时为空字符串)
  "follow_up":   单条下一轮检索查询, 与问题同语言(充分时为空字符串)
  "score":       [0,1] 之间的浮点数

问题: {q}

证据(已截断):
{evidence}

只返回 JSON, 不要输出其他内容。
"""

_VALID_SUFFICIENCY = frozenset({"sufficient", "partial", "insufficient"})


def _norm_sufficiency(value) -> str:
    """大小写归一 + 三值域校验; 域外/非字符串落 "sufficient"(宁停不空转)。"""
    if isinstance(value, str) and value.strip().lower() in _VALID_SUFFICIENCY:
        return value.strip().lower()
    return "sufficient"


def _as_str(value) -> str:
    """missing/follow_up 净化: 非字符串(含显式 null)落空串, 循环里
    `if r["follow_up"]` 语义不变, trace 不出现 None。"""
    return value.strip() if isinstance(value, str) else ""


def _safe_score(value) -> float:
    """安全强转 + [0,1] 裁剪; 失败落 0.5(基准在 try 外 float() 会炸穿)。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def reflect(question: str, evidence: str) -> dict:
    truncated = evidence[:6000]
    template = _PROMPT_ZH if _query_language(question) == "zh" else _PROMPT
    try:
        raw = chat(
            [
                {
                    "role": "user",
                    "content": template.replace("{q}", question).replace("{evidence}", truncated),
                }
            ],
            temperature=0,
            max_tokens=300,
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        log.warning(f"reflect failed: {e}; assume sufficient to avoid loops")
        return {"sufficiency": "sufficient", "missing": "", "follow_up": "", "score": 0.5}

    return {
        "sufficiency": _norm_sufficiency(data.get("sufficiency")),
        "missing": _as_str(data.get("missing")),
        "follow_up": _as_str(data.get("follow_up")),
        "score": _safe_score(data.get("score", 0.5)),
    }


__all__ = [
    "reflect",
]
