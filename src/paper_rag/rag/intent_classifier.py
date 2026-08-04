"""意图分类——一次 LLM 调用决定检索力度。

把用户问题归为三档, 每档带一套检索参数:
  - factual:   查单个事实(定义、数字、单篇细节), 少取、单轮
  - reasoning: 对比或多方面分析, 中取、两轮(默认档)
  - explore:   综述/全景, 多取、三轮

`classify()` **永不抛异常**: LLM 挂了、JSON 坏了、intent 名不认识, 一律降级到
本地启发式, 再不行落 reasoning 中间档——猜错也不会太离谱。出口四键恒定齐全,
下游 `qa_agentic` 直接下标取值, 不做防御。

与基准的偏离(逐条有测试钉死):
  1. 三档参数从模块级 `_DEFAULTS` 常量搬到 `rag.intent` 配置段(项目约定"永不
     硬编码可调项"), 缺省值与基准逐项一致; 新增 `enabled` 开关, false 时省掉
     一次 LLM 往返。
  2. prompt 语言路由: 中文问题走中文模板。基准模板的三档说明与例子全是英文
     ("how do X and Y differ"), 对中文提问引导力弱; 中文的"区别/综述/是多少"
     等信号词写进中文模板判得更准。语言判定复用 `query_rewrite._query_language`,
     不重复维护两份 CJK 正则。
  3. 本地信号词启发式兜底: 基准 LLM 不可用时一律落 reasoning; 这里先按中英信号
     词判一次(零成本、无网络), 判不出来才落中间档。仅在模型缺席或输出脏时生效,
     模型给出有效答案时绝不覆盖。
"""

from __future__ import annotations

import json
import re

from .. import config as cfg
from ..utils.logger import get_logger
from .llm import chat
from .query_rewrite import _query_language

log = get_logger("rag.intent")

_VALID_INTENTS = ("factual", "reasoning", "explore")
_DEFAULT_INTENT = "reasoning"

_PROMPT = """You classify research questions into one of three intents:

- "factual": one specific fact (definition, number, single-paper detail).
- "reasoning": comparison, analysis, multi-aspect (e.g. "how do X and Y differ").
- "explore": broad landscape/survey (e.g. "what are recent advances in ...").

Return ONLY a JSON object: {"intent": "...", "reason": "..."}.

Question: {q}
"""

# 中文模板: 三档说明改用中文提问里真实出现的信号词, 比英文例子引导力强。
_PROMPT_ZH = """请把下面的研究性问题归入三种意图之一:

- "factual": 查一个具体事实(定义、数值、某一篇论文的细节), 例如"X 是什么"、
  "X 的准确率是多少"。
- "reasoning": 对比、分析或多方面综合, 例如"X 和 Y 有什么区别"、"X 相比 Y 的
  优劣"。
- "explore": 大范围综述或全景, 例如"X 领域近年有哪些进展"、"综述一下 X 的现状"。

只返回一个 JSON 对象: {"intent": "...", "reason": "..."}, 不要输出其他内容。

问题: {q}
"""

# 本地启发式信号词。按 explore -> reasoning -> factual 顺序匹配: 前者的措辞更
# 特征化("有哪些进展"), 后者更宽泛("是什么"), 宽的放后面免得抢先命中。
_EXPLORE_CUES_ZH = ("有哪些进展", "有哪些新", "研究进展", "综述", "现状", "全景", "概览", "综观")
_REASONING_CUES_ZH = ("区别", "差异", "对比", "相比", "异同", "优劣", "哪个更", "为什么")
_FACTUAL_CUES_ZH = ("是什么", "是多少", "定义", "叫什么", "多少个", "准确率是", "什么时候")

_EXPLORE_CUES_EN = (
    "recent advances",
    "recent progress",
    "state of the art",
    "survey",
    "overview",
    "landscape",
    "what are the trends",
)
_REASONING_CUES_EN = (
    "differ",
    "difference",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "trade-off",
    "tradeoff",
    "why does",
    "why is",
)
_FACTUAL_CUES_EN = ("what is", "what was", "how many", "how much", "definition of", "when was")


def classify(question: str) -> dict:
    """判定问题意图并返回该档的检索参数。永不抛异常。"""
    conf = cfg.load()
    intent_cfg = conf.rag.intent
    llm_ready = bool(intent_cfg.enabled and conf.llm.chat_model and conf.llm.api_key)
    if llm_ready:
        llm_ready = bool(conf.llm.base_url)

    intent: str | None = None
    if llm_ready:
        intent = _classify_via_llm(question)
    elif not intent_cfg.enabled:
        log.debug("intent classification disabled; using local cue heuristic")
    else:
        log.debug("intent LLM not configured; using local cue heuristic")

    if intent is None:
        intent = _heuristic_intent(question) or _DEFAULT_INTENT

    out = {"intent": intent, **_tier_params(intent_cfg, intent)}
    log.info(f"intent={intent} cfg={out}")
    return out


def _classify_via_llm(question: str) -> str | None:
    """调 LLM 判定。返回合法 intent 名, 或 None 表示"没拿到, 交给启发式"。"""
    template = _PROMPT_ZH if _query_language(question) == "zh" else _PROMPT
    try:
        raw = chat(
            [{"role": "user", "content": template.replace("{q}", question)}],
            temperature=0,
            max_tokens=120,
        )
    except Exception as e:
        log.warning(f"intent classify failed: {e}; falling back to local cues")
        return None

    # 模型常在 JSON 前后加寒暄或 ```json 围栏(真实验收见 qwen3.8-max 记录)。
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        log.warning("intent reply has no JSON object; falling back to local cues")
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        log.warning("intent reply is not valid JSON; falling back to local cues")
        return None
    if not isinstance(data, dict):
        return None

    intent = data.get("intent")
    if isinstance(intent, str) and intent.strip().lower() in _VALID_INTENTS:
        return intent.strip().lower()
    log.warning(f"unknown intent {intent!r}; falling back to local cues")
    return None


def _heuristic_intent(question: str) -> str | None:
    """按中英信号词判定意图。判不出来返回 None(由调用方落默认档)。"""
    if not question:
        return None
    lang = _query_language(question)
    if lang == "zh":
        tiers = (
            ("explore", _EXPLORE_CUES_ZH),
            ("reasoning", _REASONING_CUES_ZH),
            ("factual", _FACTUAL_CUES_ZH),
        )
        haystack = question
    else:
        tiers = (
            ("explore", _EXPLORE_CUES_EN),
            ("reasoning", _REASONING_CUES_EN),
            ("factual", _FACTUAL_CUES_EN),
        )
        haystack = f" {question.lower()} "
    for intent, cues in tiers:
        if any(cue in haystack for cue in cues):
            return intent
    return None


def _tier_params(intent_cfg, intent: str) -> dict:
    tier = getattr(intent_cfg, intent, None) or getattr(intent_cfg, _DEFAULT_INTENT)
    return {"top_k": tier.top_k, "max_iter": tier.max_iter, "rrf_k": tier.rrf_k}


__all__ = [
    "classify",
]
