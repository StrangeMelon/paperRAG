"""Wiki 专用 LLM 路由, 兼容 DeepSeek V4 Flash 的混合思考参数。"""

from __future__ import annotations

from typing import Any

from .. import config as cfg
from ..rag.llm import chat as base_chat


def _is_configured(wiki_llm: Any) -> bool:
    return any(
        (
            wiki_llm.base_url,
            wiki_llm.api_key,
            wiki_llm.model,
            wiki_llm.thinking,
            wiki_llm.reasoning_effort,
            wiki_llm.extra_body,
        )
    )


def _extra_body(wiki_llm: Any) -> dict[str, Any]:
    extra = dict(wiki_llm.extra_body)
    if wiki_llm.thinking is not None and not ({"thinking", "enable_thinking"} & extra.keys()):
        extra["thinking"] = {"type": wiki_llm.thinking}
    return extra


def _thinking_enabled(wiki_llm: Any, extra_body: dict[str, Any]) -> bool:
    if wiki_llm.thinking is not None:
        return wiki_llm.thinking == "enabled"
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in {"enabled", "disabled"}:
        return thinking["type"] == "enabled"
    if isinstance(extra_body.get("enable_thinking"), bool):
        return bool(extra_body["enable_thinking"])
    return wiki_llm.reasoning_effort is not None


def chat(messages: list[dict], *, max_tokens: int) -> str:
    """调用 Wiki 模型; 未配置 wiki.llm 时完整沿用全局 LLM 行为。"""
    conf = cfg.load()
    wiki_llm = conf.wiki.llm
    temperature = conf.llm.temperatures.wiki
    if not _is_configured(wiki_llm):
        return base_chat(messages, temperature=temperature, max_tokens=max_tokens)

    extra_body = _extra_body(wiki_llm)
    thinking_enabled = _thinking_enabled(wiki_llm, extra_body)
    reasoning_effort = wiki_llm.reasoning_effort if thinking_enabled else None
    return base_chat(
        messages,
        model=wiki_llm.model,
        temperature=None if thinking_enabled else temperature,
        max_tokens=max_tokens,
        base_url=wiki_llm.base_url,
        api_key=wiki_llm.api_key,
        extra_body=extra_body,
        reasoning_effort=reasoning_effort,
        timeout_sec=wiki_llm.timeout_sec,
    )


__all__ = ["chat"]
