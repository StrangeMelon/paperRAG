"""极简 OpenAI 兼容 chat 客户端。

从 config 读 base_url/api_key/chat_model, 返回纯字符串回复。

OpenAI 客户端按 (base_url, api_key) 缓存——构造涉及 TLS/HTTPX 初始化, 每次调用
都重建纯属浪费。多端点缓存让全局 QA 与 Wiki 专用模型可在同一进程复用连接。

相对基准的一处确认偏离(2026-08-05): 配置项 ``llm.extra_body`` 非空时透传给
``chat.completions.create``——承载 OpenAI 兼容供应商的私有参数。直接动因是
Qwen(DashScope 兼容模式): 思考型模型非流式调用必须 ``enable_thinking: false``,
否则 400。空表(缺省)时调用形参与基准逐键一致, 换回 OpenAI 零成本。
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from .. import config as cfg
from ..mcp.resource_guards import hold_resource
from ..utils.logger import get_logger

log = get_logger("rag.llm")

# 模块级客户端池; 同一进程可同时连接全局 LLM 与 Wiki 专用 LLM。
_CLIENTS: dict[tuple[str, str], Any] = {}
_LOCK = Lock()


def get_client(*, base_url: str | None = None, api_key: str | None = None):
    """返回按端点与密钥复用的 OpenAI 客户端。

    覆盖参数为空时使用全局 llm 配置; Wiki 可传自己的 DeepSeek 端点与密钥。
    """
    c = cfg.load().llm
    resolved_base_url = base_url or c.base_url
    resolved_api_key = api_key or c.api_key
    if not (resolved_base_url and resolved_api_key):
        raise RuntimeError(
            "LLM config missing. Set OPENAI_BASE_URL / OPENAI_API_KEY / CHAT_MODEL env vars."
        )
    key = (resolved_base_url, resolved_api_key)

    if key in _CLIENTS:
        return _CLIENTS[key]

    with _LOCK:
        if key in _CLIENTS:
            return _CLIENTS[key]
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed. Run: pip install openai") from e
        client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        _CLIENTS[key] = client
        log.debug("OpenAI client built for base_url=%s", resolved_base_url)
        return client


def reset_client_for_test() -> None:
    """丢弃全部缓存客户端。供 monkeypatch 配置的测试使用。"""
    with _LOCK:
        _CLIENTS.clear()


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = 0.2,
    max_tokens: int = 1024,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    timeout_sec: float | None = None,
) -> str:
    c = cfg.load().llm
    chosen = model or c.chat_model
    if not chosen:
        raise RuntimeError("CHAT_MODEL env / config.llm.chat_model not set")
    request_kwargs: dict[str, Any] = {
        "model": chosen,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    resolved_extra_body = c.extra_body if extra_body is None else extra_body
    if resolved_extra_body:
        request_kwargs["extra_body"] = dict(resolved_extra_body)
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if timeout_sec is not None:
        request_kwargs["timeout"] = timeout_sec
    with hold_resource("llm"):
        client = (
            get_client()
            if base_url is None and api_key is None
            else get_client(base_url=base_url, api_key=api_key)
        )
        resp = client.chat.completions.create(**request_kwargs)
    return resp.choices[0].message.content or ""


__all__ = ["chat", "get_client", "reset_client_for_test"]
