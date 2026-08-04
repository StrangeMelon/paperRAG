"""极简 OpenAI 兼容 chat 客户端。

从 config 读 base_url/api_key/chat_model, 返回纯字符串回复。

OpenAI 客户端缓存为模块级单例——构造涉及 TLS/HTTPX 初始化, 每次调用都重建
纯属浪费。缓存在 config 的 base_url/api_key 变化时自动失效重建, 测试打补丁
与热改配置都不受影响。

相对基准的一处确认偏离(2026-08-05): 配置项 ``llm.extra_body`` 非空时透传给
``chat.completions.create``——承载 OpenAI 兼容供应商的私有参数。直接动因是
Qwen(DashScope 兼容模式): 思考型模型非流式调用必须 ``enable_thinking: false``,
否则 400。空表(缺省)时调用形参与基准逐键一致, 换回 OpenAI 零成本。
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from .. import config as cfg
from ..utils.logger import get_logger

log = get_logger("rag.llm")

# 模块级单例状态。
_CLIENT: Any | None = None
_CLIENT_KEY: tuple[str, str] | None = None  # 构建 _CLIENT 时用的 (base_url, api_key)
_LOCK = Lock()


def get_client():
    """返回进程共享的 OpenAI 客户端。

    首次使用时惰性构建; config 的 base_url 或 api_key 在两次调用之间变化时
    (如测试 monkeypatch、配置热更新)自动重建。
    """
    global _CLIENT, _CLIENT_KEY

    c = cfg.load().llm
    if not (c.base_url and c.api_key):
        raise RuntimeError(
            "LLM config missing. Set OPENAI_BASE_URL / OPENAI_API_KEY / CHAT_MODEL env vars."
        )
    key = (c.base_url, c.api_key)

    # 快路径——已构建且配置未变。
    if _CLIENT is not None and key == _CLIENT_KEY:
        return _CLIENT

    with _LOCK:
        if _CLIENT is not None and key == _CLIENT_KEY:
            return _CLIENT
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed. Run: pip install openai") from e
        _CLIENT = OpenAI(base_url=c.base_url, api_key=c.api_key)
        _CLIENT_KEY = key
        log.debug("OpenAI client (re)built for base_url=%s", c.base_url)
        return _CLIENT


def reset_client_for_test() -> None:
    """丢弃缓存的客户端。供 monkeypatch 配置的测试使用。"""
    global _CLIENT, _CLIENT_KEY
    with _LOCK:
        _CLIENT = None
        _CLIENT_KEY = None


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    c = cfg.load().llm
    chosen = model or c.chat_model
    if not chosen:
        raise RuntimeError("CHAT_MODEL env / config.llm.chat_model not set")
    vendor_kwargs: dict[str, Any] = {}
    if c.extra_body:
        vendor_kwargs["extra_body"] = dict(c.extra_body)
    resp = get_client().chat.completions.create(
        model=chosen,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **vendor_kwargs,
    )
    return resp.choices[0].message.content or ""


__all__ = ["chat", "get_client", "reset_client_for_test"]
