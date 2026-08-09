"""Wiki 专用 LLM 路由契约: DeepSeek 特殊参数与全局配置回退。"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _mod():
    return importlib.import_module("paper_rag.wiki.llm")


def _conf(*, wiki_llm):
    return SimpleNamespace(
        llm=SimpleNamespace(temperatures=SimpleNamespace(wiki=0.2)),
        wiki=SimpleNamespace(llm=wiki_llm),
    )


def _wiki_llm(**overrides):
    values = {
        "base_url": None,
        "api_key": None,
        "model": None,
        "thinking": None,
        "reasoning_effort": None,
        "timeout_sec": 120.0,
        "extra_body": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture(monkeypatch, mod):
    calls = []

    def _chat(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return "ok"

    monkeypatch.setattr(mod, "base_chat", _chat)
    return calls


def test_unconfigured_wiki_llm_preserves_global_call(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.cfg, "load", lambda: _conf(wiki_llm=_wiki_llm()))
    calls = _capture(monkeypatch, mod)

    assert mod.chat([{"role": "user", "content": "hi"}], max_tokens=200) == "ok"
    assert calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 200,
        }
    ]


def test_deepseek_non_thinking_is_explicit_and_keeps_temperature(monkeypatch):
    mod = _mod()
    wiki_llm = _wiki_llm(
        base_url="https://api.deepseek.com",
        api_key="sk-deepseek",
        model="deepseek-v4-flash",
        thinking="disabled",
    )
    monkeypatch.setattr(mod.cfg, "load", lambda: _conf(wiki_llm=wiki_llm))
    calls = _capture(monkeypatch, mod)

    mod.chat([{"role": "user", "content": "extract"}], max_tokens=2000)
    call = calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["temperature"] == 0.2
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["reasoning_effort"] is None
    assert call["timeout_sec"] == 120.0


def test_deepseek_thinking_omits_temperature_and_sends_effort(monkeypatch):
    mod = _mod()
    wiki_llm = _wiki_llm(
        model="deepseek-v4-flash",
        thinking="enabled",
        reasoning_effort="low",
    )
    monkeypatch.setattr(mod.cfg, "load", lambda: _conf(wiki_llm=wiki_llm))
    calls = _capture(monkeypatch, mod)

    mod.chat([{"role": "user", "content": "judge"}], max_tokens=200)
    call = calls[0]
    assert call["temperature"] is None
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert call["reasoning_effort"] == "low"


def test_provider_extra_body_can_use_enable_thinking(monkeypatch):
    mod = _mod()
    wiki_llm = _wiki_llm(
        model="vanchin/deepseek-v4-flash",
        extra_body={"enable_thinking": False},
    )
    monkeypatch.setattr(mod.cfg, "load", lambda: _conf(wiki_llm=wiki_llm))
    calls = _capture(monkeypatch, mod)

    mod.chat([{"role": "user", "content": "extract"}], max_tokens=2000)
    call = calls[0]
    assert call["temperature"] == 0.2
    assert call["extra_body"] == {"enable_thinking": False}
