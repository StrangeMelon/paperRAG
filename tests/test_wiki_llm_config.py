"""Wiki 专用 DeepSeek 配置契约。"""

from __future__ import annotations

import importlib

import pytest


def _config():
    return importlib.import_module("paper_rag.config")


def test_wiki_llm_expands_deepseek_environment(monkeypatch) -> None:
    config = _config()
    monkeypatch.delenv("PAPER_RAG_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("WIKI_LLM_API_KEY", "sk-deepseek")
    monkeypatch.setenv("WIKI_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("WIKI_LLM_THINKING", "enabled")
    monkeypatch.setenv("WIKI_LLM_REASONING_EFFORT", "low")
    config.load.cache_clear()
    try:
        wiki_llm = config.load().wiki.llm
        assert wiki_llm.base_url == "https://api.deepseek.com"
        assert wiki_llm.api_key == "sk-deepseek"
        assert wiki_llm.model == "deepseek-v4-flash"
        assert wiki_llm.thinking == "enabled"
        assert wiki_llm.reasoning_effort == "low"
        assert wiki_llm.timeout_sec == 120
    finally:
        config.load.cache_clear()


def test_wiki_llm_rejects_unknown_reasoning_effort() -> None:
    config = _config()

    with pytest.raises(ValueError):
        config._WikiLlm(reasoning_effort="extreme")


def test_wiki_llm_accepts_compat_reasoning_effort_aliases() -> None:
    config = _config()

    assert config._WikiLlm(reasoning_effort="medium").reasoning_effort == "medium"
    assert config._WikiLlm(reasoning_effort="xhigh").reasoning_effort == "xhigh"


def test_wiki_llm_requires_complete_dedicated_endpoint() -> None:
    config = _config()

    with pytest.raises(ValueError, match="base_url and api_key"):
        config._WikiLlm(base_url="https://api.deepseek.com", model="deepseek-v4-flash")
    with pytest.raises(ValueError, match="model is required"):
        config._WikiLlm(base_url="https://api.deepseek.com", api_key="sk-deepseek")
