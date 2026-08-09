"""DeepSeek V4 Flash 真实验收脚本的纯逻辑测试。"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _mod():
    return importlib.import_module("scripts.demo_deepseek_v4_flash")


def test_validate_config_requires_complete_deepseek_settings() -> None:
    mod = _mod()
    with pytest.raises(ValueError, match="WIKI_LLM_API_KEY"):
        mod._validate_config(
            SimpleNamespace(
                base_url="https://api.deepseek.com",
                api_key=None,
                model="deepseek-v4-flash",
            )
        )


def test_validate_config_rejects_other_model() -> None:
    mod = _mod()
    with pytest.raises(ValueError, match="不是 deepseek-v4-flash"):
        mod._validate_config(
            SimpleNamespace(
                base_url="https://api.deepseek.com",
                api_key="sk-test",
                model="deepseek-v4-pro",
            )
        )


def test_parse_json_reply_accepts_fenced_payload() -> None:
    mod = _mod()
    payload = mod._parse_json_reply(
        """```json
        {"concepts":[{"name":"RAG","category":"method","definition":"Grounded generation."}]}
        ```"""
    )
    assert payload["concepts"][0]["name"] == "RAG"


@pytest.mark.parametrize(
    "reply",
    [
        "no json",
        '{"concepts":[]}',
        '{"concepts":[{"name":"RAG","category":"method","definition":""}]}',
    ],
)
def test_parse_json_reply_rejects_invalid_contract(reply: str) -> None:
    mod = _mod()
    with pytest.raises(ValueError):
        mod._parse_json_reply(reply)
