"""rag.llm 极简 OpenAI 兼容客户端的行为契约测试(客户端与配置全打桩, 不发网络)。

切片 0: 配置守卫(base_url/api_key 缺失时 get_client 报错; chat_model 未设时
        chat 报错——错误信息指向环境变量, 是用户可执行的修复指引)。
切片 1: chat 语义(参数透传、model 形参覆盖配置、content 为 None 时返回空串、
        默认 temperature/max_tokens 与基准一致)。
切片 2: extra_body 确认偏离(配置为空表时 create 调用形参与基准逐键一致——
        不多传 extra_body 键; 非空时透传, 承载 Qwen enable_thinking=false)。
切片 3: 模块级单例(同配置复用同一客户端、api_key 变化自动重建、
        reset_client_for_test 丢弃缓存)。

接口约定(与基准一致):

    get_client() -> OpenAI                      # 进程级单例, 配置变化自动重建
    chat(messages, *, model=None, temperature=0.2, max_tokens=1024) -> str
    reset_client_for_test() -> None
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from paper_rag.rag import llm as llm_mod


@pytest.fixture(autouse=True)
def _fresh_singleton():
    """每个用例前后清空模块级客户端缓存, 用例之间互不串味。"""
    llm_mod.reset_client_for_test()
    yield
    llm_mod.reset_client_for_test()


def _conf(
    monkeypatch,
    *,
    base_url: str | None = "https://llm.example/v1",
    api_key: str | None = "sk-test",
    chat_model: str | None = "qwen-plus",
    extra_body: dict | None = None,
):
    import paper_rag.config as config

    conf = SimpleNamespace(
        llm=SimpleNamespace(
            base_url=base_url,
            api_key=api_key,
            chat_model=chat_model,
            small_model=None,
            extra_body=extra_body if extra_body is not None else {},
        )
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    return conf


class _FakeCompletions:
    def __init__(self, content: str | None = "ok"):
        self.calls: list[dict] = []
        self._content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _stub_client(monkeypatch, content: str | None = "ok") -> _FakeCompletions:
    completions = _FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_mod, "get_client", lambda: client)
    return completions


class _FakeOpenAI:
    constructed: list[_FakeOpenAI] = []

    def __init__(self, *, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        _FakeOpenAI.constructed.append(self)


def _stub_openai_module(monkeypatch):
    _FakeOpenAI.constructed = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))


# ---------- 切片 0: 配置守卫 ----------


def test_get_client_missing_base_url(monkeypatch):
    _conf(monkeypatch, base_url=None)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        llm_mod.get_client()


def test_get_client_missing_api_key(monkeypatch):
    _conf(monkeypatch, api_key=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_mod.get_client()


def test_chat_missing_model(monkeypatch):
    _conf(monkeypatch, chat_model=None)
    with pytest.raises(RuntimeError, match="CHAT_MODEL"):
        llm_mod.chat([{"role": "user", "content": "hi"}])


# ---------- 切片 1: chat 语义 ----------


def test_chat_passes_args_and_returns_content(monkeypatch):
    _conf(monkeypatch)
    completions = _stub_client(monkeypatch, content="回答内容")
    messages = [{"role": "user", "content": "什么是 Self-RAG?"}]

    out = llm_mod.chat(messages, temperature=0.3, max_tokens=400)

    assert out == "回答内容"
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "qwen-plus"
    assert call["messages"] is messages
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 400


def test_chat_model_param_overrides_config(monkeypatch):
    _conf(monkeypatch, chat_model="qwen-plus")
    completions = _stub_client(monkeypatch)
    llm_mod.chat([{"role": "user", "content": "hi"}], model="qwen-turbo")
    assert completions.calls[0]["model"] == "qwen-turbo"


def test_chat_none_content_returns_empty_string(monkeypatch):
    _conf(monkeypatch)
    _stub_client(monkeypatch, content=None)
    assert llm_mod.chat([{"role": "user", "content": "hi"}]) == ""


def test_chat_default_temperature_and_max_tokens(monkeypatch):
    _conf(monkeypatch)
    completions = _stub_client(monkeypatch)
    llm_mod.chat([{"role": "user", "content": "hi"}])
    call = completions.calls[0]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 1024


# ---------- 切片 2: extra_body 确认偏离 ----------


def test_empty_extra_body_keeps_baseline_call_shape(monkeypatch):
    """extra_body 为空表时, create 形参与基准逐键一致(不多传 extra_body)。"""
    _conf(monkeypatch, extra_body={})
    completions = _stub_client(monkeypatch)
    llm_mod.chat([{"role": "user", "content": "hi"}])
    assert sorted(completions.calls[0]) == ["max_tokens", "messages", "model", "temperature"]


def test_extra_body_passthrough(monkeypatch):
    """非空 extra_body 原样透传——Qwen 非流式必须 enable_thinking=false。"""
    _conf(monkeypatch, extra_body={"enable_thinking": False})
    completions = _stub_client(monkeypatch)
    llm_mod.chat([{"role": "user", "content": "hi"}])
    assert completions.calls[0]["extra_body"] == {"enable_thinking": False}


def test_extra_body_config_default_is_empty_dict():
    """default.yaml 的 extra_body 缺省为空表——未配置 Qwen 参数时行为与基准一致。"""
    import paper_rag.config as config

    config.load.cache_clear()
    try:
        assert config.load().llm.extra_body == {}
    finally:
        config.load.cache_clear()


# ---------- 切片 3: 模块级单例 ----------


def test_client_reused_for_same_config(monkeypatch):
    _conf(monkeypatch)
    _stub_openai_module(monkeypatch)
    c1 = llm_mod.get_client()
    c2 = llm_mod.get_client()
    assert c1 is c2
    assert len(_FakeOpenAI.constructed) == 1
    assert c1.base_url == "https://llm.example/v1"
    assert c1.api_key == "sk-test"


def test_client_rebuilt_when_key_changes(monkeypatch):
    conf = _conf(monkeypatch)
    _stub_openai_module(monkeypatch)
    c1 = llm_mod.get_client()
    conf.llm.api_key = "sk-rotated"
    c2 = llm_mod.get_client()
    assert c1 is not c2
    assert c2.api_key == "sk-rotated"
    assert len(_FakeOpenAI.constructed) == 2


def test_reset_client_for_test_drops_cache(monkeypatch):
    _conf(monkeypatch)
    _stub_openai_module(monkeypatch)
    c1 = llm_mod.get_client()
    llm_mod.reset_client_for_test()
    c2 = llm_mod.get_client()
    assert c1 is not c2
    assert len(_FakeOpenAI.constructed) == 2
