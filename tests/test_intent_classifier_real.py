"""rag.intent_classifier 真实 LLM 集成测试(无 mock, 真发网络)。

验收协议: 缺配置**明确失败, 不 skip**——真实验收不允许静默跳过。
`.env` 由 `tests/conftest.py` 统一加载。

覆盖:
- 英文三档判定(基准路径);
- 中文三档判定(中文模板的信号词引导有效);
- 出口契约: 四键恒齐全, 档位与 config 一致;
- 逃生门: enabled=false 时零 LLM 调用, 走本地启发式。
"""

from __future__ import annotations

import os

import pytest

import paper_rag.config as config
from paper_rag.rag import intent_classifier as ic
from paper_rag.rag import llm


@pytest.fixture(autouse=True)
def _fresh_config():
    config.load.cache_clear()
    llm.reset_client_for_test()
    yield
    config.load.cache_clear()
    llm.reset_client_for_test()


def _require_llm_env() -> None:
    missing = [
        var
        for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL")
        if not os.environ.get(var)
    ]
    if missing:
        pytest.fail(
            f"真实 LLM 配置缺失: {', '.join(missing)}。请在 .env 或环境变量中设置后重跑"
            "(验收协议: 缺配置明确失败, 不 skip)。"
        )


@pytest.mark.parametrize(
    ("expected", "question"),
    [
        ("factual", "What is the FactScore metric?"),
        ("reasoning", "How do Self-RAG and CRAG differ in their retrieval decisions?"),
        ("explore", "What are recent advances in retrieval augmented generation?"),
    ],
)
def test_real_english_three_tiers(expected, question):
    _require_llm_env()
    out = ic.classify(question)
    print(f"\n[en] {question}\n     -> {out}")
    assert out["intent"] == expected
    tier = getattr(config.load().rag.intent, expected)
    assert out["top_k"] == tier.top_k
    assert out["max_iter"] == tier.max_iter


@pytest.mark.parametrize(
    ("expected", "question"),
    [
        ("factual", "FactScore 指标是什么?"),
        ("reasoning", "Self-RAG 和 CRAG 在检索决策上有什么区别?"),
        ("explore", "检索增强生成近年有哪些研究进展?"),
    ],
)
def test_real_chinese_three_tiers(expected, question):
    """中文模板真实判定——纯中文提问也能判对三档。"""
    _require_llm_env()
    out = ic.classify(question)
    print(f"\n[zh] {question}\n     -> {out}")
    assert out["intent"] == expected
    tier = getattr(config.load().rag.intent, expected)
    assert out["top_k"] == tier.top_k


def test_real_return_shape_always_complete():
    _require_llm_env()
    out = ic.classify("Self-RAG 用反思 token 做什么?")
    print(f"\n[shape] {out}")
    assert sorted(out) == ["intent", "max_iter", "rrf_k", "top_k"]
    assert out["intent"] in ("factual", "reasoning", "explore")


def test_real_disabled_makes_no_llm_call(monkeypatch):
    """逃生门: enabled=false 时零网络调用, 本地启发式接管。"""
    _require_llm_env()
    conf = config.load()
    monkeypatch.setattr(conf.rag.intent, "enabled", False)
    monkeypatch.setattr(config, "load", lambda path=None: conf)

    called = {"n": 0}
    real_chat = ic.chat

    def _counting_chat(*a, **kw):
        called["n"] += 1
        return real_chat(*a, **kw)

    monkeypatch.setattr(ic, "chat", _counting_chat)
    out = ic.classify("检索增强生成近年有哪些研究进展?")

    assert called["n"] == 0, "enabled=false 时仍发起了 LLM 调用"
    assert out["intent"] == "explore", "本地启发式未识别中文 explore 信号词"
    print(f"\n[escape-hatch] LLM 调用 {called['n']} 次, 本地判定 {out}")
