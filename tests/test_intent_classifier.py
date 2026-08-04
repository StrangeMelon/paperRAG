"""rag.intent_classifier 意图分类的行为契约测试(LLM 与配置全打桩, 不发网络)。

切片 0: 三类正常判定(factual/reasoning/explore 各自带出配置档位)。
切片 1: 永不抛异常(带寒暄的 JSON、非 JSON、缺键、未知 intent 名、LLM 抛异常
        ——五种都落默认档而非崩溃; 真实验收已知 qwen3.8-max 会加寒暄)。
切片 2: LLM 调用形参(temperature=0 求确定性、max_tokens 上限、prompt 含问题)。
切片 3: 配置驱动档位(rag.intent 三档参数可调; enabled=false 时不发调用)。
切片 4: prompt 语言路由(zh 问题走中文模板, en 走英文模板)。
切片 5: 本地启发式兜底(LLM 不可用时按中英信号词判定, 而非一律落中间档)。

接口约定(与基准一致):

    classify(question: str) -> dict
        {"intent": "factual"|"reasoning"|"explore", "top_k": int,
         "max_iter": int, "rrf_k": int}
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paper_rag.rag import intent_classifier as ic


def _conf(
    monkeypatch,
    *,
    enabled: bool = True,
    base_url: str | None = "https://llm.example/v1",
    api_key: str | None = "sk-test",
    chat_model: str | None = "qwen-plus",
    factual: dict | None = None,
    reasoning: dict | None = None,
    explore: dict | None = None,
):
    import paper_rag.config as config

    def _tier(over: dict | None, top_k: int, max_iter: int) -> SimpleNamespace:
        base = {"top_k": top_k, "max_iter": max_iter, "rrf_k": 60}
        base.update(over or {})
        return SimpleNamespace(**base)

    conf = SimpleNamespace(
        llm=SimpleNamespace(base_url=base_url, api_key=api_key, chat_model=chat_model),
        rag=SimpleNamespace(
            intent=SimpleNamespace(
                enabled=enabled,
                factual=_tier(factual, 5, 1),
                reasoning=_tier(reasoning, 10, 2),
                explore=_tier(explore, 15, 3),
            )
        ),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    return conf


class _FakeChat:
    def __init__(self, reply: str | Exception):
        self.calls: list[dict] = []
        self._reply = reply

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    @property
    def prompt(self) -> str:
        return self.calls[0]["messages"][0]["content"]


def _stub_chat(monkeypatch, reply: str | Exception) -> _FakeChat:
    fake = _FakeChat(reply)
    monkeypatch.setattr(ic, "chat", fake)
    return fake


def _reply(intent: str, reason: str = "because") -> str:
    return json.dumps({"intent": intent, "reason": reason}, ensure_ascii=False)


# ---------- 切片 0: 三类正常判定 ----------


@pytest.mark.parametrize(
    ("intent", "top_k", "max_iter"),
    [("factual", 5, 1), ("reasoning", 10, 2), ("explore", 15, 3)],
)
def test_three_intents_carry_their_tier(monkeypatch, intent, top_k, max_iter):
    _conf(monkeypatch)
    _stub_chat(monkeypatch, _reply(intent))

    out = ic.classify("What is the FactScore of Self-RAG?")

    assert out["intent"] == intent
    assert out["top_k"] == top_k
    assert out["max_iter"] == max_iter
    assert out["rrf_k"] == 60


# ---------- 切片 1: 永不抛异常 ----------


def test_json_with_surrounding_prose(monkeypatch):
    """模型爱加"好的,以下是分类结果:"之类寒暄, 正则应抠出对象。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, f"好的, 分类如下:\n```json\n{_reply('explore')}\n```\n希望有用!")

    assert ic.classify("What are recent advances in RAG?")["intent"] == "explore"


@pytest.mark.parametrize(
    "reply",
    [
        "完全不是 JSON 的一段话。",
        "",
        "{ broken json ",
        '{"reason": "no intent key"}',
        '{"intent": "wild-guess"}',  # 不认识的 intent 名
        '{"intent": null}',
        '{"intent": ["factual"]}',  # 类型不符
    ],
)
def test_dirty_reply_falls_back_to_default_tier(monkeypatch, reply):
    _conf(monkeypatch)
    _stub_chat(monkeypatch, reply)

    out = ic.classify("Some question with no obvious signal word")

    assert out["intent"] == "reasoning"
    assert out["top_k"] == 10 and out["max_iter"] == 2


def test_llm_exception_falls_back(monkeypatch):
    _conf(monkeypatch)
    _stub_chat(monkeypatch, RuntimeError("429 rate limited"))

    out = ic.classify("Some question with no obvious signal word")

    assert out["intent"] == "reasoning"
    assert out["top_k"] == 10


def test_return_shape_is_always_complete(monkeypatch):
    """无论走哪条路径, 出口四键齐全——qa_agentic 直接下标取值, 不做防御。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, RuntimeError("boom"))

    out = ic.classify("anything")

    assert sorted(out) == ["intent", "max_iter", "rrf_k", "top_k"]


# ---------- 切片 2: LLM 调用形参 ----------


def test_classify_call_args(monkeypatch):
    _conf(monkeypatch)
    fake = _stub_chat(monkeypatch, _reply("factual"))

    ic.classify("What is the FactScore of Self-RAG?")

    call = fake.calls[0]
    assert call["temperature"] == 0, "分类要确定性, 不要采样多样性"
    assert call["max_tokens"] == 120
    assert "What is the FactScore of Self-RAG?" in fake.prompt


# ---------- 切片 3: 配置驱动档位 ----------


def test_tiers_come_from_config(monkeypatch):
    """档位参数可调——基准把 5/10/15 写死在模块里, 违反"不硬编码可调项"。"""
    _conf(monkeypatch, explore={"top_k": 24, "max_iter": 4, "rrf_k": 90})
    _stub_chat(monkeypatch, _reply("explore"))

    out = ic.classify("What are recent advances in RAG?")

    assert out["top_k"] == 24 and out["max_iter"] == 4 and out["rrf_k"] == 90


def test_disabled_skips_llm_call(monkeypatch):
    """enabled=false 时省掉一次 LLM 往返, 直接返回默认档。"""
    _conf(monkeypatch, enabled=False)
    fake = _stub_chat(monkeypatch, _reply("explore"))

    # 用不含任何信号词的问题, 隔离出"禁用 + 启发式判不出"这一条路径。
    out = ic.classify("Some question with no obvious signal word")

    assert fake.calls == [], "禁用时不应发起 LLM 调用"
    assert out["intent"] == "reasoning" and out["top_k"] == 10


@pytest.mark.parametrize("missing", [{"base_url": None}, {"api_key": None}, {"chat_model": None}])
def test_llm_not_configured_skips_call(monkeypatch, missing):
    _conf(monkeypatch, **missing)
    fake = _stub_chat(monkeypatch, _reply("explore"))

    out = ic.classify("Some question with no obvious signal word")

    assert fake.calls == [], "未配置 LLM 时不应发起调用"
    assert out["intent"] == "reasoning"


def test_intent_config_defaults_match_baseline():
    """default.yaml 的三档缺省值与基准硬编码一致——偏离只搬家, 不改行为。"""
    import paper_rag.config as config

    config.load.cache_clear()
    try:
        intent = config.load().rag.intent
        assert intent.enabled is True
        assert (intent.factual.top_k, intent.factual.max_iter) == (5, 1)
        assert (intent.reasoning.top_k, intent.reasoning.max_iter) == (10, 2)
        assert (intent.explore.top_k, intent.explore.max_iter) == (15, 3)
        assert intent.reasoning.rrf_k == 60
    finally:
        config.load.cache_clear()


# ---------- 切片 4: prompt 语言路由 ----------


def test_zh_question_uses_chinese_prompt(monkeypatch):
    _conf(monkeypatch)
    fake = _stub_chat(monkeypatch, _reply("reasoning"))

    ic.classify("Self-RAG 和 CRAG 有什么区别?")

    prompt = fake.prompt
    assert "Self-RAG 和 CRAG 有什么区别?" in prompt
    assert any("一" <= ch <= "鿿" for ch in prompt), "中文问题应使用中文模板"


def test_en_question_uses_english_prompt(monkeypatch):
    _conf(monkeypatch)
    fake = _stub_chat(monkeypatch, _reply("reasoning"))

    ic.classify("How do Self-RAG and CRAG differ?")

    prompt = fake.prompt
    assert not any("一" <= ch <= "鿿" for ch in prompt), "英文问题不应混入中文模板"
    assert "factual" in prompt and "reasoning" in prompt and "explore" in prompt


def test_language_router_is_shared_with_query_rewrite(monkeypatch):
    """语言判定复用 query_rewrite 的实现, 不重复维护两份 CJK 正则。"""
    from paper_rag.rag import query_rewrite as qr

    assert ic._query_language is qr._query_language


# ---------- 切片 5: 本地启发式兜底 ----------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Self-RAG 和 CRAG 有什么区别?", "reasoning"),
        ("对比一下这两种检索方式", "reasoning"),
        ("RAG 和微调相比有哪些差异", "reasoning"),
        ("RAG 近年有哪些进展?", "explore"),
        ("请综述一下检索增强生成的现状", "explore"),
        ("FactScore 是什么?", "factual"),
        ("Self-RAG 的准确率是多少?", "factual"),
    ],
)
def test_zh_heuristic_fallback(monkeypatch, question, expected):
    """LLM 不可用时按中文信号词判定, 而非一律落中间档。"""
    _conf(monkeypatch, enabled=False)
    _stub_chat(monkeypatch, RuntimeError("should not be called"))

    assert ic.classify(question)["intent"] == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How do Self-RAG and CRAG differ?", "reasoning"),
        ("Compare dense and sparse retrieval", "reasoning"),
        ("What are recent advances in RAG?", "explore"),
        ("Give me a survey of retrieval augmented generation", "explore"),
        ("What is FactScore?", "factual"),
    ],
)
def test_en_heuristic_fallback(monkeypatch, question, expected):
    _conf(monkeypatch, enabled=False)
    _stub_chat(monkeypatch, RuntimeError("should not be called"))

    assert ic.classify(question)["intent"] == expected


def test_heuristic_only_applies_when_llm_absent(monkeypatch):
    """LLM 可用且给出明确答案时, 启发式不得覆盖模型判定。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, _reply("factual"))

    # 问题含"区别"(启发式会判 reasoning), 但模型说 factual —— 以模型为准。
    assert ic.classify("Self-RAG 和 CRAG 有什么区别?")["intent"] == "factual"


def test_heuristic_used_when_llm_reply_is_dirty(monkeypatch):
    """模型返回脏输出时, 先试启发式再落中间档。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, "抱歉我无法分类")

    assert ic.classify("RAG 近年有哪些进展?")["intent"] == "explore"
