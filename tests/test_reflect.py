"""rag.reflect 反思式循环控制器的行为契约测试(LLM 打桩, 不发网络)。

切片 0: 三态正常解析(sufficient/partial/insufficient 四键契约)。
切片 1: JSON 提取健壮性(寒暄包裹、markdown 代码块、非 JSON 回复落缺省)。
切片 2: 异常兜底(LLM 抛异常 -> sufficient 缺省, 宁停不空转)。
切片 3: 缺陷修复——输出净化(基准把 float(score) 放在 try 外, LLM 回
        "score": "high" 会炸穿整个 QA 请求; 重建版安全强转 + [0,1] 裁剪,
        sufficiency 大小写归一 + 三值域校验, missing/follow_up 非串强转空串)。
切片 4: LLM 调用形参(temperature=0、max_tokens=300、证据 6000 字符截断、
        prompt 含原始问题)。
切片 5: prompt 语言路由(zh 问题走中文模板, en 走基准英文模板)。

接口约定(与基准一致):

    reflect(question: str, evidence: str) -> dict
        {"sufficiency": "sufficient"|"partial"|"insufficient",
         "missing": str, "follow_up": str, "score": float}
"""

from __future__ import annotations

import json

import pytest

from paper_rag.rag import reflect as rf


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
    monkeypatch.setattr(rf, "chat", fake)
    return fake


def _reply(sufficiency: str, missing: str = "", follow_up: str = "", score: float = 0.9) -> str:
    return json.dumps(
        {"sufficiency": sufficiency, "missing": missing, "follow_up": follow_up, "score": score},
        ensure_ascii=False,
    )


_FALLBACK = {"sufficiency": "sufficient", "missing": "", "follow_up": "", "score": 0.5}


# ---------- 切片 0: 三态正常解析 ----------


def test_sufficient_parses_all_four_keys(monkeypatch):
    _stub_chat(monkeypatch, _reply("sufficient", score=0.95))
    out = rf.reflect("q", "evidence text")
    assert out == {"sufficiency": "sufficient", "missing": "", "follow_up": "", "score": 0.95}


def test_insufficient_carries_missing_and_follow_up(monkeypatch):
    _stub_chat(
        monkeypatch,
        _reply("insufficient", missing="no ImageNet numbers", follow_up="ImageNet acc", score=0.2),
    )
    out = rf.reflect("q", "evidence")
    assert out["sufficiency"] == "insufficient"
    assert out["missing"] == "no ImageNet numbers"
    assert out["follow_up"] == "ImageNet acc"
    assert out["score"] == 0.2


def test_partial_is_a_valid_state(monkeypatch):
    _stub_chat(monkeypatch, _reply("partial", follow_up="more on X", score=0.5))
    assert rf.reflect("q", "e")["sufficiency"] == "partial"


# ---------- 切片 1: JSON 提取健壮性 ----------


def test_json_wrapped_in_prose(monkeypatch):
    _stub_chat(monkeypatch, "好的, 评估结果如下:\n" + _reply("partial") + "\n以上。")
    assert rf.reflect("q", "e")["sufficiency"] == "partial"


def test_json_wrapped_in_markdown_fence(monkeypatch):
    _stub_chat(monkeypatch, "```json\n" + _reply("insufficient", follow_up="f") + "\n```")
    out = rf.reflect("q", "e")
    assert out["sufficiency"] == "insufficient"
    assert out["follow_up"] == "f"


def test_non_json_reply_falls_back_to_defaults(monkeypatch):
    _stub_chat(monkeypatch, "I cannot answer in JSON, sorry.")
    assert rf.reflect("q", "e") == _FALLBACK


def test_missing_keys_fall_back_per_key(monkeypatch):
    _stub_chat(monkeypatch, json.dumps({"sufficiency": "partial"}))
    out = rf.reflect("q", "e")
    assert out == {"sufficiency": "partial", "missing": "", "follow_up": "", "score": 0.5}


# ---------- 切片 2: 异常兜底 ----------


def test_llm_exception_assumes_sufficient(monkeypatch):
    """LLM 挂掉时宁停不空转: 返回 sufficient 让循环收敛, 不炸主链路。"""
    _stub_chat(monkeypatch, RuntimeError("connection refused"))
    assert rf.reflect("q", "e") == _FALLBACK


def test_broken_json_assumes_sufficient(monkeypatch):
    _stub_chat(monkeypatch, '{"sufficiency": "partial", INVALID')
    assert rf.reflect("q", "e") == _FALLBACK


# ---------- 切片 3: 缺陷修复——输出净化 ----------


def test_non_numeric_score_does_not_crash(monkeypatch):
    """基准缺陷: float(data["score"]) 在 try 外, "high" 会 ValueError 炸穿
    QA 请求。重建版安全强转落 0.5。"""
    _stub_chat(monkeypatch, json.dumps({"sufficiency": "partial", "score": "high"}))
    out = rf.reflect("q", "e")
    assert out["sufficiency"] == "partial"
    assert out["score"] == 0.5


def test_numeric_string_score_is_accepted(monkeypatch):
    _stub_chat(monkeypatch, json.dumps({"sufficiency": "partial", "score": "0.8"}))
    assert rf.reflect("q", "e")["score"] == 0.8


@pytest.mark.parametrize(("raw", "expected"), [(1.7, 1.0), (-0.3, 0.0), (0.42, 0.42)])
def test_score_clamped_to_unit_interval(monkeypatch, raw, expected):
    _stub_chat(monkeypatch, json.dumps({"sufficiency": "partial", "score": raw}))
    assert rf.reflect("q", "e")["score"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sufficient", "sufficient"),
        ("INSUFFICIENT", "insufficient"),
        (" partial ", "partial"),
        ("unknown_state", "sufficient"),  # 域外值与缺键同缺省: 宁停不空转
        (None, "sufficient"),
        (3, "sufficient"),
    ],
)
def test_sufficiency_normalized_to_domain(monkeypatch, raw, expected):
    _stub_chat(monkeypatch, json.dumps({"sufficiency": raw, "score": 0.5}))
    assert rf.reflect("q", "e")["sufficiency"] == expected


def test_null_missing_and_follow_up_become_empty_str(monkeypatch):
    """显式 null 与非字符串一律落空串: 循环里 `if r["follow_up"]` 语义不变,
    trace 里不出现 None。"""
    _stub_chat(
        monkeypatch,
        json.dumps({"sufficiency": "partial", "missing": None, "follow_up": ["a", "b"]}),
    )
    out = rf.reflect("q", "e")
    assert out["missing"] == ""
    assert out["follow_up"] == ""


# ---------- 切片 4: LLM 调用形参 ----------


def test_call_params_deterministic_and_bounded(monkeypatch):
    fake = _stub_chat(monkeypatch, _reply("sufficient"))
    rf.reflect("my question", "my evidence")
    call = fake.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 300
    assert "my question" in fake.prompt
    assert "my evidence" in fake.prompt


def test_evidence_truncated_to_6000_chars(monkeypatch):
    fake = _stub_chat(monkeypatch, _reply("sufficient"))
    evidence = "x" * 6000 + "TAIL_MARKER"
    rf.reflect("q", evidence)
    assert "x" * 6000 in fake.prompt
    assert "TAIL_MARKER" not in fake.prompt


# ---------- 切片 5: prompt 语言路由 ----------


def test_english_question_uses_english_template(monkeypatch):
    fake = _stub_chat(monkeypatch, _reply("sufficient"))
    rf.reflect("How does Graph-Mamba handle long-range dependencies?", "e")
    assert "Evidence (truncated):" in fake.prompt
    assert "证据" not in fake.prompt


def test_chinese_question_uses_chinese_template(monkeypatch):
    fake = _stub_chat(monkeypatch, _reply("sufficient"))
    rf.reflect("区块链的网络架构是怎样设计的?", "e")
    assert "证据" in fake.prompt
    assert "Evidence (truncated):" not in fake.prompt
