"""rag.qa_stream 流式 QA 事件协议的行为契约测试(组件全打桩, 不发网络)。

打桩面: config.load / classify / _retrieve_round / reflect / _stream_chat;
abstain、citation_check、evidence_select 用真实纯函数(块分数控制决策)。

切片 0: 事件序列(intent 首、rewrite 仅首轮、retrieved 每轮、done 尾;
        answer_chunk 拼接与 done.answer 的关系)。
切片 1: 多轮循环(reflect 事件、follow_up 驱动次轮、sufficient 停轮)。
切片 2: 短路路径(检索零块直达 done zh/en 文案; abstain no_evidence 拒答文案
        经 answer_chunk 流出且流式 LLM 不被调用, 文案按语言路由)。
切片 3: weak 注入中/英 hint; confident 无 hint。
切片 4: 错误路径(检索异常/流式异常 -> error 事件终止, 无 done)。
切片 5: 引用管道(done.answer 为净化版, citations 只认证据集)。
切片 6: zh/en 系统与用户模板路由; done 载荷 schema 与 paper_ids 汇总。
"""

from __future__ import annotations

from types import SimpleNamespace

import paper_rag.config as config
from paper_rag.rag import qa_stream as qs

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"
_ID_C = "c5f29d4e39f6a0b1c2d3"

_ZH_MSG = "未找到相关内容。"
_EN_MSG = "No relevant content was found."

_RW = {"dense_queries": ["q1", "q2"], "bm25_query": "kw1 kw2"}


def _chunk(cid: str, score: float = 0.9, paper: str = "p1") -> dict:
    return {
        "chunk_id": cid,
        "paper_id": paper,
        "section": "Body",
        "modality": "text",
        "score_rerank": score,
        "text": f"body of {cid}",
    }


def _conf(monkeypatch, *, max_inner_iters: int = 3, enable_reflect: bool = True):
    conf = SimpleNamespace(
        rag=SimpleNamespace(
            max_inner_iters=max_inner_iters,
            enable_reflect=enable_reflect,
            abstain=SimpleNamespace(
                enabled=True,
                threshold_low=0.21,
                threshold_high=0.48,
                min_chunks=3,
                no_evidence_message=_ZH_MSG,
                no_evidence_message_en=_EN_MSG,
            ),
        ),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    return conf


class _FakeStream:
    """记录 (system, user) 并逐 token 产出的 _stream_chat 桩。"""

    def __init__(self, tokens: list[str] | Exception):
        self.calls: list[tuple[str, str]] = []
        self._tokens = tokens

    def __call__(self, system: str, user: str):
        self.calls.append((system, user))
        if isinstance(self._tokens, Exception):
            raise self._tokens
        yield from self._tokens

    @property
    def system(self) -> str:
        return self.calls[0][0]

    @property
    def user(self) -> str:
        return self.calls[0][1]


def _wire(
    monkeypatch,
    *,
    chunks_per_round: list[list[dict]],
    reflects: list[dict] | None = None,
    tokens: list[str] | Exception | None = None,
    intent: dict | None = None,
    retrieve_error: Exception | None = None,
    **conf_over,
) -> _FakeStream:
    _conf(monkeypatch, **conf_over)
    monkeypatch.setattr(
        qs,
        "classify",
        lambda q: intent or {"intent": "reasoning", "top_k": 4, "max_iter": 2, "rrf_k": 60},
    )
    rounds = list(chunks_per_round)

    def _retrieve(query, paper_ids, top_k):
        if retrieve_error is not None:
            raise retrieve_error
        return (rounds.pop(0) if rounds else []), _RW

    monkeypatch.setattr(qs, "_retrieve_round", _retrieve)
    refl = list(reflects or [])
    monkeypatch.setattr(
        qs,
        "reflect",
        lambda q, e: (
            refl.pop(0)
            if refl
            else {"sufficiency": "sufficient", "missing": "", "follow_up": "", "score": 0.9}
        ),
    )
    fake = _FakeStream(tokens if tokens is not None else [f"ok [chunk:{_ID_A}]"])
    monkeypatch.setattr(qs, "_stream_chat", fake)
    return fake


def _events(question: str = "What is X?", **kw) -> list[dict]:
    return list(qs.stream_answer(question, **kw))


def _names(events: list[dict]) -> list[str]:
    return [e["event"] for e in events]


# ---------- 切片 0: 事件序列 ----------


def test_happy_path_event_sequence(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    intent = {"intent": "factual", "top_k": 4, "max_iter": 1, "rrf_k": 60}
    _wire(monkeypatch, chunks_per_round=[chunks], intent=intent, tokens=["a ", f"[chunk:{_ID_A}]"])
    events = _events()
    assert _names(events) == [
        "intent",
        "rewrite",
        "retrieved",
        "abstain",
        "answer_chunk",
        "answer_chunk",
        "done",
    ]
    assert events[0]["data"]["intent"] == "factual"
    assert events[2]["data"] == {"iter": 0, "n_chunks": 3}
    assert events[-1]["data"]["answer"] == f"a [chunk:{_ID_A}]"


def test_answer_chunks_join_equals_done_answer_when_clean(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], tokens=["Hello ", f"[chunk:{_ID_A}]", " world"])
    events = _events()
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "answer_chunk")
    done = events[-1]["data"]
    assert done["suspicious"]["count"] == 0
    assert streamed == done["answer"]


def test_rewrite_event_payload_and_only_first_round(monkeypatch):
    insufficient = {"sufficiency": "insufficient", "missing": "m", "follow_up": "f2", "score": 0.1}
    _wire(
        monkeypatch,
        chunks_per_round=[[_chunk(_ID_A)], [_chunk(_ID_B)]],
        reflects=[insufficient],
    )
    events = _events()
    rewrites = [e for e in events if e["event"] == "rewrite"]
    assert len(rewrites) == 1, "rewrite 事件只在首轮发出"
    assert rewrites[0]["data"] == {"queries": ["q1", "q2"], "keywords": "kw1 kw2"}


# ---------- 切片 1: 多轮循环 ----------


def test_follow_up_drives_second_round_with_reflect_event(monkeypatch):
    insufficient = {"sufficiency": "insufficient", "missing": "m", "follow_up": "f2", "score": 0.1}
    _wire(
        monkeypatch,
        chunks_per_round=[[_chunk(_ID_A)], [_chunk(_ID_B)]],
        reflects=[insufficient],
    )
    events = _events()
    retrieved = [e["data"]["iter"] for e in events if e["event"] == "retrieved"]
    assert retrieved == [0, 1]
    reflect_events = [e for e in events if e["event"] == "reflect"]
    assert len(reflect_events) == 1
    assert reflect_events[0]["data"]["sufficiency"] == "insufficient"


def test_sufficient_reflect_stops_loop(monkeypatch):
    _wire(monkeypatch, chunks_per_round=[[_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]])
    events = _events()
    assert [e["data"]["iter"] for e in events if e["event"] == "retrieved"] == [0]
    assert any(e["event"] == "reflect" for e in events)


# ---------- 切片 2: 短路路径 ----------


def test_no_chunks_goes_straight_to_done_english(monkeypatch):
    fake = _wire(monkeypatch, chunks_per_round=[[]])
    events = _events()
    assert _names(events)[-1] == "done"
    assert not any(e["event"] in ("abstain", "answer_chunk") for e in events)
    done = events[-1]["data"]
    assert done["answer"] == "(no evidence found)"
    assert done["degraded"] == "no_chunks"
    assert done["abstain"]["decision"] == "no_chunks"
    assert not fake.calls


def test_no_chunks_chinese_message(monkeypatch):
    _wire(monkeypatch, chunks_per_round=[[]])
    events = _events("图神经网络是什么?")
    assert events[-1]["data"]["answer"] == "(未检索到证据)"


def test_abstain_no_evidence_streams_message_without_llm(monkeypatch):
    noise = [_chunk(_ID_A, 0.05), _chunk(_ID_B, 0.05), _chunk(_ID_C, 0.05)]
    fake = _wire(monkeypatch, chunks_per_round=[noise])
    events = _events()
    assert not fake.calls, "no_evidence 时流式 LLM 不得被调用"
    abstain = next(e for e in events if e["event"] == "abstain")
    assert abstain["data"]["decision"] == "no_evidence"
    chunk_events = [e for e in events if e["event"] == "answer_chunk"]
    assert len(chunk_events) == 1
    assert chunk_events[0]["data"]["text"] == _EN_MSG, "英文问题得英文拒答文案"
    assert events[-1]["data"]["answer"] == _EN_MSG


def test_abstain_no_evidence_chinese_message(monkeypatch):
    noise = [_chunk(_ID_A, 0.05), _chunk(_ID_B, 0.05), _chunk(_ID_C, 0.05)]
    _wire(monkeypatch, chunks_per_round=[noise])
    events = _events("图神经网络是什么?")
    assert events[-1]["data"]["answer"] == _ZH_MSG


# ---------- 切片 3: weak hint ----------


def test_weak_injects_hint_by_language(monkeypatch):
    mid = [_chunk(_ID_A, 0.30), _chunk(_ID_B, 0.30), _chunk(_ID_C, 0.30)]
    fake = _wire(monkeypatch, chunks_per_round=[mid])
    _events()
    assert "NOTE: The retrieved evidence appears WEAK" in fake.user

    fake2 = _wire(monkeypatch, chunks_per_round=[mid])
    _events("图神经网络是什么?")
    assert "注意" in fake2.user and "证据" in fake2.user


def test_confident_no_hint(monkeypatch):
    fake = _wire(monkeypatch, chunks_per_round=[[_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]])
    _events()
    assert "WEAK" not in fake.user


# ---------- 切片 4: 错误路径 ----------


def test_retrieve_error_yields_error_and_stops(monkeypatch):
    _wire(monkeypatch, chunks_per_round=[], retrieve_error=RuntimeError("qdrant down"))
    events = _events()
    assert _names(events) == ["intent", "error"]
    assert "retrieve failed" in events[-1]["data"]["message"]


def test_stream_error_yields_error_no_done(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], tokens=RuntimeError("connection reset"))
    events = _events()
    assert events[-1]["event"] == "error"
    assert "chat stream failed" in events[-1]["data"]["message"]
    assert not any(e["event"] == "done" for e in events)


# ---------- 切片 5: 引用管道 ----------


def test_citation_pipeline_cleans_done_answer(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    tokens = [f"ok [chunk:{_ID_A}]", " fake [chunk:ffffffffffffffffffff]", " habit [1]."]
    _wire(monkeypatch, chunks_per_round=[chunks], tokens=tokens)
    events = _events()
    done = events[-1]["data"]
    assert done["citations"] == [_ID_A]
    assert "ffffffff" not in done["answer"]
    assert "[1]" not in done["answer"]
    assert done["suspicious"]["count"] == 1
    # 流出的原始 token 与净化后的 done.answer 允许不同——前端以 done 为准
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "answer_chunk")
    assert streamed != done["answer"]


# ---------- 切片 6: 模板路由与 done 载荷 ----------


def test_chinese_question_routes_chinese_prompts(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    fake = _wire(monkeypatch, chunks_per_round=[chunks], tokens=[f"好 [chunk:{_ID_A}]"])
    _events("图神经网络是什么?")
    assert "学术研究助手" in fake.system
    assert "问题:" in fake.user
    assert "最多使用 2 个引用" in fake.user


def test_english_prompts_have_allowed_tokens(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    fake = _wire(monkeypatch, chunks_per_round=[chunks])
    _events()
    assert "academic research assistant" in fake.system
    assert "Allowed citation tokens:" in fake.user
    assert "at most 2 citations" in fake.user


def test_done_payload_schema_and_paper_ids(monkeypatch):
    chunks = [_chunk(_ID_A, paper="p2"), _chunk(_ID_B, paper="p1"), _chunk(_ID_C, paper="p2")]
    _wire(monkeypatch, chunks_per_round=[chunks])
    done = _events()[-1]["data"]
    for key in (
        "answer",
        "citations",
        "suspicious",
        "abstain",
        "n_chunks",
        "evidence_chunks",
        "evidence_selection",
        "paper_ids",
    ):
        assert key in done, f"done 缺 {key}"
    assert done["paper_ids"] == ["p1", "p2"], "paper_ids 去重且有序"
    assert done["n_chunks"] == 3
