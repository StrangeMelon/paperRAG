"""rag.qa_agentic 七 Stage 编排的行为契约测试(组件全打桩, 不发网络)。

打桩面: config.load / classify / _retrieve_round / reflect / select_evidence /
chat; abstain 与 citation_check 用真实纯函数(通过块分数与阈值控制决策);
observability 用真实模块(counter/histogram/trace_id 零依赖)。
qa_cache/history/research_memory/wiki 钩子模块尚未重建, 走 try/except 降级
(warning 属预期诚实信号)。

切片 0: 正常路径(answered 停机、输出与 trace schema、引用管道)。
切片 1: 循环编排(max_iter cap、follow_up 驱动次轮、跨轮去重与 top_k*2 截断、
        enable_reflect=False 单轮、检索全空 no_chunks 短路)。
切片 2: abstain 集成(no_evidence 短路且 chat 不被调用、按语言路由拒答文案、
        weak 注入中/英 hint、confident 无 hint)。
切片 3: chat 失败降级(evidence-only 响应、degraded 记账)。
切片 4: prompt 组装(白名单令牌、"最多 2 个引用"、zh/en 系统与用户模板路由)。
切片 5: 外壳(trace_id 16hex、loop trace 与 latency、memory 键、指标计数)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import paper_rag.config as config
from paper_rag.observability import reset as metrics_reset
from paper_rag.observability import snapshot as metrics_snapshot
from paper_rag.rag import qa_agentic as qa
from paper_rag.rag.evidence_retrieval import RetrievalExecution

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"
_ID_C = "c5f29d4e39f6a0b1c2d3"

_ZH_MSG = "未找到相关内容。"
_EN_MSG = "No relevant content was found."


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
        llm=SimpleNamespace(temperatures=SimpleNamespace(answer=0.2)),
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
    def system(self) -> str:
        return self.calls[0]["messages"][0]["content"]

    @property
    def user(self) -> str:
        return self.calls[0]["messages"][1]["content"]


def _wire(
    monkeypatch,
    *,
    chunks_per_round: list[list[dict]],
    reflects: list[dict] | None = None,
    reply: str | Exception = "stub answer",
    intent: dict | None = None,
    **conf_over,
) -> tuple[_FakeChat, list[str]]:
    """打桩全部组件, 返回 (fake_chat, 每轮检索收到的查询列表)。"""
    _conf(monkeypatch, **conf_over)
    monkeypatch.setattr(
        qa,
        "classify",
        lambda q: intent or {"intent": "reasoning", "top_k": 4, "max_iter": 2, "rrf_k": 60},
    )
    queries: list[str] = []
    rounds = list(chunks_per_round)

    def _retrieve(query, paper_ids, top_k, **kw):
        queries.append(query)
        return rounds.pop(0) if rounds else []

    monkeypatch.setattr(qa, "_retrieve_round", _retrieve)
    refl = list(reflects or [])
    monkeypatch.setattr(
        qa,
        "reflect",
        lambda q, e: (
            refl.pop(0)
            if refl
            else {"sufficiency": "sufficient", "missing": "", "follow_up": "", "score": 0.9}
        ),
    )
    monkeypatch.setattr(
        qa, "select_evidence", lambda q, chunks, intent=None: (chunks[:2], {"strategy": "stub"})
    )
    fake = _FakeChat(reply)
    monkeypatch.setattr(qa, "chat", fake)
    return fake, queries


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics_reset()
    yield
    metrics_reset()


# ---------- 切片 0: 正常路径 ----------


def test_happy_path_answered_schema(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], reply=f"claim [chunk:{_ID_A}]")
    out = qa.answer("What is X?")

    assert out["answer"] == f"claim [chunk:{_ID_A}]"
    assert out["citations"] == [_ID_A]
    assert len(out["chunks"]) == 3
    assert len(out["evidence_chunks"]) == 2, "LLM 只看 select_evidence 的紧凑证据集"
    trace = out["trace"]
    for key in (
        "intent",
        "iters",
        "stopped_by",
        "abstain",
        "evidence_selection",
        "wiki_context",
        "trace_id",
        "loop",
        "memory_before",
        "memory",
    ):
        assert key in trace, f"trace 缺 {key}"
    assert trace["stopped_by"] == "answered"
    assert trace["abstain"]["decision"] == "confident"


def test_fabricated_citation_dropped_and_suspicious_stripped(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    reply = f"ok [chunk:{_ID_A}] fake [chunk:ffffffffffffffffffff] habit [1]."
    _wire(monkeypatch, chunks_per_round=[chunks], reply=reply)
    out = qa.answer("What is X?")
    assert out["citations"] == [_ID_A]
    assert "ffffffff" not in out["answer"]
    assert "[1]" not in out["answer"]
    assert out["suspicious_citations"]["count"] == 1


def test_citations_validated_against_evidence_chunks_not_pool(monkeypatch):
    # select_evidence 桩只放行前两块 -> 引用第三块(在池中但不在证据集)应被删。
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], reply=f"x [chunk:{_ID_C}]")
    out = qa.answer("q")
    assert out["citations"] == []
    assert _ID_C not in out["answer"]


# ---------- 切片 1: 循环编排 ----------


def test_max_iter_capped_by_config(monkeypatch):
    intent = {"intent": "explore", "top_k": 4, "max_iter": 5, "rrf_k": 60}
    insufficient = {"sufficiency": "insufficient", "missing": "m", "follow_up": "f", "score": 0.1}
    _, queries = _wire(
        monkeypatch,
        chunks_per_round=[[_chunk(_ID_A)]] * 5,
        reflects=[insufficient] * 5,
        intent=intent,
        max_inner_iters=2,
    )
    qa.answer("q")
    assert len(queries) == 2, "intent 档 5 轮被配置 cap 到 2 轮"


def test_follow_up_drives_next_round_and_pool_dedups(monkeypatch):
    intent = {"intent": "reasoning", "top_k": 2, "max_iter": 2, "rrf_k": 60}
    r1 = [_chunk(_ID_A), _chunk(_ID_B)]
    r2 = [_chunk(_ID_B), _chunk(_ID_C)]  # B 与首轮重复
    insufficient = {
        "sufficiency": "insufficient",
        "missing": "m",
        "follow_up": "follow-up query",
        "score": 0.2,
    }
    _, queries = _wire(
        monkeypatch, chunks_per_round=[r1, r2], reflects=[insufficient], intent=intent
    )
    out = qa.answer("original q")
    assert queries == ["original q", "follow-up query"]
    ids = [c["chunk_id"] for c in out["chunks"]]
    assert ids == [_ID_A, _ID_B, _ID_C], "跨轮按 chunk_id 去重且保序"


def test_final_chunks_truncated_to_twice_top_k(monkeypatch):
    intent = {"intent": "factual", "top_k": 1, "max_iter": 1, "rrf_k": 60}
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], intent=intent)
    out = qa.answer("q")
    assert len(out["chunks"]) == 2, "final_chunks 截断到 top_k*2"


def test_reflect_disabled_single_round(monkeypatch):
    called = []
    _, queries = _wire(
        monkeypatch,
        chunks_per_round=[[_chunk(_ID_A)], [_chunk(_ID_B)]],
        enable_reflect=False,
    )
    monkeypatch.setattr(qa, "reflect", lambda q, e: called.append(1))
    out = qa.answer("q")
    assert len(queries) == 1
    assert not called
    assert out["trace"]["stopped_by"] == "answered"


def test_retrieve_empty_short_circuits_no_chunks(monkeypatch):
    fake, _ = _wire(monkeypatch, chunks_per_round=[[]])
    out = qa.answer("What is X?")
    assert out["answer"] == "(no evidence found in the indexed papers)"
    assert out["citations"] == [] and out["chunks"] == []
    assert out["trace"]["stopped_by"] == "no_evidence"
    assert out["trace"]["degraded"] == "no_chunks"
    assert out["trace"]["abstain"]["decision"] == "no_chunks"
    assert not fake.calls, "无块时不得调用 LLM"


def test_no_chunks_chinese_message(monkeypatch):
    _wire(monkeypatch, chunks_per_round=[[]])
    out = qa.answer("图神经网络是什么?")
    assert out["answer"] == "(未在已索引文献中检索到证据)"


# ---------- 切片 2: abstain 集成 ----------


def test_abstain_no_evidence_skips_llm_english_message(monkeypatch):
    weak_chunks = [_chunk(_ID_A, 0.05), _chunk(_ID_B, 0.05), _chunk(_ID_C, 0.05)]
    fake, _ = _wire(monkeypatch, chunks_per_round=[weak_chunks])
    out = qa.answer("What is X?")
    assert out["answer"] == _EN_MSG, "英文问题应得英文拒答文案"
    assert not fake.calls, "no_evidence 时必须跳过 LLM"
    assert out["trace"]["stopped_by"] == "no_evidence_abstain"
    assert out["chunks"], "块仍随响应返回供检查"


def test_abstain_no_evidence_chinese_message(monkeypatch):
    weak_chunks = [_chunk(_ID_A, 0.05), _chunk(_ID_B, 0.05), _chunk(_ID_C, 0.05)]
    _wire(monkeypatch, chunks_per_round=[weak_chunks])
    out = qa.answer("图神经网络是什么?")
    assert out["answer"] == _ZH_MSG, "中文问题应得中文拒答文案"


def test_abstain_weak_injects_hint_by_language(monkeypatch):
    mid = [_chunk(_ID_A, 0.30), _chunk(_ID_B, 0.30), _chunk(_ID_C, 0.30)]
    fake, _ = _wire(monkeypatch, chunks_per_round=[mid], reply=f"x [chunk:{_ID_A}]")
    qa.answer("What is X?")
    assert "NOTE: The retrieved evidence appears WEAK" in fake.user

    fake2, _ = _wire(monkeypatch, chunks_per_round=[mid], reply=f"x [chunk:{_ID_A}]")
    qa.answer("图神经网络是什么?")
    assert "注意" in fake2.user and "证据" in fake2.user


def test_abstain_confident_no_hint(monkeypatch):
    fake, _ = _wire(monkeypatch, chunks_per_round=[[_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]])
    qa.answer("What is X?")
    assert "WEAK" not in fake.user


# ---------- 切片 3: chat 失败降级 ----------


def test_chat_failure_degrades_to_evidence_only(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], reply=RuntimeError("boom"))
    out = qa.answer("What is X?")
    assert out["answer"] == "(LLM unavailable; see chunks for evidence)"
    assert out["citations"] == []
    assert out["evidence_chunks"], "降级响应仍带证据集供人工查看"
    assert out["trace"]["degraded"] == "chat_error:RuntimeError"


# ---------- 切片 4: prompt 组装 ----------


def test_user_prompt_has_allowed_tokens_and_two_citation_cap(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    fake, _ = _wire(monkeypatch, chunks_per_round=[chunks])
    qa.answer("What is X?")
    assert f"[chunk:{_ID_A}]" in fake.user and f"[chunk:{_ID_B}]" in fake.user
    assert f"[chunk:{_ID_C}]" not in fake.user, "白名单只列证据集, 不列全池"
    assert "at most 2 citations" in fake.user
    assert "academic research assistant" in fake.system


def test_chinese_question_routes_chinese_prompts(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    fake, _ = _wire(monkeypatch, chunks_per_round=[chunks], reply=f"x [chunk:{_ID_A}]")
    qa.answer("图神经网络是什么?")
    assert "学术研究助手" in fake.system
    assert "【1】" in fake.system, "中文系统 prompt 应禁止全角引用形态"
    assert "最多使用 2 个引用" in fake.user
    assert "问题:" in fake.user


# ---------- 切片 5: 外壳 ----------


def test_trace_id_and_loop_trace(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    _wire(monkeypatch, chunks_per_round=[chunks], reply=f"x [chunk:{_ID_A}]")
    out = qa.answer("What is X?")
    tid = out["trace"]["trace_id"]
    assert len(tid) == 16 and int(tid, 16) >= 0
    loop = out["trace"]["loop"]
    assert loop["intent"] == "reasoning"
    assert loop["stopped_by"] == "answered"
    assert isinstance(loop["latency_ms"], int)
    assert loop["n_chunks"] == 3 and loop["n_evidence_chunks"] == 2
    assert loop["citations"] == [_ID_A]


def test_memory_keys_without_conversation(monkeypatch):
    _wire(monkeypatch, chunks_per_round=[[_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]])
    out = qa.answer("q")
    for key in ("memory_before", "memory"):
        mem = out["trace"][key]
        assert mem["has_compressed_memory"] is False
        assert mem["memory_role"] == "query_context_only_not_evidence"


def test_counters_incremented(monkeypatch):
    _wire(
        monkeypatch,
        chunks_per_round=[[_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]],
        reply=f"x [chunk:{_ID_A}]",
    )
    qa.answer("What is X?")
    names = {c["name"] for c in metrics_snapshot()["counters"]}
    assert "paper_rag_qa_total" in names
    assert "paper_rag_qa_abstain_total" in names
    assert "paper_rag_qa_citations_total" in names
    hist = {h["name"] for h in metrics_snapshot()["histograms"]}
    assert "paper_rag_qa_latency_seconds" in hist


def test_qa_agentic_delegates_retrieval_to_shared_domain_service(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B), _chunk(_ID_C)]
    fake_chat, _ = _wire(monkeypatch, chunks_per_round=[chunks], reply=f"x [chunk:{_ID_A}]")
    calls: list[dict] = []

    def fake_retrieve(query, **kwargs):
        calls.append({"query": query, **kwargs})
        return RetrievalExecution(
            retrieval_id="r_shared",
            public_decision="confident",
            internal_decision="confident",
            candidate_chunks=chunks,
            evidence_chunks=chunks[:2],
            wiki_entries=[],
            allowed_chunk_ids=[_ID_A, _ID_B],
            trace={
                "intent": {"intent": "reasoning", "top_k": 4, "max_iter": 2},
                "iters": [{"query": query, "n_retrieved": 3, "reflect": None}],
                "stopped_by": "answered",
                "abstain": {"decision": "confident"},
                "evidence_selection": {"strategy": "shared"},
                "wiki_context": {"entries": [], "fingerprint": ""},
            },
        )

    monkeypatch.setattr(qa, "retrieve_evidence", fake_retrieve)

    out = qa.answer("What is X?")

    assert len(calls) == 1
    assert calls[0]["query"] == "What is X?"
    assert out["trace"]["evidence_selection"] == {"strategy": "shared"}
    assert fake_chat.calls


def test_qa_cache_can_short_circuit_after_scope_validation(monkeypatch):
    _conf(monkeypatch)
    monkeypatch.setattr(
        qa,
        "_resolve_wiki_context_safe",
        lambda question, paper_ids: {"entries": [], "fingerprint": ""},
    )
    monkeypatch.setattr(
        qa,
        "_check_cache",
        lambda question, paper_ids, trace_id: {
            "answer": "cached",
            "citations": [],
            "chunks": [],
            "suspicious_citations": {"numeric": [], "author_year": [], "count": 0},
            "trace": {
                "intent": {"intent": "factual"},
                "iters": [],
                "stopped_by": "cache_hit",
            },
        },
    )
    monkeypatch.setattr(
        qa,
        "retrieve_evidence",
        lambda *args, **kwargs: pytest.fail("cache hit must skip retrieval"),
    )

    out = qa.answer("cached question")

    assert out["answer"] == "cached"
    assert out["trace"]["stopped_by"] == "cache_hit"
