"""rag.qa_stream 真实流式 LLM 集成测试(intent/reflect/流式作答全真实)。

验收协议: 缺配置**明确失败, 不 skip**。`.env` 由 `tests/conftest.py` 统一加载。

检索层按先例**数据注入**(monkeypatch `_retrieve_round` 返回手工真实形态
chunks + 改写载荷):
- 英文强证据: 事件序列合法、answer_chunk > 1(真流式)、引用纪律;
- 低分噪声: abstain 短路、流式 LLM 零调用、拒答文案经 answer_chunk 流出。
"""

from __future__ import annotations

import os

import pytest

import paper_rag.config as config
from paper_rag.rag import llm
from paper_rag.rag import qa_stream as qs
from paper_rag.rag.citation_check import detect_suspicious_citations

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"
_ID_C = "c5f29d4e39f6a0b1c2d3"

_RW = {"dense_queries": ["q"], "bm25_query": "kw"}


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
        pytest.fail(f"真实验收缺少环境变量: {missing}; 请配置 .env 后重跑, 不允许 skip")


def _chunk(cid: str, text: str, score: float = 0.9) -> dict:
    return {
        "chunk_id": cid,
        "paper_id": "p-real",
        "section": "Method",
        "modality": "text",
        "score_rerank": score,
        "text": text,
    }


def test_real_streaming_full_loop(monkeypatch):
    _require_llm_env()
    chunks = [
        _chunk(
            _ID_A,
            "Graph-Mamba extends the Mamba selective state space model to graphs; node "
            "prioritization lets the recurrent state retain information from distant nodes, "
            "capturing long-range dependencies with linear complexity.",
        ),
        _chunk(
            _ID_B,
            "On long-range graph benchmarks Graph-Mamba outperforms attention baselines.",
        ),
        _chunk(_ID_C, "Ablations show node filtering is the key contributor.", 0.8),
    ]
    monkeypatch.setattr(qs, "_retrieve_round", lambda q, p, k: (chunks, _RW))
    events = list(qs.stream_answer("How does Graph-Mamba handle long-range dependencies?"))

    assert events[0]["event"] == "intent"
    assert events[-1]["event"] == "done"
    n_answer_chunks = sum(1 for e in events if e["event"] == "answer_chunk")
    assert n_answer_chunks > 1, "真实流式应产生多个 answer_chunk"
    done = events[-1]["data"]
    assert done["citations"], "强证据问题引用不应为空"
    evidence_ids = {c["chunk_id"] for c in done["evidence_chunks"]}
    assert set(done["citations"]) <= evidence_ids
    assert detect_suspicious_citations(done["answer"])["count"] == 0


def test_real_noise_abstains_without_stream_llm(monkeypatch):
    _require_llm_env()
    noise = [
        _chunk(_ID_A, "区块链共识层采用 PBFT 算法。", 0.02),
        _chunk(_ID_B, "联盟链控制节点准入。", 0.02),
        _chunk(_ID_C, "智能合约承载能源结算。", 0.02),
    ]
    monkeypatch.setattr(qs, "_retrieve_round", lambda q, p, k: (noise, _RW))
    calls = {"n": 0}
    real_stream = qs._stream_chat

    def counting_stream(system, user):
        calls["n"] += 1
        yield from real_stream(system, user)

    monkeypatch.setattr(qs, "_stream_chat", counting_stream)
    events = list(qs.stream_answer("上海明天的天气怎么样"))

    assert calls["n"] == 0, "abstain 短路后流式 LLM 不得被调用"
    abstain = next(e for e in events if e["event"] == "abstain")
    assert abstain["data"]["decision"] == "no_evidence"
    done = events[-1]["data"]
    assert done["answer"] == config.load().rag.abstain.no_evidence_message
    assert done["citations"] == []
