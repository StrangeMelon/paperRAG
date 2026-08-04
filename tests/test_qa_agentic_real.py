"""rag.qa_agentic 真实 LLM 集成测试(intent/reflect/作答 LLM 全真实, 无 mock)。

验收协议: 缺配置**明确失败, 不 skip**。`.env` 由 `tests/conftest.py` 统一加载。

检索层按 qa_simple 先例**数据注入**(monkeypatch `_retrieve_round` 返回手工
真实形态 chunks; 检索本身由 P6 与 demo_qa_agentic.py 真实覆盖):
- 英文强证据: answered + confident + 引用纪律;
- 低分噪声块: abstain 短路, 作答 LLM 零调用, 拒答文案按语言路由。
"""

from __future__ import annotations

import os

import pytest

import paper_rag.config as config
from paper_rag.rag import llm
from paper_rag.rag import qa_agentic as qa
from paper_rag.rag.citation_check import detect_suspicious_citations

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"
_ID_C = "c5f29d4e39f6a0b1c2d3"


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


def test_real_english_full_loop_answered(monkeypatch):
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
            "On long-range graph benchmarks Graph-Mamba outperforms attention baselines "
            "while using fewer FLOPs.",
        ),
        _chunk(_ID_C, "Ablations show node filtering is the key contributor.", 0.8),
    ]
    monkeypatch.setattr(qa, "_retrieve_round", lambda q, p, k, **kw: chunks)
    out = qa.answer("How does Graph-Mamba handle long-range dependencies?")

    assert out["trace"]["stopped_by"] == "answered"
    assert out["trace"]["abstain"]["decision"] == "confident"
    assert out["citations"], "强证据问题引用不应为空"
    evidence_ids = {c["chunk_id"] for c in out["evidence_chunks"]}
    assert set(out["citations"]) <= evidence_ids
    assert detect_suspicious_citations(out["answer"])["count"] == 0


def test_real_low_score_noise_abstains_without_answer_llm(monkeypatch):
    _require_llm_env()
    noise = [
        _chunk(_ID_A, "区块链共识层采用 PBFT 算法。", 0.02),
        _chunk(_ID_B, "联盟链控制节点准入。", 0.02),
        _chunk(_ID_C, "智能合约承载能源结算。", 0.02),
    ]
    monkeypatch.setattr(qa, "_retrieve_round", lambda q, p, k, **kw: noise)
    calls = {"n": 0}
    real_chat = qa.chat

    def counting_chat(*args, **kwargs):
        calls["n"] += 1
        return real_chat(*args, **kwargs)

    monkeypatch.setattr(qa, "chat", counting_chat)
    out = qa.answer("上海明天的天气怎么样")

    assert out["trace"]["stopped_by"] == "no_evidence_abstain"
    assert calls["n"] == 0, "abstain 短路后作答 LLM 不得被调用"
    assert out["answer"] == config.load().rag.abstain.no_evidence_message
    assert out["citations"] == []
