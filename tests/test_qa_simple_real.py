"""rag.qa_simple 真实 LLM 集成测试(LLM 无 mock, 真发网络)。

验收协议: 缺配置**明确失败, 不 skip**。`.env` 由 `tests/conftest.py` 统一加载。

检索层按开题确认采用**数据注入**(monkeypatch retrieve 返回手工构造的真实
形态 chunks, 非行为 mock)——检索本身已由 P6 各课与 scripts/demo_qa_simple.py
的真实链路验收覆盖; 本测试专打 LLM 边界的引用纪律:
- 英文: 只引给定 id、无数字/作者-年份形态残留、答案带引用令牌;
- 中文: 中文 prompt 路由下同样纪律, 且答案为中文。
"""

from __future__ import annotations

import os

import pytest

import paper_rag.config as config
from paper_rag.rag import llm
from paper_rag.rag import qa_simple as qs
from paper_rag.rag.citation_check import detect_suspicious_citations

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"


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


def _chunk(cid: str, text: str, section: str = "Method") -> dict:
    return {
        "chunk_id": cid,
        "paper_id": "p-real",
        "section": section,
        "modality": "text",
        "score": 0.9,
        "text": text,
    }


def _assert_discipline(out: dict, allowed: set[str]) -> None:
    assert out["citations"], "强证据问题引用不应为空"
    assert set(out["citations"]) <= allowed
    assert detect_suspicious_citations(out["answer"])["count"] == 0
    assert "[chunk:" in out["answer"]


def test_real_english_answer_citation_discipline(monkeypatch):
    _require_llm_env()
    chunks = [
        _chunk(
            _ID_A,
            "Graph-Mamba extends the Mamba selective state space model to graphs; its node "
            "prioritization lets the recurrent state retain information from distant nodes, "
            "capturing long-range dependencies with linear complexity.",
        ),
        _chunk(
            _ID_B,
            "On long-range graph benchmarks Graph-Mamba outperforms attention baselines "
            "while using fewer FLOPs.",
            section="Experiments",
        ),
    ]
    monkeypatch.setattr(qs, "retrieve", lambda q, top_k=8, paper_ids=None: chunks)
    out = qs.answer("How does Graph-Mamba handle long-range dependencies?")
    _assert_discipline(out, {_ID_A, _ID_B})


def test_real_chinese_answer_citation_discipline(monkeypatch):
    _require_llm_env()
    chunks = [
        _chunk(
            _ID_A,
            "面向综合能源服务的区块链网络架构自下而上分为数据层、网络层、共识层、"
            "合约层与应用层; 共识层采用实用拜占庭容错(PBFT)算法, 以联盟链形式控制节点准入。",
            section="方法",
        ),
    ]
    monkeypatch.setattr(qs, "retrieve", lambda q, top_k=8, paper_ids=None: chunks)
    out = qs.answer("综合能源服务里区块链的网络架构是怎样设计的")
    _assert_discipline(out, {_ID_A})
    assert any("一" <= ch <= "鿿" for ch in out["answer"]), "中文问题应得中文答案"
