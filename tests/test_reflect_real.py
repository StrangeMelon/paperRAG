"""rag.reflect 真实 LLM 集成测试(无 mock, 真发网络)。

验收协议: 缺配置**明确失败, 不 skip**——真实验收不允许静默跳过。
`.env` 由 `tests/conftest.py` 统一加载。

只打真实 LLM 边界(证据文本手工构造, 不依赖 Qdrant/嵌入):
- 英文强证据 -> sufficient;
- 证据与问题无关 -> 非 sufficient 且给出 follow_up;
- 中文问题 + 中文证据 -> 中文模板下判充分, 四键契约与 score 域成立。
"""

from __future__ import annotations

import os

import pytest

import paper_rag.config as config
from paper_rag.rag import llm
from paper_rag.rag.reflect import reflect


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


def _assert_contract(r: dict) -> None:
    assert set(r.keys()) == {"sufficiency", "missing", "follow_up", "score"}
    assert r["sufficiency"] in ("sufficient", "partial", "insufficient")
    assert isinstance(r["missing"], str) and isinstance(r["follow_up"], str)
    assert 0.0 <= r["score"] <= 1.0


def test_real_strong_evidence_is_sufficient():
    _require_llm_env()
    evidence = (
        "[chunk:gm-01] Graph-Mamba extends the Mamba selective state space model to "
        "graph-structured data. Its node prioritization and permutation strategies let the "
        "recurrent state selectively retain information from distant nodes, capturing "
        "long-range dependencies with linear complexity in sequence length.\n"
        "[chunk:gm-02] Experiments on long-range graph benchmarks show Graph-Mamba "
        "outperforms attention-based baselines while using fewer FLOPs, confirming that the "
        "selective state space handles long-range interactions effectively."
    )
    r = reflect("How does Graph-Mamba handle long-range dependencies?", evidence)
    _assert_contract(r)
    assert r["sufficiency"] == "sufficient"


def test_real_irrelevant_evidence_is_not_sufficient():
    _require_llm_env()
    evidence = (
        "[chunk:bc-01] 区块链在综合能源服务中的网络架构分为数据层、共识层与应用层, "
        "通过智能合约协调多主体间的能源交易。"
    )
    r = reflect(
        "How does Graph-Mamba compare with vanilla Transformers on ImageNet accuracy?",
        evidence,
    )
    _assert_contract(r)
    assert r["sufficiency"] != "sufficient"
    assert r["follow_up"], "不充分时应给出下一轮检索方向"


def test_real_chinese_question_uses_zh_template_and_passes():
    _require_llm_env()
    evidence = (
        "[chunk:zh-01] 面向综合能源服务的区块链网络架构自下而上分为数据层、网络层、"
        "共识层、合约层与应用层: 数据层存储能源交易账本, 网络层采用 P2P 组网, "
        "共识层使用实用拜占庭容错(PBFT)算法, 合约层承载能源计量与结算的智能合约, "
        "应用层对接源网荷储各参与方。\n"
        "[chunk:zh-02] 该架构通过分层解耦支持多能源主体接入, 并以联盟链形式控制"
        "节点准入, 兼顾交易吞吐与监管合规。"
    )
    r = reflect("综合能源服务里区块链的网络架构是怎样设计的", evidence)
    _assert_contract(r)
    assert r["sufficiency"] == "sufficient"
