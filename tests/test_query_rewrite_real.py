"""rag/query_rewrite.py 真实集成测试: 无 mock, 真发 LLM 调用。

与 tests/test_query_rewrite.py(全打桩边界测试)互补: 这里验证真实模型确实
按模板产出可解析的 JSON, 且中文问题的 keywords 真的中英双语混出——这一点
只有真实模型能证明, 打桩测试只能证明"如果模型这么答, 我们解析对了"。

验收协议: 缺配置明确失败, 不 skip。仓库根的 `.env` 由 `tests/conftest.py`
在收集测试前统一读入(不覆盖已导出的环境变量), 无需手工 `set -a; . ./.env`。
"""

from __future__ import annotations

import os
import re

import pytest

import paper_rag.config as config
from paper_rag.rag import llm
from paper_rag.rag import query_rewrite as qr

_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿]")


@pytest.fixture(autouse=True)
def _clean_state():
    config.load.cache_clear()
    llm.reset_client_for_test()
    os.environ.pop("PAPER_RAG_FORCE_LOCAL_REWRITE", None)
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


def test_real_english_question_produces_variants_and_hyde():
    _require_llm_env()
    q = "What are the differences between RAG-Sequence and RAG-Token?"

    out = qr.rewrite(q)

    print(f"\n[real][en] dense_queries={out['dense_queries']}\n[real][en] bm25={out['bm25_query']}")
    assert out["dense_queries"][0] == q, "首项必须是原问题"
    assert len(out["dense_queries"]) > 1, "真实模型未产出任何改写/HyDE"
    assert out["bm25_query"].strip(), "bm25_query 不应为空"
    # 英文路径不应混入中文模板产物之外的东西: raw 至少解析出一个已知键。
    assert any(k in out["raw"] for k in ("variants", "keywords", "hyde")), (
        f"未从真实回复解析出任何已知键: raw={out['raw']}"
    )


def test_real_chinese_question_yields_bilingual_keywords():
    """中文提问时 keywords 必须中英混出——否则 BM25 永远打不中英文论文块。"""
    _require_llm_env()
    q = "自我反思式检索增强生成是怎么工作的?"

    out = qr.rewrite(q)

    bm25 = out["bm25_query"]
    print(f"\n[real][zh] dense_queries={out['dense_queries']}\n[real][zh] bm25={bm25}")
    assert out["dense_queries"][0] == q, "首项必须是原问题"
    assert len(out["dense_queries"]) > 1, "中文问题未产出任何改写/HyDE"
    assert _CJK.search(bm25), f"keywords 缺中文术语: {bm25}"
    assert _LATIN.search(bm25), f"keywords 缺英文术语, 跨语言 BM25 断层未跨越: {bm25}"


def test_real_escape_hatch_skips_llm_entirely(monkeypatch):
    """逃生门置真时一次 LLM 都不发——真实环境下也成立(省钱/离线可用)。"""
    _require_llm_env()
    calls = {"n": 0}
    real_chat = qr.chat

    def _counting(*a, **kw):
        calls["n"] += 1
        return real_chat(*a, **kw)

    monkeypatch.setattr(qr, "chat", _counting)
    monkeypatch.setenv("PAPER_RAG_FORCE_LOCAL_REWRITE", "1")

    out = qr.rewrite("检索增强生成怎么缓解幻觉?")

    assert calls["n"] == 0, "逃生门置真时仍发起了真实 LLM 调用"
    assert out["dense_queries"][0] == "检索增强生成怎么缓解幻觉?"
