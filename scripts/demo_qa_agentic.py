"""rag/qa_agentic.py 真实链路验收: P7 全组件总合龙。

数据与 qa_simple Demo 同源, 真实 BGE-M3 + embedded Qdrant + FTS5 + reranker +
真实 Qwen(intent 分类/reflect/作答全真实)。

验收点:
- [1] 英文域内: answered 停机、abstain confident、citations 非空且 ⊆ 证据集、
      suspicious=0、trace 完整(intent/iters/abstain/evidence_selection/loop);
- [2] 中文域内: 中文 prompt 路由下中文答案, 同样引用纪律;
- [3] 域外"上海明天的天气": **abstain 在 LLM 作答前短路**——作答 chat 调用
      计数为 0、answer == 配置拒答文案(中文)、stopped_by=no_evidence_abstain。
      与 qa_simple 域外实证(声明不足仍引 3 块)正面对照: agentic 链路里
      引用数从 3 归零;
- [4] 打印 observability render() 的 Prometheus 指标文本(qa_total/abstain/
      latency 直方图), 证明指标链路上线。

临时数据隔离在 demo-qa-agentic-data/; 任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-qa-agentic-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def _assert_answered(out: dict, label: str) -> None:
    from paper_rag.rag.citation_check import detect_suspicious_citations

    evidence_ids = {c["chunk_id"] for c in out["evidence_chunks"]}
    assert out["citations"], f"{label}: 引用不应为空"
    assert set(out["citations"]) <= evidence_ids, f"{label}: 引用了证据集之外的 id"
    assert detect_suspicious_citations(out["answer"])["count"] == 0
    trace = out["trace"]
    assert trace["stopped_by"] == "answered"
    assert trace["abstain"]["decision"] == "confident"
    assert trace["loop"]["latency_ms"] > 0
    assert trace["evidence_selection"]["candidates"], f"{label}: 选证记账缺失"


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    if not (SRC_DATA / "qdrant").is_dir() or not ZH_CHUNKS.is_file():
        print("缺少存量产物(demo-ingest-pipeline-data 或中文期刊 chunks.json)", file=sys.stderr)
        raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_DATA / "data/index/papers.sqlite", DEMO_ROOT / "papers.sqlite")
    shutil.copytree(SRC_DATA / "qdrant", DEMO_ROOT / "qdrant")

    conf = _point_config_at_demo_data()
    assert conf.llm.base_url and conf.llm.api_key and conf.llm.chat_model, (
        "qa_agentic 真实验收需要 .env 提供 OPENAI_BASE_URL/OPENAI_API_KEY/CHAT_MODEL"
    )

    from paper_rag.embed import bge_m3
    from paper_rag.observability import render
    from paper_rag.rag import qa_agentic
    from paper_rag.store import qdrant_store, sqlite_store

    # 只统计"作答"chat(intent/reflect 各有自己的 chat 引用, 不经过这里)
    answer_calls = {"n": 0}
    real_chat = qa_agentic.chat

    def counting_chat(*args, **kwargs):
        answer_calls["n"] += 1
        return real_chat(*args, **kwargs)

    qa_agentic.chat = counting_chat

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    sqlite_store.upsert_sections_and_chunks(
        zh_chunks[0]["paper_id"], zh_payload["sections"], zh_chunks
    )
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(
        f"[0/4] 双语双库就绪: 英文库副本 + 中文期刊 {len(zh_chunks)} 块; "
        f"模型 {conf.llm.chat_model}\n"
    )

    # ── 1) 英文域内: 全链 answered ──
    out_en = qa_agentic.answer("How does Graph-Mamba handle long-range dependencies?")
    _assert_answered(out_en, "英文")
    t = out_en["trace"]
    print(
        f"[1/4] 英文: intent={t['intent']['intent']} iters={len(t['iters'])} "
        f"abstain={t['abstain']['decision']}({t['abstain']['evidence_score']:.3f}) "
        f"citations={len(out_en['citations'])} latency={t['loop']['latency_ms']}ms\n"
        f"      {out_en['answer'][:140]}…\n"
    )

    # ── 2) 中文域内: 中文路由 ──
    out_zh = qa_agentic.answer("综合能源服务里区块链的网络架构是怎样设计的")
    _assert_answered(out_zh, "中文")
    assert any("一" <= ch <= "鿿" for ch in out_zh["answer"])
    t = out_zh["trace"]
    print(
        f"[2/4] 中文: intent={t['intent']['intent']} iters={len(t['iters'])} "
        f"abstain={t['abstain']['decision']}({t['abstain']['evidence_score']:.3f}) "
        f"citations={len(out_zh['citations'])}\n"
        f"      {out_zh['answer'][:140]}…\n"
    )

    # ── 3) 域外: abstain 在作答前短路(对照 qa_simple 的 citations=3) ──
    calls_before = answer_calls["n"]
    out_ood = qa_agentic.answer("上海明天的天气怎么样")
    t = out_ood["trace"]
    assert answer_calls["n"] == calls_before, "no_evidence 时作答 LLM 不得被调用"
    assert t["stopped_by"] == "no_evidence_abstain"
    assert out_ood["answer"] == conf.rag.abstain.no_evidence_message, "中文问题得中文拒答文案"
    assert out_ood["citations"] == []
    print(
        f"[3/4] 域外: abstain={t['abstain']['decision']}"
        f"({t['abstain']['evidence_score']:.4f}) -> 作答 LLM 被跳过, citations=0 "
        f"(qa_simple 同题引用 3 个噪声块, agentic 链路归零)\n"
        f"      拒答文案: {out_ood['answer'][:50]}…\n"
    )

    # ── 4) 指标链路 ──
    metrics = render()
    assert "paper_rag_qa_total" in metrics
    assert 'paper_rag_qa_abstain_total{decision="no_evidence"}' in metrics
    assert "paper_rag_qa_latency_seconds_count" in metrics
    qa_lines = [
        ln
        for ln in metrics.splitlines()
        if ln.startswith(("paper_rag_qa_total", "paper_rag_qa_abstain_total"))
    ]
    print("[4/4] Prometheus 指标(节选):")
    for ln in qa_lines:
        print(f"      {ln}")

    print("\nDEMO PASSED: 英文/中文全链 answered + 域外 abstain 短路 + 指标链路 全部通过")


if __name__ == "__main__":
    main()
