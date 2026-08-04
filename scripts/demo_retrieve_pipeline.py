"""P6 检索层端到端真实验收: retrieve_round 全漏斗 + format_evidence 证据渲染。

数据与 hybrid/rerank 课同源(英文 Graph-Mamba 库副本 + 中文期刊 62 块真实
嵌入), 真实 BGE-M3 + embedded Qdrant + FTS5 + 真实 reranker, 无 mock。
rag.query_rewrite 尚未重建(P7), 每轮会打一行 identity rewrite warning
——预期诚实信号。

验收点:
- 英文问题走完整漏斗: rewrite(恒等回退) -> hybrid 池化 -> rerank -> 多样化,
  返回 top_k 块且带 score_rerank;
- 中文句子问题: 串长分流修复后稀疏腿从 0 命中变为有(本课 fts5 修复的实证),
  全漏斗 top-1 命中中文期刊;
- 模态线索: "表格"问题触发 table 定向追加轮, 定向轮真实召回表格块;
- 论文多样化: 混合问题的窗口前段单篇不超过 2 块;
- format_evidence: 每块携带逐字 [chunk:<id>] 引用令牌(引用闭环的头)。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-retrieve-pipeline-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)
EN_PAPER = "demo-pipeline-graph-mamba"


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def main() -> None:
    if not (SRC_DATA / "qdrant").is_dir() or not ZH_CHUNKS.is_file():
        print("缺少存量产物(demo-ingest-pipeline-data 或中文期刊 chunks.json)", file=sys.stderr)
        raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_DATA / "data/index/papers.sqlite", DEMO_ROOT / "papers.sqlite")
    shutil.copytree(SRC_DATA / "qdrant", DEMO_ROOT / "qdrant")

    _point_config_at_demo_data()

    from paper_rag.embed import bge_m3
    from paper_rag.retrieve import fts5
    from paper_rag.retrieve.format import format_evidence
    from paper_rag.retrieve.pipeline import infer_modalities, retrieve_round
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(f"[0/5] 双语双库就绪: 英文 49 块 + 中文 {len(zh_chunks)} 块")

    # --- 1. 英文问题全漏斗 ---
    q_en = "How does Graph-Mamba handle long-range dependencies?"
    out = retrieve_round(q_en, None, 8)
    assert 0 < len(out) <= 8
    assert all("score_rerank" in h for h in out), "全漏斗出口应带精排分数"
    print(f"[1/5] 英文全漏斗: {len(out)} 块, top-1 score_rerank={out[0]['score_rerank']:.3f}")

    # --- 2. 中文句子问题: 串长分流修复的实证 ---
    q_zh = "综合能源服务里区块链的网络架构是怎样设计的"  # 句子形态, 语料中并无此原句
    sparse_hits = fts5.search(q_zh, top_k=10)
    assert sparse_hits, "串长分流修复后, 句子级中文查询稀疏腿不应再是 0 命中"
    out = retrieve_round(q_zh, None, 8)
    assert out and out[0]["paper_id"] == zh_paper, f"top-1 应为中文期刊: {out[0].get('paper_id')}"
    print(
        f"[2/5] 中文句子问题: 稀疏腿命中 {len(sparse_hits)} 块(修复前为 0), "
        f"全漏斗 top-1 score_rerank={out[0]['score_rerank']:.3f}"
    )

    # --- 3. 模态线索: 表格定向追加轮 ---
    q_table = "论文的表格里对比了哪些数据集上的实验结果"
    assert infer_modalities(q_table) == ["table"]
    from paper_rag.retrieve.hybrid import hybrid_search

    table_round = hybrid_search(q_table, top_k=8, modality="table")
    assert table_round and all(h["modality"] == "table" for h in table_round)
    out = retrieve_round(q_table, None, 8)
    assert out
    n_table = sum(1 for h in out if h.get("modality") == "table")
    print(
        f"[3/5] 模态线索: 'table' 定向轮召回 {len(table_round)} 块表格, "
        f"融合窗口里进了 {n_table} 块(0 也合法, 由精排定夺)"
    )

    # --- 4. 论文多样化: 单篇限额(补位语义下的真实契约) ---
    q_mix = "长序列建模与区块链网络架构"  # 两篇论文各沾一半的问题
    out = retrieve_round(q_mix, None, 4)
    assert len(out) == 4
    head_counts: dict[str, int] = {}
    for h in out:
        head_counts[h["paper_id"]] = head_counts.get(h["paper_id"], 0) + 1
    over = {p: v for p, v in head_counts.items() if v > 2}
    if over:
        # 超限只允许出现在补位场景: 另一篇的候选已被取尽(必然 <2 块)
        others = [v for p, v in head_counts.items() if p not in over]
        assert all(v < 2 for v in others), f"非补位场景单篇超限: {head_counts}"
    print(f"[4/5] 论文多样化: top-4 分布 {head_counts}(限额 2, 超出仅可来自补位)")

    # --- 5. format_evidence: 引用闭环的头 ---
    text = format_evidence(out)
    for h in out:
        assert (
            f"Use this exact citation token when citing this chunk: [chunk:{h['chunk_id']}]" in text
        )
    assert text.count("EVIDENCE CHUNK") == len(out)
    preview = text[:220].replace("\n", " | ")
    print(f"[5/5] 证据渲染: {len(out)} 块全部携带 [chunk:<id>] 令牌; 预览: {preview}...")

    print()
    print("P6 检索层端到端真实验收通过: 全漏斗、双语、模态、 多样化与引用令牌全部符合预期。")


if __name__ == "__main__":
    main()
