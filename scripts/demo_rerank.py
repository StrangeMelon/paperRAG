"""BGE reranker 真实验收: 真双语双库候选精排, cross-encoder 纠偏实证。

数据与 hybrid 课同源(英文库副本 + 中文期刊 62 块真实嵌入), 本 Demo 自建
demo-rerank-data/ 副本, 不依赖其他 Demo 的运行残留。真实加载
BAAI/bge-reranker-v2-m3(约 2.3G, 已预缓存于 data/index/models)。

验收点:
- 英文查询: hybrid 16 候选 -> 精排 8, score_rerank 降序, 与 RRF 序对比展示
  名次变化(cross-encoder 逐词交互的纠偏实感);
- 中文相关性对照: 相关 zh 块得分显著高于无关 en 块(zh 零适配的实证);
- 跨语言对(中文查询-英文块): 相关块仍排在无关块之前;
- 契约: 调用方的 fused 列表不被改写;
- enabled=false: 退回 RRF 原序截断。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-rerank-data"
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

    conf = _point_config_at_demo_data()

    from paper_rag.embed import bge_m3
    from paper_rag.retrieve import hybrid, rerank
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(f"[0/5] 双语双库就绪: 英文 49 块 + 中文 {len(zh_chunks)} 块")

    # --- 1. 英文查询: RRF 序 -> 精排序 ---
    q_en = "How does Graph-Mamba select important nodes for long-range context?"
    fused = hybrid.hybrid_search(q_en)
    ranked = rerank.rerank(q_en, fused)
    assert len(fused) == 16 and len(ranked) == 8, f"漏斗形状异常: {len(fused)} -> {len(ranked)}"
    rr_scores = [h["score_rerank"] for h in ranked]
    assert rr_scores == sorted(rr_scores, reverse=True)
    moved = sum(
        1 for i, h in enumerate(ranked) if i < len(fused) and fused[i]["chunk_id"] != h["chunk_id"]
    )
    print(
        f"[1/5] 英文精排: 16 -> 8, top-1 score_rerank={rr_scores[0]:.3f}, "
        f"前 8 名中 {moved} 个名次与 RRF 序不同(cross-encoder 纠偏)"
    )

    # --- 2. 中文相关性对照: 相关 zh 块 vs 无关 en 块 ---
    q_zh = "区块链的智能合约如何支撑综合能源服务"
    zh_relevant = next(c for c in zh_chunks if "智能合约" in c["text"])
    rows = sqlite_store.list_chunks_for_papers([EN_PAPER])
    en_irrelevant = next(r for r in rows if r.modality == "text" and "Mamba" in r.text)
    pair_cands = [
        {"chunk_id": "irrelevant-en", "text": en_irrelevant.text},
        {"chunk_id": "relevant-zh", "text": zh_relevant["text"]},
    ]
    out = rerank.rerank(q_zh, pair_cands, top_k=2)
    assert out[0]["chunk_id"] == "relevant-zh", f"相关 zh 块应排第一: {out}"
    gap = out[0]["score_rerank"] - out[1]["score_rerank"]
    assert gap > 0.1, f"zh 相关性区分度不足: gap={gap:.3f}"
    print(
        f"[2/5] 中文对照: 相关块 {out[0]['score_rerank']:.3f} vs 无关块 "
        f"{out[1]['score_rerank']:.3f} (gap={gap:.3f})"
    )

    # --- 3. 跨语言对: 中文查询, 英文相关块应胜过中文无关块 ---
    # (不能拿元数据卡片当无关对照——摘要里就有 node selection, 对该问题高度相关)
    q_cross = "Graph-Mamba 如何挑选重要节点来建模长程上下文"
    gmb = next(r for r in rows if "GMB" in r.text and r.modality == "text")
    cross = rerank.rerank(
        q_cross,
        [
            {"chunk_id": "zh-irrelevant", "text": zh_relevant["text"]},  # 智能合约块
            {"chunk_id": "en-gmb", "text": gmb.text},
        ],
        top_k=2,
    )
    assert cross[0]["chunk_id"] == "en-gmb", "跨语言相关块应胜过同语言无关块"
    print(
        f"[3/5] 跨语言对: 英文 GMB 正文 {cross[0]['score_rerank']:.3f} > "
        f"中文无关块 {cross[1]['score_rerank']:.3f} (不被语言相同迷惑)"
    )

    # --- 4. 契约: 调用方 fused 不被改写 ---
    assert all("score_rerank" not in h for h in fused), "基准会原地写键+重排, 重建版不改写输入"
    print("[4/5] 输入不可变契约: fused 列表未被改写")

    # --- 5. enabled=false 退回 RRF 原序 ---
    conf.reranker.enabled = False
    passthrough = rerank.rerank(q_en, fused)
    conf.reranker.enabled = True
    assert [h["chunk_id"] for h in passthrough] == [h["chunk_id"] for h in fused[:8]]
    assert all("score_rerank" not in h for h in passthrough)
    print("[5/5] enabled=false: 退回 RRF 原序截断 top_k")

    print()
    print("reranker 真实验收通过: 双语精排、跨语言区分度、降级与不可变契约全部符合预期。")


if __name__ == "__main__":
    main()
