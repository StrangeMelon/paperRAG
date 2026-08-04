"""dense 稠密检索真实验收: 复用入库课产物 demo-ingest-pipeline-data(只读)。

前置: 已实跑过 scripts/demo_ingest_pipeline.py, 该目录内存有真实 Graph-Mamba
论文(49 chunks)的 embedded Qdrant 向量库。本脚本不写入、不清理该目录, 仅真实
加载 BGE-M3 编码查询并检索; 需确保没有其他进程占用 embedded Qdrant。

验收点:
- 真实英文问题 top-1 命中本论文正文(与入库课检索闭环同量级 score);
- paper_ids 过滤: 正确 id 命中, 虚构 id 滤空;
- modality 过滤: "metadata" 恰好只回全库唯一一张元数据卡片;
- 中文问题跨语言命中同一篇英文论文(BGE-M3 同空间, 检索层零中文适配)。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-ingest-pipeline-data"
PAPER_ID = "demo-pipeline-graph-mamba"


def _point_config_at_demo_data() -> None:
    """只读复用入库课的隔离数据; 模型缓存保持真实 data/index/models。"""
    import paper_rag.config as config

    conf = config.load()
    conf.paths.data_root = str(DEMO_ROOT / "data")
    conf.paths.sqlite_path = str(DEMO_ROOT / "data/index/papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]


def main() -> None:
    if not (DEMO_ROOT / "qdrant").is_dir():
        print(
            f"缺少入库课产物 {DEMO_ROOT.name}/qdrant, 请先跑 scripts/demo_ingest_pipeline.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _point_config_at_demo_data()

    from paper_rag.retrieve import dense

    # --- 1. 英文问题主路径 ---
    q_en = "How does Graph-Mamba capture long-range dependencies with selective state spaces?"
    hits = dense.retrieve(q_en, top_k=3)
    assert len(hits) == 3, f"top-3 应有 3 条, 实得 {len(hits)}"
    assert hits[0]["paper_id"] == PAPER_ID, f"top-1 非本论文: {hits[0].get('paper_id')}"
    assert hits[0]["score"] > 0.6, f"top-1 score 异常偏低: {hits[0]['score']:.3f}"
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"], "score 未降序"
    top_desc = [(f"{h['score']:.3f}", h.get("modality"), h.get("section", "")[:40]) for h in hits]
    print(f"[1/4] 英文问题 top-3 = {top_desc}")

    # --- 2. paper_ids 过滤: 命中与滤空 ---
    hits_in = dense.retrieve(q_en, top_k=3, paper_ids=[PAPER_ID])
    assert hits_in and hits_in[0]["paper_id"] == PAPER_ID
    hits_out = dense.retrieve(q_en, top_k=3, paper_ids=["no-such-paper"])
    assert hits_out == [], f"虚构 paper_id 应滤空, 实得 {len(hits_out)} 条"
    print(f"[2/4] paper_ids 过滤: 正确 id 命中 {len(hits_in)} 条, 虚构 id 滤空")

    # --- 3. modality 过滤: 全库唯一元数据卡片 ---
    cards = dense.retrieve("Graph-Mamba paper metadata", top_k=8, modality="metadata")
    assert len(cards) == 1, f"metadata 卡片应恰好 1 张, 实得 {len(cards)}"
    assert cards[0]["modality"] == "metadata"
    print(f"[3/4] modality='metadata' 恰好命中 1 张卡片, score={cards[0]['score']:.3f}")

    # --- 4. 中文问题跨语言检索(BGE-M3 同空间, dense 层零适配) ---
    q_zh = "Graph-Mamba 如何用选择性状态空间建模长程依赖?"
    hits_zh = dense.retrieve(q_zh, top_k=3)
    assert hits_zh and hits_zh[0]["paper_id"] == PAPER_ID
    assert hits_zh[0]["score"] > 0.4, f"跨语言 score 异常偏低: {hits_zh[0]['score']:.3f}"
    print(
        f"[4/4] 中文问题跨语言命中: top-1 score={hits_zh[0]['score']:.3f} "
        f"(英文同题 {hits[0]['score']:.3f}), section={hits_zh[0].get('section', '')[:40]}"
    )

    print()
    print("dense 稠密检索真实验收通过: 编码-检索对称闭环、双过滤、中英同空间。")


if __name__ == "__main__":
    main()
