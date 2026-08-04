"""hybrid RRF 融合真实验收: 真双语双库语料, 双腿互补性的实证。

数据(只读复制): 英文 Graph-Mamba 库(SQLite + embedded Qdrant 一并复制到
demo-hybrid-data/)+ 中文期刊 62 chunks——本 Demo 把中文期刊真实嵌入
(BGE-M3 编码 context_text 写入 Qdrant 副本), 构成两条腿都覆盖双语的语料。

验收点:
- 精确词查询(GMB): 稀疏腿拉抬, 融合 top-3 字面含 GMB;
- 跨语言改述查询(中文问英文论文): 稀疏腿 0 命中、dense 腿命中, 融合不空
  ——两腿失败模式互补的直接证据;
- 双腿 top-10 重叠度实测(dense 课欠的账: 低重叠正是 hybrid 存在的理由);
- 中文查询融合命中中文期刊, 结果带 score_rrf(降序)与 score_dense(abstain 伏笔);
- 注入 fts5 故障 -> warning + rank_bm25 接管, 融合仍可用。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-hybrid-data"
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
    from paper_rag.retrieve import dense, fts5, hybrid
    from paper_rag.store import qdrant_store, sqlite_store

    # --- 0. 中文期刊真实入双库: SQLite + BGE-M3 嵌入 -> Qdrant 副本 ---
    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)
    vectors = bge_m3.encode([c["context_text"] for c in zh_chunks])
    n_up = qdrant_store.upsert_chunks(zh_chunks, vectors)
    assert n_up == len(zh_chunks)
    print(f"[0/5] 中文期刊已入双库: SQLite {len(zh_chunks)} 块, Qdrant 新增 {n_up} 向量")

    # --- 1. 精确词查询: 稀疏腿拉抬 ---
    fused = hybrid.hybrid_search("GMB block")
    assert fused, "融合结果不应为空"
    assert any("GMB" in h.get("text", "") for h in fused[:3]), "top-3 应有字面含 GMB 的块"
    assert all(h["paper_id"] == EN_PAPER for h in fused[:3])
    print(
        f"[1/5] 'GMB block': 融合 {len(fused)} 块, top-3 含字面 GMB, top-1 score_rrf={fused[0]['score_rrf']:.4f}"
    )

    # --- 2. 跨语言改述: 两腿失败模式互补的直接证据 ---
    q_zh_about_en = "图上相距很远的节点如何高效交换信息"
    sparse_leg = fts5.search(q_zh_about_en, top_k=20, paper_ids=[EN_PAPER])
    dense_leg = dense.retrieve(q_zh_about_en, top_k=20, paper_ids=[EN_PAPER])
    fused = hybrid.hybrid_search(q_zh_about_en, paper_ids=[EN_PAPER])
    assert sparse_leg == [], "中文 bigram 在纯英文论文里应 0 命中"
    assert dense_leg and fused, "dense 跨语言应命中, 融合不空"
    assert fused[0]["chunk_id"] == dense_leg[0]["chunk_id"]  # 稀疏腿空时融合序=dense 序
    print(
        f"[2/5] 跨语言改述: sparse=0 / dense={len(dense_leg)} / 融合={len(fused)}, "
        f"top-1 section={fused[0].get('section', '')!r} (dense 独腿救场)"
    )

    # --- 3. 双腿重叠度实测(dense 课欠的账) ---
    q = "long-range dependencies in graph sequence modeling"
    d10 = {h["chunk_id"] for h in dense.retrieve(q, top_k=10)}
    s10 = {h["chunk_id"] for h in fts5.search(q, top_k=10)}
    overlap = len(d10 & s10)
    assert overlap < 10, "双腿完全重合的话 hybrid 就没有存在意义了"
    print(f"[3/5] 双腿 top-10 重叠 {overlap}/10 —— 失败模式不重叠, 融合有增益空间")

    # --- 4. 中文查询融合 + 下游契约字段 ---
    fused = hybrid.hybrid_search("综合能源服务区块链的网络架构")
    assert fused and fused[0]["paper_id"] == zh_paper
    rrf_scores = [h["score_rrf"] for h in fused]
    assert rrf_scores == sorted(rrf_scores, reverse=True)
    assert any("score_dense" in h for h in fused), "abstain 需要的绝对相似度信号应保留"
    print(
        f"[4/5] 中文融合: top-1 命中中文期刊 section={fused[0].get('section', '')[:24]!r}, "
        f"score_rrf 降序 + score_dense 保留"
    )

    # --- 5. 注入 fts5 故障: 回退 rank_bm25, 融合仍可用 ---
    def _outage(*args, **kwargs):
        raise RuntimeError("injected fts5 outage")

    original = fts5.search
    fts5.search = _outage
    try:
        fused = hybrid.hybrid_search("区块链")
    finally:
        fts5.search = original
    assert fused and fused[0]["paper_id"] == zh_paper, "回退 bm25 后中文融合仍应命中"
    print(f"[5/5] 注入 fts5 故障: warning 回退 rank_bm25, '区块链' 融合仍命中 {len(fused)} 块")

    print()
    print("hybrid RRF 融合真实验收通过: 双腿互补实证、双语融合、故障回退全部符合预期。")


if __name__ == "__main__":
    main()
