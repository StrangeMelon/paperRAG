"""rank_bm25 后备稀疏检索真实验收: 与 FTS5 双后端对照 + 规模护栏演示。

输入(均只读): fts5 课同款真实数据——英文 Graph-Mamba 库(复制到
demo-sparse-bm25-data/ 再操作)与中文期刊(能源区块链)62 chunks。

验收点:
- 中文检索双后端对照: zh 粒度统一为 bigram 后, 两个稀疏后端对同一中文查询
  都能召回、top 结果高度重叠(FTS5 降级到 rank_bm25 时中文行为不突变);
- 后端差异如实呈现: FTS5 有 porter('prioritizes' 召回 prioritization 词族),
  rank_bm25 无词干化(同查询空结果, 换原词可命中)——降级路径的已知行为差;
- paper_ids 先全库打分后过滤;
- 规模护栏: 压低 bm25_max_chunks 后拒绝建索引(明确告警+空结果), FTS5 不受
  影响; 恢复上限后行数自愈自动重建。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-sparse-bm25-data"
SRC_SQLITE = REPO_ROOT / "demo-ingest-pipeline-data/data/index/papers.sqlite"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)
EN_PAPER = "demo-pipeline-graph-mamba"


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def main() -> None:
    for required in (SRC_SQLITE, ZH_CHUNKS):
        if not required.is_file():
            print(f"缺少存量产物: {required.relative_to(REPO_ROOT)}", file=sys.stderr)
            raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_SQLITE, DEMO_ROOT / "papers.sqlite")

    conf = _point_config_at_demo_data()

    from paper_rag.retrieve import fts5, sparse_bm25
    from paper_rag.store import sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)

    # --- 1. 中文检索双后端对照 ---
    q = "区块链"
    hits_f = fts5.search(q, top_k=10)
    hits_b = sparse_bm25.search(q, top_k=10)
    assert hits_f and hits_f[0]["paper_id"] == zh_paper, "FTS5 中文应命中"
    assert hits_b and hits_b[0]["paper_id"] == zh_paper, "rank_bm25 中文应命中"
    overlap = len({h["chunk_id"] for h in hits_f} & {h["chunk_id"] for h in hits_b})
    assert overlap >= 5, f"双后端 top-10 重叠过低: {overlap}"
    print(
        f"[1/5] 中文对照 '{q}': fts5 命中 {len(hits_f)}, bm25 命中 {len(hits_b)}, top-10 重叠 {overlap}"
    )

    # --- 2. 英文精确命中(GMB)双后端一致 ---
    gmb_f = fts5.search("GMB", top_k=5)
    gmb_b = sparse_bm25.search("GMB", top_k=5)
    assert gmb_f and gmb_b
    assert all(h["paper_id"] == EN_PAPER for h in gmb_f + gmb_b)
    with_section = [h for h in gmb_b if h["section"]]
    assert with_section, "payload 应带真实 section(基准硬编码 None)"
    print(
        f"[2/5] 'GMB': fts5 {len(gmb_f)} 块, bm25 {len(gmb_b)} 块, "
        f"bm25 命中块 section 示例={with_section[0]['section'][:30]!r}"
    )

    # --- 3. 后端差异如实呈现: porter 只在 FTS5 侧 ---
    # 词对经真实语料预核对: 'prioritizes' 精确词形 0 块, 同词干 'prioritization' 17 块
    # ('dependency' 不能用——语料里真实存在 2 块单数词形, 两后端都会命中)
    pri_f = fts5.search("prioritizes", top_k=5)
    pri_b = sparse_bm25.search("prioritizes", top_k=5)
    pri_b_exact = sparse_bm25.search("prioritization", top_k=5)
    assert pri_f, "FTS5 有 porter, prioritizes 应召回 prioritization 词族"
    assert not pri_b, "rank_bm25 无词干化, prioritizes 应空(已知行为差, 如实记账)"
    assert pri_b_exact, "换原词 prioritization 后 rank_bm25 应命中"
    print(
        f"[3/5] porter 差异: 'prioritizes' fts5={len(pri_f)} 块 / bm25=0 块; "
        f"'prioritization' bm25={len(pri_b_exact)} 块"
    )

    # --- 4. paper_ids 先打分后过滤 ---
    only_zh = sparse_bm25.search("网络架构", top_k=10, paper_ids=[zh_paper])
    assert only_zh and all(h["paper_id"] == zh_paper for h in only_zh)
    assert sparse_bm25.search("区块链", top_k=10, paper_ids=["no-such-paper"]) == []
    print(f"[4/5] paper_ids 过滤: 命中 {len(only_zh)} 块 / 虚构 id 滤空")

    # --- 5. 规模护栏与自愈恢复 ---
    total = len(sparse_bm25.build_index().chunk_ids)
    conf.retrieve.bm25_max_chunks = 50  # 压到语料量(111)之下
    sparse_bm25.invalidate()
    guarded = sparse_bm25.search("区块链", top_k=5)
    assert guarded == [], "超限时应拒绝并返回空"
    assert fts5.search("区块链", top_k=5), "护栏只影响 rank_bm25, FTS5 应正常"
    conf.retrieve.bm25_max_chunks = 200000
    recovered = sparse_bm25.search("区块链", top_k=5)  # 行数自愈: 0 != 111 -> 重建
    assert recovered and recovered[0]["paper_id"] == zh_paper
    print(
        f"[5/5] 规模护栏: 上限 50 < 语料 {total} 时 bm25 拒绝(fts5 正常); 恢复上限后自愈重建可检索"
    )

    print()
    print("rank_bm25 后备稀疏检索真实验收通过: zh bigram 双后端一致, 护栏与自愈符合预期。")


if __name__ == "__main__":
    main()
