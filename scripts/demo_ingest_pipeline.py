"""入库流水线端到端真实验收: 一篇真实论文从 PDF 到可检索, 全链无 mock。

输入: demo-local-data 里的真实 Graph-Mamba PDF(采集课的存量产物)。
隔离: 运行时数据全部落在 demo-ingest-pipeline-data/(SQLite 真实临时库 +
embedded Qdrant 隔离目录), 仅模型缓存复用真实 data/index/models(BGE-M3 与
MinerU 权重, 只读)。真实 GPU MinerU 解析约 1-3 分钟。

验收点:
- 状态机走完: 论文最终 status=done, parsed_with="mineru+complete"
  (解析器名 + 章节完整性打分), ingest_runs 四步(parse/chunk/embed/index)全 ok;
- 元数据卡片: chunks[0] modality="metadata", 别名含 "GM"(Graph-Mamba 缩写);
- 检索闭环: 真实英文问题编码后在 Qdrant 命中本论文(全链第一次问答式检索);
- 幂等: 重复 ingest -> skipped/done; force=True 重建后 Qdrant 点数不变
  (先删后插的替换语义, 不残留脏向量);
- wiki 钩子: 持久化入队成功, 返回 {"queued": True, ...}(任务由 wiki_worker 异步消费)。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-ingest-pipeline-data"
PDF = REPO_ROOT / "demo-local-data/papers/sha1_28acb520c921be7a1968207519dfa95d6af88800/raw.pdf"


def _isolate_config() -> None:
    """运行时路径与 Qdrant 全部指向 demo 目录; 模型缓存保持真实(只读复用)。"""
    import paper_rag.config as config

    conf = config.load()
    conf.paths.data_root = str(DEMO_ROOT / "data")
    conf.paths.papers_dir = str(DEMO_ROOT / "data/papers")
    conf.paths.parsed_dir = str(DEMO_ROOT / "data/parsed")
    conf.paths.index_dir = str(DEMO_ROOT / "data/index")
    conf.paths.sqlite_path = str(DEMO_ROOT / "data/index/papers.sqlite")
    conf.paths.bm25_path = str(DEMO_ROOT / "data/index/bm25.pkl")
    # models_dir 不动: 复用已缓存的 BGE-M3, 不重复下载
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]


def main() -> None:
    if not PDF.is_file():
        print(f"缺少输入 PDF: {PDF.relative_to(REPO_ROOT)}", file=sys.stderr)
        raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()

    _isolate_config()

    from paper_rag.embed import bge_m3
    from paper_rag.ingest.schema import FetchResult, PaperMeta
    from paper_rag.store import qdrant_store, sqlite_store
    from paper_rag.store.ingest_pipeline import ingest
    from paper_rag.utils.paths import ensure_dirs
    from scripts.init_store import init_qdrant, init_sqlite

    ensure_dirs()
    init_sqlite()
    init_qdrant()

    paper_id = "demo-pipeline-graph-mamba"
    result = FetchResult(
        meta=PaperMeta(
            paper_id=paper_id,
            title="Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces",
            authors=["Chloe Wang", "Oleksii Tsepa", "Jun Ma", "Bo Wang"],
            year=2024,
            arxiv_id="2402.00789",
            abstract=(
                "Graph-Mamba improves long-range context modeling in graph networks "
                "by integrating a Mamba block with input-dependent node selection."
            ),
            source="local_pdf",
        ),
        pdf_path=str(PDF),
    )

    # --- 1. 端到端入库(真实 MinerU GPU 解析, 约 1-3 分钟) ---
    res = ingest(result)
    assert res["status"] == "done", f"入库未完成: {res}"
    n_chunks = res["chunks"]
    assert n_chunks > 40, f"chunk 数异常: {n_chunks}"
    assert res["wiki"].get("queued") is True, f"wiki 持久化入队失败: {res['wiki']}"
    print(f"[1/5] 端到端入库完成: status=done, chunks={n_chunks}(含元数据卡片)")

    # --- 2. SQLite 状态机与步骤记录 ---
    paper = sqlite_store.get_paper(paper_id)
    assert paper is not None and paper.status == "done"
    assert paper.parsed_with == "mineru+complete", f"parsed_with={paper.parsed_with}"
    from sqlmodel import Session, select

    with Session(sqlite_store.get_engine()) as session:
        runs = session.exec(
            select(sqlite_store.IngestRun)
            .where(sqlite_store.IngestRun.paper_id == paper_id)
            .order_by(sqlite_store.IngestRun.id)
        ).all()
    steps = [(r.step, r.status) for r in runs]
    assert steps == [("parse", "ok"), ("chunk", "ok"), ("embed", "ok"), ("index", "ok")], steps
    print(f"[2/5] 状态机: status=done, parsed_with={paper.parsed_with}, 四步全 ok")

    # --- 3. 元数据卡片 ---
    rows = sqlite_store.list_chunks_for_papers([paper_id])
    assert len(rows) == n_chunks, f"SQLite chunk 数 {len(rows)} != {n_chunks}"
    cards = [r for r in rows if r.modality == "metadata"]
    assert len(cards) == 1
    card_meta = json.loads(cards[0].metadata_json)
    assert card_meta.get("aliases") == ["GM"], f"别名异常: {card_meta.get('aliases')}"
    print(f"[3/5] 元数据卡片: 1 张, 别名={card_meta.get('aliases')}")

    # --- 4. 检索闭环: 真实问题 -> 向量 -> Qdrant 命中本论文 ---
    q = bge_m3.encode_one(
        "How does Graph-Mamba capture long-range dependencies with selective state spaces?"
    )
    hits = qdrant_store.search(q, top_k=3)
    assert hits, "Qdrant 检索无结果"
    assert hits[0]["paper_id"] == paper_id, f"top-1 非本论文: {hits[0].get('paper_id')}"
    top_desc = [(f"{h['score']:.3f}", h.get("modality"), h.get("section", "")[:40]) for h in hits]
    print(f"[4/5] 检索闭环: top-3 = {top_desc}")

    # --- 5. 幂等与替换语义 ---
    res2 = ingest(result)
    assert res2 == {"paper_id": paper_id, "status": "skipped", "reason": "done"}, res2
    client = qdrant_store.get_client()
    coll = "paper_chunks"
    before = client.count(collection_name=coll).count
    res3 = ingest(result, force=True)
    after = client.count(collection_name=coll).count
    assert res3["status"] == "done"
    assert before == after == n_chunks, f"替换语义被破坏: {before} -> {after}, 期望 {n_chunks}"
    print(f"[5/5] 幂等: 重复 ingest -> skipped/done; force 重建后 Qdrant 点数 {after} 不变")

    (DEMO_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "chunks": n_chunks,
                "parsed_with": paper.parsed_with,
                "top3": top_desc,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print()
    print("入库流水线端到端真实验收通过: PDF -> 解析 -> 切块 -> 嵌入 -> 索引 -> 可检索。")


if __name__ == "__main__":
    main()
