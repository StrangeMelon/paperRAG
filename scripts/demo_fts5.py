"""FTS5 稀疏检索真实验收: 真实英文论文库 + 真实中文期刊 chunk, 无 mock。

输入(均只读):
- demo-ingest-pipeline-data/data/index/papers.sqlite —— 入库课端到端产物,
  真实 Graph-Mamba 论文 49 chunks(复制到 demo-fts5-data/ 再操作, 不污染原库);
- demo-builder-data/parsed/sha1_ab3d...--mineru/chunks.json —— builder 课
  真实中文期刊(能源区块链)62 chunks。

验收点:
- 基准缺陷现场复现: 用基准同款建表语句索引中文原文, 召回坍缩到接近零
  (仅被标点巧合切出的独立完整 run 可命中, 真实期刊实测 1/26);
- 自愈回填: 复制来的库没有 chunks_fts(基准"先入库后首查恒空"时序),
  首次 search 自动全量回填并返回结果;
- 英文精确命中(GMB)与 porter 词干召回(dependency -> dependencies);
- 中文期刊入库后 bigram 子串检索可用(修复后的核心行为);
- paper_ids 过滤命中/滤空。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-fts5-data"
SRC_SQLITE = REPO_ROOT / "demo-ingest-pipeline-data/data/index/papers.sqlite"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)
EN_PAPER = "demo-pipeline-graph-mamba"

_BASELINE_TOKENIZE = "unicode61 remove_diacritics 2 tokenchars '_'"


def _point_config_at_demo_data() -> None:
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    config.load = lambda path=None: conf  # type: ignore[assignment]


def _baseline_zh_replication(zh_texts: list[str]) -> None:
    """[0] 基准同款分词器索引中文原文, 证明召回坍缩到接近零(ADR-0001 问题 1)。

    并非严格 0 命中: 查询串两侧恰好都被标点切开时(如 "系统。区块链、物联网"),
    unicode61 会把它收成独立完整 token, 靠这种巧合可命中个位数块。
    """
    contain = sum(1 for t in zh_texts if "区块链" in t)
    db = sqlite3.connect(":memory:")
    db.execute(f'CREATE VIRTUAL TABLE t USING fts5(text, tokenize = "{_BASELINE_TOKENIZE}")')
    db.executemany("INSERT INTO t VALUES (?)", [(t,) for t in zh_texts])
    n = db.execute("SELECT count(*) FROM t WHERE t MATCH ?", ('"区块链"',)).fetchone()[0]
    assert n * 5 < contain, f"基准召回应接近零: 命中 {n} / 字面含 {contain}"
    print(
        f"[0/5] 基准复现: 字面含 '区块链' 的 chunk 共 {contain} 个, 基准索引仅命中 {n} 个"
        f"(召回 {n / contain:.0%}, 只有被标点巧合切出的独立完整 run 才可命中)"
    )


def main() -> None:
    for required in (SRC_SQLITE, ZH_CHUNKS):
        if not required.is_file():
            print(f"缺少存量产物: {required.relative_to(REPO_ROOT)}", file=sys.stderr)
            raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_SQLITE, DEMO_ROOT / "papers.sqlite")

    _point_config_at_demo_data()

    from paper_rag.retrieve import fts5
    from paper_rag.store import sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    _baseline_zh_replication([c["text"] for c in zh_chunks])

    # --- 1. 自愈回填: 复制库无 chunks_fts, 首查不为空 ---
    with sqlite3.connect(DEMO_ROOT / "papers.sqlite") as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "chunks_fts" not in tables, "复制来的库不应已有 FTS 表"
    hits = fts5.search("long-range dependencies in graph sequence modeling", top_k=3)
    assert hits and hits[0]["paper_id"] == EN_PAPER, f"自愈回填后应命中英文论文: {hits[:1]}"
    print(f"[1/5] 自愈回填: 首查自动重建索引, top-1 score_bm25={hits[0]['score_bm25']:.3f}")

    # --- 2. 稀疏检索的看家本领: 精确命中缩写 GMB ---
    hits = fts5.search("GMB", top_k=5)
    assert hits and all(h["paper_id"] == EN_PAPER for h in hits)
    assert any("GMB" in h["text"] for h in hits), "命中块应字面含 GMB"
    print(f"[2/5] 精确命中: 'GMB' -> {len(hits)} 块, top-1 text 含 GMB={'GMB' in hits[0]['text']}")

    # --- 3. porter 词干召回 ---
    hits = fts5.search("dependency", top_k=5)
    assert hits and any("dependencies" in h["text"].lower() for h in hits), (
        "porter 应让 dependency 召回 dependencies"
    )
    print(f"[3/5] porter 词干: 'dependency' 召回含 'dependencies' 的块, 共 {len(hits)} 条")

    # --- 4. 中文期刊入库 -> bigram 子串检索可用 ---
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)
    expected = sum(1 for c in zh_chunks if "区块链" in c["text"])
    hits = fts5.search("区块链", top_k=20)  # 行数 49->111 不一致, 触发自愈重建
    zh_hits = [h for h in hits if h["paper_id"] == zh_paper]
    assert zh_hits and "区块链" in zh_hits[0]["text"]
    long_query = "综合能源服务区块链的网络架构"
    hits_long = fts5.search(long_query, top_k=5)
    assert hits_long and hits_long[0]["paper_id"] == zh_paper
    print(
        f"[4/5] 中文检索: '区块链' 命中 {len(zh_hits)} 块(全库字面含它的块 {expected} 个); "
        f"长查询 '{long_query}' top-1 score={hits_long[0]['score_bm25']:.3f}"
    )

    # --- 5. paper_ids 过滤 ---
    only_zh = fts5.search("网络架构", top_k=10, paper_ids=[zh_paper])
    assert only_zh and all(h["paper_id"] == zh_paper for h in only_zh)
    assert fts5.search("区块链", top_k=10, paper_ids=["no-such-paper"]) == []
    n = fts5.reindex_all()
    assert n == 49 + len(zh_chunks), f"全量重建行数应为两篇之和, 实得 {n}"
    print(f"[5/5] paper_ids 过滤命中/滤空; reindex_all -> {n} 行(49 英 + {len(zh_chunks)} 中)")

    print()
    print("FTS5 稀疏检索真实验收通过: 基准中文 0 命中缺陷已修复, 英文精确/词干召回正常。")


if __name__ == "__main__":
    main()
