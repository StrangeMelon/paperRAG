"""rag/evidence_select.py 真实链路验收: 真实检索出口 -> 确定性证据选择。

数据与 P6 retrieve_pipeline Demo 同源(英文 Graph-Mamba 库副本 + 中文期刊真实
嵌入), 真实 BGE-M3 + embedded Qdrant + FTS5 + 真实 reranker, 无 mock、无 LLM。

验收点:
- [1] 英文问题: 宽窗(8 块)收敛为紧凑证据集(≤4 块, 单篇 ≤2), trace 逐候选
      记账四项得分;
- [2] 中文问题: 选中块的 lexical_overlap > 0——CJK bigram 修复在真实数据上
      生效(基准 [a-z0-9]+ 对中文恒 0);
- [3] 跨语料混合问题: 单篇限额在真实混合窗口上成立;
- [4] 确定性: 同一检索出口跑两遍, 选集逐块一致。

临时数据隔离在 demo-evidence-select-data/, 结束后保留供检查(与其他 demo 目录
互不干扰); 不触碰 data/ 与 demo-ingest-pipeline-data/。任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-evidence-select-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def _per_paper_counts(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["paper_id"]] = counts.get(c["paper_id"], 0) + 1
    return counts


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
    from paper_rag.rag.evidence_select import select_evidence
    from paper_rag.retrieve.pipeline import retrieve_round
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    zh_paper = zh_chunks[0]["paper_id"]
    sqlite_store.upsert_sections_and_chunks(zh_paper, zh_payload["sections"], zh_chunks)
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(f"[0/4] 双语双库就绪: 英文库副本 + 中文期刊 {len(zh_chunks)} 块\n")

    # ── 1) 英文问题: 宽窗收敛 + trace 记账 ──
    q_en = "How does Graph-Mamba handle long-range dependencies?"
    window = retrieve_round(q_en, None, 8)
    assert len(window) > 4, "宽窗应大于证据集上限, 否则选择器无事可做"
    selected, trace = select_evidence(q_en, window, intent="factual")
    assert 0 < len(selected) <= 4
    assert max(_per_paper_counts(selected).values()) <= 2
    assert len(trace["candidates"]) == len(window), "落选块也要记账"
    for cand in trace["candidates"]:
        for key in ("selection_score", "model_score", "lexical_overlap", "section_hint"):
            assert key in cand
    top = trace["candidates"][0]
    print(
        f"[1/4] 英文: 宽窗 {len(window)} -> 选 {len(selected)} 块; top-1 "
        f"{top['chunk_id']} (model={top['model_score']:.3f}, "
        f"overlap={top['lexical_overlap']:.3f}, hint={top['section_hint']})"
    )

    # ── 2) 中文问题: CJK bigram 重叠在真实数据上生效 ──
    q_zh = "综合能源服务里区块链的网络架构是怎样设计的"
    window_zh = retrieve_round(q_zh, None, 8)
    selected_zh, trace_zh = select_evidence(q_zh, window_zh, intent="reasoning")
    assert selected_zh and selected_zh[0]["paper_id"] == zh_paper
    top_zh = trace_zh["candidates"][0]
    assert top_zh["lexical_overlap"] > 0.0, "中文问题 overlap 恒 0 意味着基准盲区未修复"
    print(
        f"[2/4] 中文: 宽窗 {len(window_zh)} -> 选 {len(selected_zh)} 块; top-1 "
        f"overlap={top_zh['lexical_overlap']:.3f} > 0 (CJK bigram 生效), "
        f"section={selected_zh[0].get('section', '')[:12]}"
    )

    # ── 3) 混合问题: 真实混合窗口上的单篇限额 ──
    q_mix = "长序列建模与区块链网络架构"
    window_mix = retrieve_round(q_mix, None, 8)
    selected_mix, _ = select_evidence(q_mix, window_mix)
    counts = _per_paper_counts(selected_mix)
    assert max(counts.values()) <= 2, f"单篇限额被打破: {counts}"
    print(f"[3/4] 混合问题: 选 {len(selected_mix)} 块, 论文分布 {counts} (单篇 ≤2)")

    # ── 4) 确定性: 同一窗口两遍选择逐块一致 ──
    again, trace_again = select_evidence(q_mix, window_mix)
    assert [c["chunk_id"] for c in again] == [c["chunk_id"] for c in selected_mix]
    assert trace_again["candidates"] == _last_candidates(select_evidence, q_mix, window_mix)
    print("[4/4] 确定性: 同一窗口两遍选择, 选集与全部候选得分逐项一致\n")

    print("DEMO PASSED: evidence_select 真实链路收敛 + 中文重叠 + 单篇限额 + 确定性 全部通过")


def _last_candidates(fn, question, window):
    return fn(question, window)[1]["candidates"]


if __name__ == "__main__":
    main()
