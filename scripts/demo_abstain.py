"""rag/abstain.py 真实链路验收: 真实检索出口 -> 三路证据充分性裁决。

数据与 evidence_select Demo 同源(英文 Graph-Mamba 库副本 + 中文期刊真实嵌入),
真实 BGE-M3 + embedded Qdrant + FTS5 + 真实 reranker, 无 mock、无 LLM。阈值取
真实配置 `rag.abstain`(YAML 校准值), 不用函数签名缺省。

验收点:
- [1] 域内英文/中文问题: 走完整精排链后 score_rerank 证据分过高阈 -> confident;
- [2] 域外问题("上海明天的天气怎么样"): 检索照样返回 top-k 噪声块, 但证据分
      塌陷, 裁决为 no_evidence/weak——这正是"无证据宁拒答"被执行的现场;
- [3] fail open: 同一批域外块剥掉高质字段(rerank/dense/score)后只剩低质信号,
      裁决翻回 confident + low_degraded; 全剥则 confident + missing;
- [4] 协议完备: 空列表 no_chunks(schema 拉平带 signal_quality), enabled=False
      逃生门, weak hint zh/en 路由。

临时数据隔离在 demo-abstain-data/, 结束后保留供检查; 不触碰 data/ 与其他
demo 目录。任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-abstain-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)

HIGH_QUALITY = ("score_rerank", "score_dense", "score")


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def _decide_with_cfg(chunks: list[dict], abstain_cfg, **over):
    from paper_rag.rag import abstain

    kwargs = {
        "enabled": abstain_cfg.enabled,
        "threshold_low": abstain_cfg.threshold_low,
        "threshold_high": abstain_cfg.threshold_high,
        "min_chunks": abstain_cfg.min_chunks,
    }
    kwargs.update(over)
    return abstain.decide(chunks, **kwargs)


def _fmt(result: dict) -> str:
    return (
        f"decision={result['decision']} score={result['evidence_score']:.4f} "
        f"top={result['top_chunk_score']:.4f} field={result['score_field']} "
        f"quality={result['signal_quality']}"
    )


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
    abstain_cfg = conf.rag.abstain
    assert abstain_cfg.threshold_low == 0.21 and abstain_cfg.threshold_high == 0.48, (
        "Demo 依赖 YAML 校准阈值, 不是函数签名缺省"
    )

    from paper_rag.embed import bge_m3
    from paper_rag.rag import abstain
    from paper_rag.retrieve.pipeline import retrieve_round
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    sqlite_store.upsert_sections_and_chunks(
        zh_chunks[0]["paper_id"], zh_payload["sections"], zh_chunks
    )
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(
        f"[0/4] 双语双库就绪: 英文库副本 + 中文期刊 {len(zh_chunks)} 块; "
        f"阈值 low={abstain_cfg.threshold_low} high={abstain_cfg.threshold_high} "
        f"min_chunks={abstain_cfg.min_chunks}\n"
    )

    # ── 1) 域内问题: 证据分过高阈 -> confident ──
    q_en = "How does Graph-Mamba handle long-range dependencies?"
    chunks_en = retrieve_round(q_en, None, 8)
    r_en = _decide_with_cfg(chunks_en, abstain_cfg)
    print(f"[1/4] 域内英文: {_fmt(r_en)}")
    assert r_en["decision"] == abstain.DECISION_CONFIDENT
    assert r_en["score_field"] == "score_rerank" and r_en["signal_quality"] == "high"

    q_zh = "综合能源服务里区块链的网络架构是怎样设计的"
    chunks_zh = retrieve_round(q_zh, None, 8)
    r_zh = _decide_with_cfg(chunks_zh, abstain_cfg)
    print(f"      域内中文: {_fmt(r_zh)}")
    assert r_zh["decision"] == abstain.DECISION_CONFIDENT, "中文域内问题不应被拒答"

    # ── 2) 域外问题: 检索照样有 top-k, 但证据分塌陷 -> 拒答/弱证据 ──
    q_out = "上海明天的天气怎么样"
    chunks_out = retrieve_round(q_out, None, 8)
    r_out = _decide_with_cfg(chunks_out, abstain_cfg)
    print(f"[2/4] 域外问题: 检索仍返回 {len(chunks_out)} 块 -> {_fmt(r_out)}")
    assert len(chunks_out) > 0, "检索对域外问题也会返回'最相似'块, 这正是需要 abstain 的原因"
    assert r_out["decision"] in (abstain.DECISION_NO_EVIDENCE, abstain.DECISION_WEAK)
    assert r_out["evidence_score"] < r_en["evidence_score"]
    assert r_out["evidence_score"] < r_zh["evidence_score"]
    if r_out["decision"] == abstain.DECISION_NO_EVIDENCE:
        print(f"      -> LLM 将被跳过, 用户收到拒答文案: {abstain_cfg.no_evidence_message[:24]}…")

    # ── 3) fail open: 剥掉高质字段后低质信号不触发拒答 ──
    degraded = [{k: v for k, v in c.items() if k not in HIGH_QUALITY} for c in chunks_out]
    assert any("score_rrf" in c or "score_bm25" in c for c in degraded)
    r_deg = _decide_with_cfg(degraded, abstain_cfg)
    print(f"[3/4] 同批域外块剥高质字段: {_fmt(r_deg)}")
    assert r_deg["decision"] == abstain.DECISION_CONFIDENT
    assert r_deg["signal_quality"] == "low_degraded", "低质信号必须放行并透出降级态"

    naked = [{"chunk_id": c["chunk_id"], "text": c.get("text", "")} for c in chunks_out]
    r_naked = _decide_with_cfg(naked, abstain_cfg)
    assert (r_naked["decision"], r_naked["signal_quality"]) == (
        abstain.DECISION_CONFIDENT,
        "missing",
    )
    print(f"      全剥分数字段: {_fmt(r_naked)} (fail open 而非阻塞)")

    # ── 4) 协议完备: no_chunks / 逃生门 / hint 路由 ──
    r_empty = _decide_with_cfg([], abstain_cfg)
    assert r_empty["decision"] == abstain.DECISION_NO_CHUNKS
    assert r_empty["signal_quality"] == "no_chunks", "确认偏离 b: schema 拉平"
    r_off = _decide_with_cfg(chunks_out, abstain_cfg, enabled=False)
    assert (r_off["decision"], r_off["signal_quality"]) == (abstain.DECISION_CONFIDENT, "disabled")
    assert abstain.weak_evidence_hint("zh").lstrip().startswith("注意")
    assert abstain.weak_evidence_hint("en") is abstain.WEAK_EVIDENCE_HINT
    print(
        f"[4/4] 协议完备: 空列表 -> {r_empty['decision']}(quality={r_empty['signal_quality']}); "
        f"enabled=False -> {r_off['decision']}; weak hint zh/en 路由正常\n"
    )

    print("DEMO PASSED: 域内放行 + 域外拒答 + fail open 降级 + 协议完备 全部通过")


if __name__ == "__main__":
    main()
