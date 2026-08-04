"""rag/reflect.py 真实链路验收: 真实检索出口 -> 真实 Qwen 反思 -> 循环转动。

数据与 abstain Demo 同源(英文 Graph-Mamba 库副本 + 中文期刊真实嵌入), 真实
BGE-M3 + embedded Qdrant + FTS5 + 真实 reranker + **真实 LLM**(约 3 次调用)。

验收点:
- [1] 题内强证据英文问题(证据 rerank ~0.99): 不得判 insufficient。实测同一
      输入跨调用在 sufficient/partial 间摆动(DashScope temperature=0 不保证
      可复现), 断言钉稳定不变量, 具体档位打印观察; 受控完备证据下的
      == sufficient 断言由 tests/test_reflect_real.py 承担;
- [2] 证据缺口问题(库里无 ImageNet 对比数据): 判非 sufficient 且给出
      follow_up, **把 follow_up 真实喂回 retrieve_round 跑第二轮**——agentic
      循环在真实链路上转起来, 证据池按 chunk_id 跨轮累积;
- [3] 中文问题走中文模板: 题内强证据不得判 insufficient; 实测模型给出诚实的
      partial(摘要块确实没讲透组网细节), missing/follow_up 均为流利中文——
      中文模板与"follow_up 与问题同语言"指令生效的直接证据。

临时数据隔离在 demo-reflect-data/, 结束后保留供检查; 不触碰 data/ 与其他
demo 目录。任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-reflect-data"
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


def _fmt(r: dict) -> str:
    return (
        f"sufficiency={r['sufficiency']} score={r['score']:.2f} "
        f"missing={r['missing'][:40]!r} follow_up={r['follow_up'][:50]!r}"
    )


def _assert_contract(r: dict) -> None:
    assert set(r.keys()) == {"sufficiency", "missing", "follow_up", "score"}
    assert r["sufficiency"] in ("sufficient", "partial", "insufficient")
    assert isinstance(r["missing"], str) and isinstance(r["follow_up"], str)
    assert 0.0 <= r["score"] <= 1.0


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
        "reflect 真实验收需要 .env 提供 OPENAI_BASE_URL/OPENAI_API_KEY/CHAT_MODEL"
    )

    from paper_rag.embed import bge_m3
    from paper_rag.rag.reflect import reflect
    from paper_rag.retrieve.format import format_evidence
    from paper_rag.retrieve.pipeline import retrieve_round
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    sqlite_store.upsert_sections_and_chunks(
        zh_chunks[0]["paper_id"], zh_payload["sections"], zh_chunks
    )
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(
        f"[0/3] 双语双库就绪: 英文库副本 + 中文期刊 {len(zh_chunks)} 块; 模型 {conf.llm.chat_model}\n"
    )

    # ── 1) 题内强证据: 不得判 insufficient; 具体档位存在跨调用摆动 ──
    q_en = "How does Graph-Mamba handle long-range dependencies?"
    chunks1 = retrieve_round(q_en, None, 8)
    r1 = reflect(q_en, format_evidence(chunks1))
    _assert_contract(r1)
    print(f"[1/3] 强证据英文: {_fmt(r1)}")
    assert r1["sufficiency"] in ("sufficient", "partial"), "题内强证据被判 insufficient"
    if r1["sufficiency"] == "sufficient":
        print("      -> 循环收敛: stopped_by=answered, 不再花第二轮\n")
    else:
        assert r1["follow_up"], "非充分时应给出下一轮检索方向"
        print("      -> 临界证据判 partial(跨调用可在相邻档摆动), follow_up 将驱动下一轮\n")

    # ── 2) 证据缺口: 判不充分 + follow_up 真实驱动第二轮 ──
    q_gap = (
        "How does Graph-Mamba compare with vanilla Transformers "
        "on ImageNet classification accuracy?"
    )
    chunks2 = retrieve_round(q_gap, None, 8)
    r2 = reflect(q_gap, format_evidence(chunks2))
    _assert_contract(r2)
    print(f"[2/3] 缺口问题: {_fmt(r2)}")
    assert r2["sufficiency"] != "sufficient", "库里没有 ImageNet 对比数据, 不该判充分"
    assert r2["follow_up"], "不充分时应给出下一轮检索方向"
    pool = {c["chunk_id"]: c for c in chunks2}
    chunks2b = retrieve_round(r2["follow_up"], None, 8)
    for c in chunks2b:
        pool.setdefault(c["chunk_id"], c)
    print(
        f"      -> follow_up 驱动第二轮: 首轮 {len(chunks2)} 块, 第二轮 {len(chunks2b)} 块, "
        f"证据池并集 {len(pool)} 块 (agentic 循环真实转动)\n"
    )

    # ── 3) 中文问题: 中文模板, 题内强证据不得判 insufficient ──
    q_zh = "综合能源服务里区块链的网络架构是怎样设计的"
    chunks3 = retrieve_round(q_zh, None, 8)
    r3 = reflect(q_zh, format_evidence(chunks3))
    _assert_contract(r3)
    print(f"[3/3] 中文问题: {_fmt(r3)}")
    assert r3["sufficiency"] in ("sufficient", "partial"), "题内强证据被判 insufficient"
    if r3["sufficiency"] != "sufficient":
        assert r3["follow_up"], "非充分时应给出下一轮检索方向"
        assert any("一" <= ch <= "鿿" for ch in r3["follow_up"]), (
            "中文问题的 follow_up 应为中文(模板指令: 与问题同语言)"
        )
        print("      -> 诚实的 partial: 缺口与 follow_up 均为中文, 将驱动下一轮中文检索")

    print("\nDEMO PASSED: 强证据收敛 + 缺口反思驱动第二轮 + 中文模板 全部通过 (LLM 调用 3 次)")


if __name__ == "__main__":
    main()
