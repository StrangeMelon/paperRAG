"""rag/qa_simple.py 真实链路验收: 真实 dense 检索 -> 真实 Qwen -> 引用管道。

数据与 abstain/reflect Demo 同源(英文 Graph-Mamba 库副本 + 中文期刊真实嵌入),
真实 BGE-M3 + embedded Qdrant + **真实 LLM**(3 次调用)。本 Demo 同时兑现
citation_check 课确认的"真实链路覆盖并入 qa_simple Demo"。

验收点(硬不变量端到端):
- [1] 英文问题: citations 非空且全部 ⊆ 检索集; 答案剥净后 detect 复检
      count==0(执行端兜底成立); 答案文本携带 [chunk:<id>] 令牌;
- [2] 中文问题: 中文系统 prompt 路由下答案为中文, 同样引用纪律;
- [3] 域外问题(行为观察, 不断言): qa_simple 无 abstain, 检索仍返回最相似
      噪声块——与 abstain 课的对照教学点, 打印其行为。

临时数据隔离在 demo-qa-simple-data/; 任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-qa-simple-data"
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


def _assert_citation_discipline(out: dict, label: str) -> None:
    from paper_rag.rag.citation_check import detect_suspicious_citations

    retrieved_ids = {c["chunk_id"] for c in out["chunks"]}
    assert out["citations"], f"{label}: 强证据问题引用不应为空"
    assert set(out["citations"]) <= retrieved_ids, f"{label}: 引用了检索集之外的 id"
    assert detect_suspicious_citations(out["answer"])["count"] == 0, (
        f"{label}: 剥净后仍残留可疑引用形态"
    )
    assert "[chunk:" in out["answer"], f"{label}: 答案文本应携带引用令牌"


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
        "qa_simple 真实验收需要 .env 提供 OPENAI_BASE_URL/OPENAI_API_KEY/CHAT_MODEL"
    )

    from paper_rag.embed import bge_m3
    from paper_rag.rag.qa_simple import answer
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    sqlite_store.upsert_sections_and_chunks(
        zh_chunks[0]["paper_id"], zh_payload["sections"], zh_chunks
    )
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    print(
        f"[0/3] 双语双库就绪: 英文库副本 + 中文期刊 {len(zh_chunks)} 块; "
        f"模型 {conf.llm.chat_model}\n"
    )

    # ── 1) 英文问题: 硬不变量端到端 ──
    out_en = answer("How does Graph-Mamba handle long-range dependencies?", top_k=6)
    _assert_citation_discipline(out_en, "英文")
    print(
        f"[1/3] 英文: citations={len(out_en['citations'])} "
        f"suspicious={out_en['suspicious_citations']['count']}\n"
        f"      {out_en['answer'][:150]}…\n"
    )

    # ── 2) 中文问题: 中文 prompt 路由 + 同样纪律 ──
    out_zh = answer("综合能源服务里区块链的网络架构是怎样设计的", top_k=6)
    _assert_citation_discipline(out_zh, "中文")
    assert any("一" <= ch <= "鿿" for ch in out_zh["answer"]), "中文问题应得中文答案"
    print(
        f"[2/3] 中文: citations={len(out_zh['citations'])} "
        f"suspicious={out_zh['suspicious_citations']['count']}\n"
        f"      {out_zh['answer'][:150]}…\n"
    )

    # ── 3) 域外问题: 行为观察(qa_simple 无 abstain 的对照教学点) ──
    out_ood = answer("上海明天的天气怎么样", top_k=6)
    print(
        f"[3/3] 域外(无 abstain 对照): 检索仍返回 {len(out_ood['chunks'])} 块, "
        f"citations={len(out_ood['citations'])}\n"
        f"      {out_ood['answer'][:150]}…\n"
    )

    print(
        "DEMO PASSED: 英文/中文引用纪律端到端成立, citation_check 真实链路覆盖兑现 (LLM 调用 3 次)"
    )


if __name__ == "__main__":
    main()
