"""rag/qa_stream.py 真实链路验收: 事件流终端渲染器(P7 收官 Demo)。

数据与 qa_agentic Demo 同源, 真实 BGE-M3 + embedded Qdrant + FTS5 + reranker +
真实 Qwen(intent/reflect 非流式 + 作答**流式**)。

CLI 渲染约定(按用户要求: 详细回复全程可见):
- intent/rewrite/retrieved/reflect/abstain 每事件一行, 带关键数据;
- answer_chunk 逐 token 实时打印(打字机效果), **完整答案不截断**;
- done 后打印 citations/suspicious/paper_ids/证据选择明细。

验收点:
- [1] 英文域内: 事件序列合法(intent 首、done 尾), answer_chunk 事件数 > 1
      (真流式而非一次性), 引用纪律(citations ⊆ 证据集、无可疑残留);
- [2] 中文域内: 中文 prompt 路由下中文答案流式打出, 同样纪律;
- [3] 域外"上海天气": abstain 短路, 拒答文案也经 answer_chunk 流出,
      流式 LLM 未被调用。

临时数据隔离在 demo-qa-stream-data/; 任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-qa-stream-data"
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


def _run_streaming(question: str, label: str) -> dict:
    """跑一问并把事件流实时渲染到终端; 返回 {"events": [...], "done": {...}}。"""
    from paper_rag.rag.qa_stream import stream_answer

    print(f"┌─ {label}: {question}")
    events: list[dict] = []
    streaming_answer = False
    for ev in stream_answer(question):
        events.append(ev)
        name, data = ev["event"], ev["data"]
        if name == "answer_chunk":
            if not streaming_answer:
                print("│ [answer]   ", end="", flush=True)
                streaming_answer = True
            # 打字机效果: 逐 token 实时打出, 换行缩进对齐
            sys.stdout.write(data["text"].replace("\n", "\n│             "))
            sys.stdout.flush()
            continue
        if streaming_answer:
            print(flush=True)  # 收尾 answer 行
            streaming_answer = False
        if name == "intent":
            print(
                f"│ [intent]    {data['intent']} (top_k={data['top_k']}, max_iter={data['max_iter']})"
            )
        elif name == "rewrite":
            print(f"│ [rewrite]   queries={data['queries']}")
            print(f"│             keywords={data['keywords']!r}")
        elif name == "retrieved":
            print(f"│ [retrieved] 第 {data['iter'] + 1} 轮: {data['n_chunks']} 块")
        elif name == "reflect":
            print(
                f"│ [reflect]   {data['sufficiency']} (score={data['score']:.2f})"
                + (f" follow_up={data['follow_up']!r}" if data["follow_up"] else "")
            )
        elif name == "abstain":
            print(
                f"│ [abstain]   {data['decision']} (score={data['evidence_score']:.4f}, "
                f"field={data['score_field']})"
            )
        elif name == "done":
            print("│ [done]")
            print(f"│   citations  : {data['citations']}")
            print(f"│   suspicious : {data['suspicious']}")
            if "paper_ids" in data:
                print(f"│   paper_ids  : {data['paper_ids']}")
        elif name == "error":
            print(f"│ [error]     {data['message']}")
    print("└─\n")
    assert events[-1]["event"] == "done", f"{label}: 事件流必须以 done 收尾"
    return {"events": events, "done": events[-1]["data"]}


def _assert_streamed_answer(result: dict, label: str) -> None:
    from paper_rag.rag.citation_check import detect_suspicious_citations

    events, done = result["events"], result["done"]
    assert events[0]["event"] == "intent", f"{label}: intent 必须是首事件"
    n_chunks_events = sum(1 for e in events if e["event"] == "answer_chunk")
    assert n_chunks_events > 1, f"{label}: answer_chunk 仅 {n_chunks_events} 个, 不是真流式"
    evidence_ids = {c["chunk_id"] for c in done["evidence_chunks"]}
    assert done["citations"], f"{label}: 引用不应为空"
    assert set(done["citations"]) <= evidence_ids
    assert detect_suspicious_citations(done["answer"])["count"] == 0
    print(
        f"  ✓ {label}: {n_chunks_events} 个流式片段, citations={len(done['citations'])}, 纪律成立\n"
    )


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
        "qa_stream 真实验收需要 .env 提供 OPENAI_BASE_URL/OPENAI_API_KEY/CHAT_MODEL"
    )

    from paper_rag.embed import bge_m3
    from paper_rag.rag import qa_stream
    from paper_rag.store import qdrant_store, sqlite_store

    # 只统计流式作答调用(intent/reflect 各有自己的 chat 引用)
    stream_calls = {"n": 0}
    real_stream = qa_stream._stream_chat

    def counting_stream(system, user):
        stream_calls["n"] += 1
        yield from real_stream(system, user)

    qa_stream._stream_chat = counting_stream

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

    # ── 1) 英文域内: 流式全答案 ──
    r1 = _run_streaming("How does Graph-Mamba handle long-range dependencies?", "[1/3] 英文")
    _assert_streamed_answer(r1, "英文")

    # ── 2) 中文域内: 中文流式全答案 ──
    r2 = _run_streaming("综合能源服务里区块链的网络架构是怎样设计的", "[2/3] 中文")
    _assert_streamed_answer(r2, "中文")
    assert any("一" <= ch <= "鿿" for ch in r2["done"]["answer"])

    # ── 3) 域外: abstain 短路, 拒答文案也经 answer_chunk 流出 ──
    calls_before = stream_calls["n"]
    r3 = _run_streaming("上海明天的天气怎么样", "[3/3] 域外")
    done3 = r3["done"]
    assert stream_calls["n"] == calls_before, "no_evidence 时流式 LLM 不得被调用"
    abstain_ev = next(e for e in r3["events"] if e["event"] == "abstain")
    assert abstain_ev["data"]["decision"] == "no_evidence"
    chunk_evs = [e for e in r3["events"] if e["event"] == "answer_chunk"]
    assert len(chunk_evs) == 1 and chunk_evs[0]["data"]["text"] == done3["answer"]
    assert done3["answer"] == conf.rag.abstain.no_evidence_message, "中文问题得中文拒答文案"
    print("  ✓ 域外: abstain 短路, 流式 LLM 零调用, 拒答文案经 answer_chunk 流出\n")

    print("DEMO PASSED: 英文/中文流式全答案 + 事件序列 + 域外 abstain 短路 全部通过")


if __name__ == "__main__":
    main()
