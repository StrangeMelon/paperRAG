"""对 RAG 库提问的 CLI 入口(核心三步 init_store -> ingest_one -> ask 的收尾)。

Usage:
    python scripts/ask.py "What is the main contribution?"
    python scripts/ask.py "..." --paper-id arxiv:2310.12345 --top-k 6
    python scripts/ask.py "..." --no-llm     # 只看检索结果, 不调 LLM
    python scripts/ask.py "..." --agentic    # 完整 agentic 链路 + trace 摘要
    python scripts/ask.py "..." --stream     # 流式打字机(qa_stream 事件渲染)

相对基准的确认偏离: 基准停在 phase-1 只接 qa_simple(agentic 的消费方是
DeerFlow 网关, 没人回来升级 CLI); 重建版补 --agentic/--stream 两个互斥模式,
默认模式仍 qa_simple 与基准同构。输出标头保持基准英文(工具层文案, 答案本身
已由引擎按问题语言路由)。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_rag import config as cfg
from paper_rag.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
log = get_logger("ask")


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。

    CLI 是独立进程入口, 没有 conftest 帮忙加载 .env——用户直跑时 LLM 配置
    必须在这里补上(2026-08-05 用户实跑暴露: 缺此步 rewrite/reflect 全降级、
    作答直接 error)。
    """
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask a question against the paper RAG store.")
    p.add_argument("question")
    p.add_argument("--paper-id", action="append", help="Restrict search to specific paper_id(s)")
    p.add_argument("--top-k", type=int, default=8)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-llm", action="store_true", help="Only show retrieved chunks (no LLM answer)"
    )
    mode.add_argument(
        "--agentic",
        action="store_true",
        help="Full agentic loop (intent/reflect/abstain) with trace summary",
    )
    mode.add_argument(
        "--stream", action="store_true", help="Stream the agentic answer token by token"
    )
    return p.parse_args(argv)


def _print_answer_block(out: dict) -> None:
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print(f"\n=== CITATIONS ({len(out['citations'])}) ===")
    for cid in out["citations"]:
        print(f"  - {cid}")


def _run_stream(question: str, paper_ids: list[str] | None) -> int:
    """qa_stream 事件流的终端打字机渲染; error 事件返回非零。"""
    from paper_rag.rag.qa_stream import stream_answer

    streaming = False
    failed = False
    for ev in stream_answer(question, paper_ids=paper_ids):
        name, data = ev["event"], ev["data"]
        if name == "answer_chunk":
            if not streaming:
                print("\n=== ANSWER (streaming) ===\n")
                streaming = True
            sys.stdout.write(data["text"])
            sys.stdout.flush()
            continue
        if streaming:
            print(flush=True)
            streaming = False
        if name == "intent":
            print(
                f"[intent]    {data['intent']} (top_k={data['top_k']}, max_iter={data['max_iter']})"
            )
        elif name == "rewrite":
            print(f"[rewrite]   {len(data['queries'])} queries; keywords={data['keywords']!r}")
        elif name == "retrieved":
            print(f"[retrieved] iter={data['iter']} n_chunks={data['n_chunks']}")
        elif name == "reflect":
            print(f"[reflect]   {data['sufficiency']} (score={data['score']:.2f})")
        elif name == "abstain":
            print(
                f"[abstain]   {data['decision']} (score={data['evidence_score']:.4f}, "
                f"field={data['score_field']})"
            )
        elif name == "done":
            print(f"\n=== CITATIONS ({len(data['citations'])}) ===")
            for cid in data["citations"]:
                print(f"  - {cid}")
        elif name == "error":
            print(f"[error]     {data['message']}", file=sys.stdout)
            failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_dotenv(_REPO_ROOT / ".env")
    cfg.load()

    if args.no_llm:
        from paper_rag.retrieve.dense import retrieve
        from paper_rag.retrieve.format import format_evidence

        chunks = retrieve(args.question, top_k=args.top_k, paper_ids=args.paper_id)
        print(format_evidence(chunks))
        return 0

    if args.stream:
        return _run_stream(args.question, args.paper_id)

    if args.agentic:
        from paper_rag.rag.qa_agentic import answer

        out = answer(args.question, paper_ids=args.paper_id)
        _print_answer_block(out)
        trace = out.get("trace") or {}
        abstain = trace.get("abstain") or {}
        loop = trace.get("loop") or {}
        print("\n=== TRACE ===")
        intent = trace.get("intent") or {}
        print(
            f"  intent={intent.get('intent')} iters={len(trace.get('iters') or [])} "
            f"stopped_by={trace.get('stopped_by')}"
        )
        print(
            f"  abstain={abstain.get('decision')}({abstain.get('evidence_score')}) "
            f"latency={loop.get('latency_ms')}ms"
        )
        return 0

    from paper_rag.rag.qa_simple import answer

    out = answer(args.question, top_k=args.top_k, paper_ids=args.paper_id)
    _print_answer_block(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
