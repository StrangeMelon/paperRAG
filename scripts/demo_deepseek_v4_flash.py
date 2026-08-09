"""DeepSeek V4 Flash Wiki 调用真实验收。

只发起一次短请求, 不读写数据库。验证 Wiki 专用配置、DeepSeek 特殊参数、
OpenAI 兼容端点连通性, 以及 Wiki 抽取所需的 JSON 输出。

Usage:
    uv run python scripts/demo_deepseek_v4_flash.py
    uv run python scripts/demo_deepseek_v4_flash.py --thinking enabled --reasoning-effort low
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_PROMPT = """从下面的论文摘要中抽取一个适合建立 Wiki 词条的研究概念。

摘要:
Retrieval-augmented generation combines external document retrieval with language model
generation to improve factual accuracy and provide evidence-grounded answers.

只返回 JSON, 不要 Markdown 代码块:
{"concepts":[{"name":"...","category":"concept|method","definition":"..."}]}
"""


def _load_dotenv(path: Path) -> None:
    """读取简单 KEY=VALUE 行, 不覆盖已导出的环境变量。"""
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DeepSeek V4 Flash for Wiki extraction.")
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        help="Override wiki.llm.thinking for this request.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="Override reasoning effort; only sent when thinking is enabled.",
    )
    return parser.parse_args(argv)


def _validate_config(wiki_llm: Any) -> None:
    missing = [
        name
        for name, value in (
            ("WIKI_LLM_BASE_URL", wiki_llm.base_url),
            ("WIKI_LLM_API_KEY", wiki_llm.api_key),
            ("WIKI_LLM_MODEL", wiki_llm.model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"缺少 Wiki DeepSeek 配置: {', '.join(missing)}。请在 .env 或当前 shell 中设置后重跑。"
        )
    if not str(wiki_llm.model).endswith("deepseek-v4-flash"):
        raise ValueError(
            f"WIKI_LLM_MODEL={wiki_llm.model!r}, 不是 deepseek-v4-flash; "
            "为避免误扣其他模型费用, 本验收已停止。"
        )


def _parse_json_reply(reply: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError(f"回复中没有 JSON 对象: {reply[:300]!r}")
    payload = json.loads(match.group(0))
    concepts = payload.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError(f"concepts 不是非空列表: {payload!r}")
    concept = concepts[0]
    if not isinstance(concept, dict):
        raise ValueError(f"首个概念不是对象: {concept!r}")
    for field in ("name", "category", "definition"):
        if not isinstance(concept.get(field), str) or not concept[field].strip():
            raise ValueError(f"首个概念缺少非空字段 {field}: {concept!r}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_dotenv(REPO_ROOT / ".env")

    import paper_rag.config as config
    from paper_rag.rag import llm as base_llm
    from paper_rag.wiki import llm as wiki_llm_client

    config.load.cache_clear()
    base_llm.reset_client_for_test()
    conf = config.load()
    wiki_llm = conf.wiki.llm
    try:
        _validate_config(wiki_llm)
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    if args.thinking is not None:
        wiki_llm.thinking = args.thinking
    if args.reasoning_effort is not None:
        wiki_llm.reasoning_effort = args.reasoning_effort
    if wiki_llm.thinking == "disabled":
        wiki_llm.reasoning_effort = None

    print("DeepSeek V4 Flash Wiki smoke test")
    print(f"  endpoint={wiki_llm.base_url}")
    print(f"  model={wiki_llm.model}")
    print(f"  thinking={wiki_llm.thinking or 'provider_default'}")
    print(f"  reasoning_effort={wiki_llm.reasoning_effort or 'none'}")
    print(f"  timeout_sec={wiki_llm.timeout_sec}")

    started = time.perf_counter()
    try:
        reply = wiki_llm_client.chat(
            [{"role": "user", "content": _PROMPT}],
            max_tokens=400,
        )
        elapsed = time.perf_counter() - started
        payload = _parse_json_reply(reply)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"\nDEMO FAILED after {elapsed:.2f}s: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    concept = payload["concepts"][0]
    print(f"\n  elapsed={elapsed:.2f}s")
    print(f"  concept.name={concept['name']}")
    print(f"  concept.category={concept['category']}")
    print(f"  concept.definition={concept['definition']}")
    print("\nDEMO PASSED: DeepSeek V4 Flash 连通、特殊参数与 Wiki JSON 输出均正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
