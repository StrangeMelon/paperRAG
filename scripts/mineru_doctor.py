#!/usr/bin/env python3
"""诊断本地 MinerU/magic-pdf 是否具备运行条件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_rag.parse import mineru_local  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 Doctor 命令行参数。"""

    parser = argparse.ArgumentParser(
        description="检查本地 MinerU GPU OCR 运行环境。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出适合程序读取的 JSON 报告。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="存在失败检查时返回非零退出码。",
    )
    parser.add_argument(
        "--try-parse",
        metavar="PDF",
        help="使用指定 PDF 执行一次真实 MinerU 解析。",
    )
    parser.add_argument(
        "--paper-id",
        default="mineru:doctor",
        help="试解析产物使用的论文 ID。",
    )
    return parser.parse_args()


def _print_human(payload: dict[str, Any]) -> None:
    """把诊断报告格式化为便于终端阅读的文本。"""

    print("MinerU doctor")
    print(f"  ok: {payload['ok']}")
    print(f"  cli: {payload.get('cli_path')}")
    print(f"  config: {payload.get('config_path')}")
    print("\nChecks:")

    for check in payload["checks"]:
        mark = "OK" if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['name']}: {check['detail']}")

        if not check["ok"] and check.get("hint"):
            print(f"        hint: {check['hint']}")

    if payload.get("try_parse"):
        trial = payload["try_parse"]
        print("\nTry parse:")
        print(f"  ok: {trial['ok']}")

        if trial["ok"]:
            print(f"  out_dir: {trial['out_dir']}")
        else:
            print(f"  reason: {trial['reason']}")
            if trial.get("hint"):
                print(f"  hint: {trial['hint']}")


def main() -> int:
    """运行诊断并输出报告。"""

    args = parse_args()
    payload = mineru_local.diagnose().to_dict()

    if args.try_parse:
        try:
            output_dir = mineru_local.parse_pdf(
                args.paper_id,
                args.try_parse,
            )
            payload["try_parse"] = {
                "ok": True,
                "out_dir": str(output_dir),
            }
        except mineru_local.MineruError as exc:
            reason, hint = mineru_local.classify_failure(str(exc))
            payload["try_parse"] = {
                "ok": False,
                "reason": reason,
                "error": str(exc),
                "hint": hint,
            }

    if args.json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_human(payload)

    ok = bool(payload["ok"])
    if payload.get("try_parse"):
        ok = ok and bool(payload["try_parse"]["ok"])

    if args.strict and not ok:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
