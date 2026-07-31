#!/usr/bin/env python
"""解析调度器的真实降级验收 Demo。

用真实的 PDF、真实的配置加载和真实的 PyMuPDF 后端演示三件事:

1. MinerU 不可用时, 调度器降级到 PyMuPDF 并记录 ``degraded``;
2. 没有文字层的扫描件在两个后端都失败时, 记录 ``failed`` 并抛出 ``ParseError``,
   绝不把空结果伪装成 ``succeeded``;
3. 关闭 ``mineru.fallback_to_pymupdf`` 时, 调度器原样重抛 ``MineruError``。

MinerU 的失败不是 mock 出来的: Demo 生成一份临时配置, 把 ``mineru.cli`` 指向一个
不存在的可执行文件名, 生产代码 ``_resolve_cli()`` 找不到它就会真实抛出
``MineruError``。解析输出写入隔离的临时目录, 结束时清理, 不污染 ``data/parsed/``。

用法::

    uv run python scripts/demo_parse_dispatcher.py /absolute/text-paper.pdf
    uv run python scripts/demo_parse_dispatcher.py /absolute/scanned.pdf --keep-output

退出码::

    0  降级成功: 拿到可用正文, 且全部不变量成立
    1  不变量被破坏: 调度器行为与契约不符
    2  所有后端都没有产生正文(扫描件的正确结果), 状态已如实记录为 failed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MISSING_CLI_NAME = "paper-rag-demo-missing-mineru-cli"
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)

_failures: list[str] = []


def _step(index: int, total: int, message: str) -> None:
    print(f"\n[{index}/{total}] {message}")


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """打印并累计一条不变量检查结果。"""

    mark = "OK  " if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"    [{mark}] {label}{suffix}")
    if not condition:
        _failures.append(label)
    return condition


def _inspect_pdf(pdf_path: Path) -> tuple[int, int]:
    """返回 (页数, 前五页可提取字符数)。字符数为 0 表示没有文字层。"""

    try:
        import fitz
    except ImportError:
        print("PyMuPDF 未安装; 请执行: uv sync --extra dev --extra ingest")
        raise SystemExit(1) from None

    document = fitz.open(str(pdf_path))
    try:
        page_count = len(document)
        characters = 0
        for page_index in range(min(page_count, 5)):
            characters += len((document[page_index].get_text("text") or "").strip())
    finally:
        document.close()
    return page_count, characters


def _write_demo_config(
    config_path: Path,
    *,
    parsed_root: Path,
    fallback_to_pymupdf: bool,
) -> None:
    """基于真实 default.yaml 生成隔离的临时配置。"""

    raw: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    # 解析产物重定向到临时目录, 其余路径保持绝对以免落回项目 data/。
    data_root = parsed_root.parent
    raw["paths"] = {
        "data_root": str(data_root),
        "papers_dir": str(data_root / "papers"),
        "parsed_dir": str(parsed_root),
        "index_dir": str(data_root / "index"),
        "sqlite_path": str(data_root / "index" / "papers.sqlite"),
        "bm25_path": str(data_root / "index" / "bm25.pkl"),
        "models_dir": str(data_root / "index" / "models"),
    }
    raw["mineru"] = {
        **raw.get("mineru", {}),
        "mode": "local",
        "cli": MISSING_CLI_NAME,  # 真实不存在 -> 生产代码抛 MineruError
        "fallback_to_pymupdf": fallback_to_pymupdf,
    }
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _activate_config(config_path: Path) -> Any:
    """让生产配置加载器真实读取临时配置, 并返回加载结果。"""

    from paper_rag import config as cfg

    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    return cfg.load()


def _print_status_file(output_dir: Path) -> dict[str, Any]:
    status_path = output_dir / "parse_status.json"
    if not status_path.is_file():
        print(f"    parse_status.json 缺失: {status_path}")
        return {}
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"    {status_path}")
    for line in json.dumps(payload, ensure_ascii=False, indent=2).splitlines():
        print(f"      {line}")
    return payload


def _body_without_page_markers(output_dir: Path) -> str:
    markdown_path = output_dir / "paper.md"
    if not markdown_path.is_file():
        return ""
    return PAGE_MARKER_RE.sub(
        "", markdown_path.read_text(encoding="utf-8")
    ).strip()


def _scenario_degradation(
    pdf_path: Path,
    paper_id: str,
    workspace: Path,
    *,
    page_count: int,
    has_text_layer: bool,
) -> bool:
    """场景 A: MinerU 不可用且允许降级。返回是否拿到可用正文。"""

    from paper_rag.parse import dispatcher
    from paper_rag.utils.ids import to_safe_dirname

    parsed_root = workspace / "degrade" / "parsed"
    config_path = workspace / "degrade.yaml"
    _write_demo_config(config_path, parsed_root=parsed_root, fallback_to_pymupdf=True)
    config = _activate_config(config_path)

    print(f"    mineru.mode={config.mineru.mode}")
    print(f"    mineru.cli={config.mineru.cli}  (真实不存在, 用于触发 MineruError)")
    print(f"    mineru.method={config.mineru.method}  mineru.lang={config.mineru.lang}")
    print(f"    mineru.fallback_to_pymupdf={config.mineru.fallback_to_pymupdf}")
    print(f"    paths.parsed_dir={config.paths.parsed_dir}")

    expected_dir = parsed_root / to_safe_dirname(paper_id)
    print(f"\n    调用 dispatcher.parse_pdf({paper_id!r}, ...)")

    try:
        output_dir, parser_name = dispatcher.parse_pdf(paper_id, pdf_path)
    except dispatcher.ParseError as exc:
        print(f"    -> ParseError: {exc}")
        print("\n    parse_status.json:")
        status = _print_status_file(expected_dir)

        _check("扫描件被判定为 failed", status.get("status") == "failed", str(status.get("status")))
        _check("失败后端记录为 pymupdf", status.get("parser") == "pymupdf")
        _check("失败原因非空", bool(str(status.get("reason", "")).strip()))
        _check(
            "空结果没有被伪装成 succeeded",
            status.get("status") != "succeeded",
        )
        _check(
            "去掉页标记后确实没有正文",
            _body_without_page_markers(expected_dir) == "",
        )
        return False

    print(f"    -> 返回后端: {parser_name}")
    print(f"    -> 输出目录: {output_dir}")
    print("\n    parse_status.json:")
    status = _print_status_file(output_dir)

    body = _body_without_page_markers(output_dir)
    markdown = (output_dir / "paper.md").read_text(encoding="utf-8")
    marker_count = len(PAGE_MARKER_RE.findall(markdown))
    preview = body[:200].replace("\n", " ")
    print(f"\n    正文预览: {preview!r}")

    _check("返回目录与 parsed_dir 约定一致", output_dir == expected_dir, str(output_dir))
    _check("实际后端是 pymupdf", parser_name == "pymupdf", str(parser_name))
    _check("状态是 degraded", status.get("status") == "degraded", str(status.get("status")))
    _check("状态记录的后端是 pymupdf", status.get("parser") == "pymupdf")
    _check(
        "reason 保留了 MinerU 的失败原因",
        bool(str(status.get("reason", "")).strip()),
        str(status.get("reason", "")),
    )
    _check("paper_id 已记录", status.get("paper_id") == paper_id)
    _check("去掉页标记后仍有实义正文", bool(body), f"{len(body)} 字符")
    _check(
        "页标记数量等于 PDF 页数",
        marker_count == page_count,
        f"{marker_count} / {page_count}",
    )
    if not has_text_layer:
        _check(
            "无文字层的 PDF 不应产出正文",
            not body,
            "该 PDF 没有文字层却产出了正文, 请人工核对",
        )

    second_dir, second_parser = dispatcher.parse_pdf(paper_id, pdf_path)
    _check(
        "重复解析结果稳定",
        (second_dir, second_parser) == (output_dir, parser_name)
        and _body_without_page_markers(second_dir) == body,
    )
    return True


def _scenario_fallback_disabled(
    pdf_path: Path,
    paper_id: str,
    workspace: Path,
) -> None:
    """场景 B: 关闭降级开关, 调度器必须原样重抛 MineruError。"""

    from paper_rag.parse import dispatcher, mineru_local
    from paper_rag.utils.ids import to_safe_dirname

    parsed_root = workspace / "strict" / "parsed"
    config_path = workspace / "strict.yaml"
    _write_demo_config(config_path, parsed_root=parsed_root, fallback_to_pymupdf=False)
    config = _activate_config(config_path)

    print(f"    mineru.fallback_to_pymupdf={config.mineru.fallback_to_pymupdf}")
    expected_dir = parsed_root / to_safe_dirname(paper_id)

    try:
        dispatcher.parse_pdf(paper_id, pdf_path)
    except mineru_local.MineruError as exc:
        print(f"    -> 按预期重抛 MineruError: {exc}")
        raised = "mineru"
    except dispatcher.ParseError as exc:
        print(f"    -> 错误地降级后失败: ParseError: {exc}")
        raised = "parse_error"
    else:
        print("    -> 未抛出任何异常")
        raised = "none"

    print("\n    parse_status.json:")
    status = _print_status_file(expected_dir)

    _check("禁用降级时重抛 MineruError", raised == "mineru", raised)
    _check("状态是 failed", status.get("status") == "failed", str(status.get("status")))
    _check("状态记录的后端是 mineru", status.get("parser") == "mineru")
    _check(
        "没有调用 PyMuPDF 生成 paper.md",
        not (expected_dir / "paper.md").is_file(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="解析调度器的真实降级与失败隔离 Demo。",
    )
    parser.add_argument("pdf", help="用于验收的真实 PDF 绝对路径。")
    parser.add_argument(
        "--paper-id",
        default=None,
        help="解析目录使用的论文 ID; 默认根据文件名生成。",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="保留临时解析产物, 便于人工检查(默认结束时清理)。",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"PDF 不存在: {pdf_path}")
        return 1

    paper_id = args.paper_id or f"local:{pdf_path.stem}"
    workspace = Path(tempfile.mkdtemp(prefix="paper-rag-dispatcher-demo-"))
    previous_config = os.environ.get("PAPER_RAG_CONFIG")
    total = 5

    print("=" * 78)
    print("解析调度器真实降级 Demo")
    print("=" * 78)
    print(f"PDF        : {pdf_path}")
    print(f"paper_id   : {paper_id}")
    print(f"临时工作区 : {workspace}")

    try:
        _step(1, total, "检查输入 PDF 是否有文字层")
        page_count, characters = _inspect_pdf(pdf_path)
        has_text_layer = characters > 0
        print(f"    页数: {page_count}")
        print(f"    前五页可提取字符数: {characters}")
        print(
            "    判断: "
            + (
                "有文字层 -> 期望 PyMuPDF 降级成功 (degraded)"
                if has_text_layer
                else "无文字层(扫描件) -> 期望两个后端都失败 (failed)"
            )
        )

        _step(2, total, "生成隔离临时配置, 把 mineru.cli 指向不存在的可执行文件")
        _step(3, total, "场景 A: MinerU 不可用且允许降级")
        produced_text = _scenario_degradation(
            pdf_path,
            paper_id,
            workspace,
            page_count=page_count,
            has_text_layer=has_text_layer,
        )

        _step(4, total, "场景 B: 关闭 fallback_to_pymupdf, 必须重抛 MineruError")
        _scenario_fallback_disabled(pdf_path, paper_id, workspace)

        _step(5, total, "汇总")
        if _failures:
            print(f"    不变量失败 {len(_failures)} 项:")
            for label in _failures:
                print(f"      - {label}")
            print("\n    结果: FAILED — 调度器行为与契约不符。")
            return 1
        if not produced_text:
            print("    全部不变量成立, 但没有任何后端产出正文。")
            print("    这对没有文字层的扫描件是正确结果: 状态已如实记录为 failed,")
            print("    未伪装成 succeeded。请改用真实 MinerU OCR 处理该论文。")
            print("\n    结果: NO USABLE TEXT — 按约定以非零退出码报告。")
            return 2
        print("    全部不变量成立, 且 PyMuPDF 降级产出了可用正文。")
        print("\n    结果: PASSED")
        return 0
    finally:
        if previous_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = previous_config
        if args.keep_output:
            print(f"\n保留临时产物: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)
            print(f"\n已清理临时产物: {workspace}")


if __name__ == "__main__":
    raise SystemExit(main())
