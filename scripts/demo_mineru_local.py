"""使用用户提供的真实 PDF 演示本地 MinerU 解析。"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from paper_rag import config as cfg
from paper_rag.parse.language import resolve_ocr_language
from paper_rag.parse.mineru_local import _resolve_cli, parse_pdf
from paper_rag.utils.ids import make_paper_id, to_safe_dirname


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本机 MinerU 将真实 PDF 解析为标准项目产物。",
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        help="待解析 PDF 的本地路径; 省略时交互输入",
    )
    parser.add_argument(
        "--paper-id",
        help="可选论文 ID; 默认根据 PDF 内容生成 sha1 ID",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="可选持久化 data_root; 未提供时使用临时目录并自动清理",
    )
    parser.add_argument(
        "--cli",
        help="MinerU CLI 名称或路径; 默认读取 mineru.cli",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "txt", "ocr"),
        default="ocr",
        help="MinerU 解析模式, 默认强制 OCR",
    )
    parser.add_argument(
        "--lang",
        choices=("auto", "ch", "en"),
        default="auto",
        help="OCR 语言路由模式, 默认 auto 逐篇自动判断",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="解析超时秒数, 默认 600",
    )
    return parser.parse_args()


def _resolve_pdf_path(value: Path | None) -> Path:
    if value is None:
        entered = input("请输入待解析 PDF 的本地路径: ").strip()
        if not entered:
            raise ValueError("PDF 路径不能为空")
        value = Path(entered)

    pdf_path = value.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    return pdf_path


def _inspect_pdf(pdf_path: Path) -> tuple[int, str]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF 未安装, 请执行: uv sync --extra dev --extra ingest"
        ) from exc

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"无法打开 PDF: {pdf_path}") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"PDF 需要密码: {pdf_path}")
        if document.page_count < 1:
            raise ValueError(f"PDF 没有页面: {pdf_path}")
        first_text = (document[0].get_text("text") or "").replace("\x00", "").strip()
        return document.page_count, first_text
    finally:
        document.close()


def _write_isolated_config(
    config_path: Path,
    data_root: Path,
    *,
    cli_path: str,
    method: str,
    lang: str | None,
    timeout: int,
) -> None:
    paths = {
        "data_root": str(data_root),
        "papers_dir": str(data_root / "papers"),
        "parsed_dir": str(data_root / "parsed"),
        "index_dir": str(data_root / "index"),
        "sqlite_path": str(data_root / "index" / "papers.sqlite"),
        "bm25_path": str(data_root / "index" / "bm25.pkl"),
        "models_dir": str(data_root / "index" / "models"),
    }
    mineru = {
        "mode": "local",
        "cli": cli_path,
        "method": method,
        "lang": lang,
        "timeout_sec": timeout,
        "fallback_to_pymupdf": True,
    }
    config_path.write_text(
        yaml.safe_dump(
            {"paths": paths, "mineru": mineru},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _model_directory(config_path: Path) -> Path | None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    configured = payload.get("models-dir")
    if not isinstance(configured, str) or not configured.strip():
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = cfg.PROJECT_ROOT / path
    return path.resolve()


def _run_demo(
    source_pdf: Path,
    *,
    paper_id: str,
    page_count: int,
    first_page_text: str,
    cli_path: str,
    method: str,
    lang: str | None,
    timeout: int,
    data_root: Path,
    config_path: Path,
    persistent: bool,
) -> Path:
    print("[1/6] 检查真实 PDF 与 MinerU 运行入口")
    print(f"      source={source_pdf}")
    print(f"      bytes={source_pdf.stat().st_size}")
    print(f"      pages={page_count}")
    print(f"      first_page_text={first_page_text[:160]!r}")
    print(f"      cli={cli_path}")

    print("[2/6] 检查项目自己的 MinerU 配置和模型目录")
    mineru_config = (cfg.PROJECT_ROOT / "config" / "magic-pdf.json").resolve()
    if not mineru_config.is_file():
        raise FileNotFoundError(
            f"MinerU 配置不存在: {mineru_config}\n"
            "请先在重写项目中创建 config/magic-pdf.json。"
        )
    model_dir = _model_directory(mineru_config)
    print(f"      config={mineru_config}")
    print(f"      models_dir={model_dir or '<not-configured>'}")
    print(f"      models_dir_exists={bool(model_dir and model_dir.is_dir())}")

    print("[3/6] 加载隔离业务配置")
    _write_isolated_config(
        config_path,
        data_root,
        cli_path=cli_path,
        method=method,
        lang=lang,
        timeout=timeout,
    )
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    expected_dir = Path(config.paths.parsed_dir) / to_safe_dirname(paper_id)
    print(f"      paper_id={paper_id}")
    print(f"      method={config.mineru.method}")
    print(f"      lang={config.mineru.lang}")
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    decision = resolve_ocr_language(source_pdf, config.mineru.lang)
    print("      -- OCR 语言路由判断 --")
    print(f"      document_language={decision.document_language}")
    print(f"      mineru_language={decision.mineru_language}")
    print(f"      source={decision.source}")
    print(f"      reason={decision.reason}")

    print("[4/6] 调用生产解析器并等待真实 MinerU 完成")
    result_dir = parse_pdf(paper_id, source_pdf)
    assert result_dir == expected_dir
    markdown_path = result_dir / "paper.md"
    assert markdown_path.is_file()
    assert markdown_path.stat().st_size > 0
    print(f"      parsed_dir={result_dir}")
    print(f"      markdown_bytes={markdown_path.stat().st_size}")

    print("[5/6] 独立检查标准化产物")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "\x00" not in markdown
    local_figure_refs = re.findall(r"\]\((figures/[^)]+)\)", markdown)
    for relative_path in local_figure_refs:
        assert (result_dir / relative_path).is_file(), relative_path
    figures_dir = result_dir / "figures"
    figure_files = [path for path in figures_dir.iterdir() if path.is_file()]
    raw_files = [path for path in (result_dir / "_mineru_raw").rglob("*") if path.is_file()]
    layout_path = result_dir / "layout.json"
    if layout_path.exists():
        json.loads(layout_path.read_text(encoding="utf-8"))
    language_path = result_dir / "language.json"
    assert language_path.is_file(), "缺少 language.json"
    language_payload = json.loads(language_path.read_text(encoding="utf-8"))
    assert language_payload["mineru_language"] in {"ch", "en"}
    print(f"      raw_files={len(raw_files)}")
    print(f"      figures={len(figure_files)}")
    print(f"      figure_references={len(local_figure_refs)}")
    print(f"      layout_json={layout_path.exists()}")
    print(f"      language_json={language_payload}")

    print("[6/6] 展示 Markdown 开头并汇总")
    preview = " ".join(markdown[:500].split())
    print(f"      markdown_preview={preview!r}")
    print(f"      collected_dir={result_dir}")
    print(f"      will_be_preserved={persistent}")
    print("\n真实 MinerU 解析 Demo 验收通过。")
    return result_dir


def main() -> None:
    args = _parse_args()
    source_pdf = _resolve_pdf_path(args.pdf_path)
    page_count, first_page_text = _inspect_pdf(source_pdf)
    paper_id = args.paper_id or make_paper_id(pdf_path=source_pdf)
    requested_cli = args.cli or cfg.load().mineru.cli
    cli_path = _resolve_cli(requested_cli)
    if cli_path is None:
        raise RuntimeError(
            f"找不到 MinerU CLI: {requested_cli}\n"
            "请执行: uv sync --extra dev --extra ingest --extra mineru"
        )

    original_config = os.environ.get("PAPER_RAG_CONFIG")
    try:
        with tempfile.TemporaryDirectory(prefix="paper-rag-mineru-config-") as config_temp:
            config_path = Path(config_temp) / "demo-config.yaml"
            if args.output_root is not None:
                data_root = args.output_root.expanduser().resolve()
                data_root.mkdir(parents=True, exist_ok=True)
                result_dir = _run_demo(
                    source_pdf,
                    paper_id=paper_id,
                    page_count=page_count,
                    first_page_text=first_page_text,
                    cli_path=cli_path,
                    method=args.method,
                    lang=args.lang,
                    timeout=args.timeout,
                    data_root=data_root,
                    config_path=config_path,
                    persistent=True,
                )
                print(f"解析结果已保留: {result_dir}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="paper-rag-mineru-demo-"
                ) as output_temp:
                    _run_demo(
                        source_pdf,
                        paper_id=paper_id,
                        page_count=page_count,
                        first_page_text=first_page_text,
                        cli_path=cli_path,
                        method=args.method,
                        lang=args.lang,
                        timeout=args.timeout,
                        data_root=Path(output_temp) / "data",
                        config_path=config_path,
                        persistent=False,
                    )
                print("临时解析结果已清理, 默认 data/ 未被修改。")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config


if __name__ == "__main__":
    main()
