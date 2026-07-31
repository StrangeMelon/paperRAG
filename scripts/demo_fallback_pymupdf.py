"""使用用户提供的真实 PDF 演示 PyMuPDF 兜底解析。"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import yaml

from paper_rag import config as cfg
from paper_rag.utils.ids import make_paper_id, to_safe_dirname


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PyMuPDF 将真实 PDF 按页解析为 paper.md。",
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="待解析 PDF 的本地路径",
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
    return parser.parse_args()


def _read_pdf_pages(pdf_path: Path) -> list[str]:
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
        return [
            (page.get_text("text") or "").replace("\x00", "").strip()
            for page in document
        ]
    finally:
        document.close()


def _write_isolated_config(config_path: Path, data_root: Path) -> None:
    paths = {
        "data_root": str(data_root),
        "papers_dir": str(data_root / "papers"),
        "parsed_dir": str(data_root / "parsed"),
        "index_dir": str(data_root / "index"),
        "sqlite_path": str(data_root / "index" / "papers.sqlite"),
        "bm25_path": str(data_root / "index" / "bm25.pkl"),
        "models_dir": str(data_root / "index" / "models"),
    }
    config_path.write_text(
        yaml.safe_dump({"paths": paths}, sort_keys=False),
        encoding="utf-8",
    )


def _run_demo(
    source_pdf: Path,
    *,
    paper_id: str,
    page_texts: list[str],
    data_root: Path,
    config_path: Path,
    persistent: bool,
) -> Path:
    print("[1/6] 检查你提供的真实 PDF")
    extracted_pages = sum(bool(text) for text in page_texts)
    extracted_chars = sum(len(text) for text in page_texts)
    first_text = next((text for text in page_texts if text), "")
    print(f"      source={source_pdf}")
    print(f"      bytes={source_pdf.stat().st_size}")
    print(f"      pages={len(page_texts)}")
    print(f"      pages_with_text={extracted_pages}")
    print(f"      extracted_characters={extracted_chars}")
    print(f"      first_text={first_text[:160]!r}")

    print("[2/6] 加载隔离配置并确定论文 ID")
    _write_isolated_config(config_path, data_root)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    assert Path(config.paths.data_root) == data_root
    print(f"      paper_id={paper_id}")
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    from paper_rag.parse.fallback_pymupdf import parse_pdf

    print("[3/6] 使用生产解析器生成 Markdown")
    result_dir = parse_pdf(paper_id, source_pdf)
    expected_dir = (
        Path(config.paths.parsed_dir) / to_safe_dirname(paper_id)
    )
    markdown_path = result_dir / "paper.md"
    assert result_dir == expected_dir
    assert markdown_path.is_file()
    print(f"      parsed_dir={result_dir}")
    print(f"      markdown={markdown_path}")
    print(f"      markdown_bytes={markdown_path.stat().st_size}")

    print("[4/6] 验证页标记与逐页文本")
    markdown = markdown_path.read_text(encoding="utf-8")
    markers = re.findall(
        r"^<!-- page \d+ -->$",
        markdown,
        flags=re.MULTILINE,
    )
    assert len(markers) == len(page_texts)
    for page_number, text in enumerate(page_texts, start=1):
        marker = f"<!-- page {page_number} -->"
        assert marker in markdown
        if text:
            assert text in markdown
        print(
            f"      page={page_number} "
            f"text_characters={len(text)}"
        )

    print("[5/6] 重复解析并验证输出稳定")
    second_dir = parse_pdf(paper_id, str(source_pdf))
    assert second_dir == result_dir
    assert markdown_path.read_text(encoding="utf-8") == markdown
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "paper.md"
    ]
    print(f"      reused_dir={second_dir}")
    print("      markdown_content_unchanged=True")

    print("[6/6] 汇总解析能力和输出策略")
    mode = "text" if extracted_pages else "page-markers-only"
    print(f"      mode={mode}")
    print(f"      collected_dir={result_dir}")
    print(f"      will_be_preserved={persistent}")
    if not extracted_pages:
        print("      该 PDF 可能是扫描件, 需要 MinerU OCR 才能提取正文")
    print("\n真实 PyMuPDF 兜底解析 Demo 验收通过。")
    return result_dir


def main() -> None:
    args = _parse_args()
    source_pdf = args.pdf_path.expanduser().resolve()
    if not source_pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")

    page_texts = _read_pdf_pages(source_pdf)
    paper_id = args.paper_id or make_paper_id(pdf_path=source_pdf)
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-pymupdf-config-"
        ) as config_temp:
            config_path = Path(config_temp) / "demo-config.yaml"

            if args.output_root is not None:
                data_root = args.output_root.expanduser().resolve()
                data_root.mkdir(parents=True, exist_ok=True)
                result_dir = _run_demo(
                    source_pdf,
                    paper_id=paper_id,
                    page_texts=page_texts,
                    data_root=data_root,
                    config_path=config_path,
                    persistent=True,
                )
                print(f"解析结果已保留: {result_dir}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="paper-rag-pymupdf-demo-"
                ) as output_temp:
                    _run_demo(
                        source_pdf,
                        paper_id=paper_id,
                        page_texts=page_texts,
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
