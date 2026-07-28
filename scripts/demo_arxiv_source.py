"""使用真实 arXiv API 和 PDF 演示论文采集。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from paper_rag import config as cfg
from paper_rag.utils.ids import (
    make_paper_id,
    normalize_arxiv,
    split_arxiv_version,
    to_safe_dirname,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查询真实 arXiv 服务、下载论文 PDF, 并验证完整落盘结果。",
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help=(
            "arXiv ID、摘要页链接或 PDF 链接; "
            "未提供时进入交互输入"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="可选持久化 data_root; 未提供时使用临时目录并自动清理",
    )
    return parser.parse_args()


def _read_identifier(argument: str | None) -> str:
    if argument is not None:
        identifier = argument.strip()
    else:
        identifier = input(
            "请输入 arXiv ID、摘要页链接或 PDF 下载链接: "
        ).strip()

    if not identifier:
        raise ValueError("arXiv 标识符不能为空")
    if normalize_arxiv(identifier) is None:
        raise ValueError(
            "无法识别 arXiv ID; 示例: 1706.03762 或 "
            "https://arxiv.org/pdf/1706.03762.pdf"
        )
    return identifier


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
        raise ValueError(f"下载内容不是可打开的 PDF: {pdf_path}") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"下载的 PDF 需要密码: {pdf_path}")
        if document.page_count < 1:
            raise ValueError(f"下载的 PDF 没有页面: {pdf_path}")
        text = " ".join(document[0].get_text("text").split())
        preview = text[:160] if text else "<no extractable text>"
        return document.page_count, preview
    finally:
        document.close()


def _run_demo(
    identifier: str,
    *,
    data_root: Path,
    config_path: Path,
    persistent: bool,
) -> Path:
    normalized = normalize_arxiv(identifier)
    assert normalized is not None
    _, requested_version = split_arxiv_version(identifier)

    print("[1/6] 识别你输入的 arXiv 链接或 ID")
    print(f"      input={identifier}")
    print(f"      normalized_id={normalized}")
    print(f"      requested_version={requested_version or '<latest>'}")

    print("[2/6] 加载隔离配置")
    _write_isolated_config(config_path, data_root)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    assert Path(config.paths.data_root) == data_root
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    from paper_rag.ingest.arxiv_source import ArxivSource

    print("[3/6] 查询真实 arXiv API 并下载 PDF")
    result = ArxivSource().fetch(identifier)
    downloaded_pdf = Path(result.pdf_path)
    expected_id = make_paper_id(arxiv_id=normalized)
    assert result.meta.paper_id == expected_id
    assert result.meta.arxiv_id == normalized
    assert result.meta.source == "arxiv"
    assert result.meta.venue == "arXiv"
    assert result.meta.extra == (
        {"arxiv_version": requested_version}
        if requested_version
        else {}
    )
    print(f"      paper_id={result.meta.paper_id}")
    print(f"      title={result.meta.title}")
    print(f"      authors={', '.join(result.meta.authors)}")
    print(f"      year={result.meta.year} doi={result.meta.doi or '<none>'}")
    print(f"      entry_url={result.meta.urls[0]}")
    print(f"      pdf_url={result.meta.urls[1]}")

    print("[4/6] 验证真实 PDF 内容")
    page_count, preview = _inspect_pdf(downloaded_pdf)
    print(f"      pdf_path={downloaded_pdf}")
    print(f"      bytes={downloaded_pdf.stat().st_size}")
    print(f"      pages={page_count}")
    print(f"      first_page={preview!r}")

    print("[5/6] 检查标准目录、元数据与采集审计文件")
    target_dir = Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
    meta_path = target_dir / "meta.json"
    source_path = target_dir / "source.txt"
    assert downloaded_pdf == target_dir / "raw.pdf"
    assert json.loads(meta_path.read_text(encoding="utf-8")) == (
        result.meta.model_dump(mode="json")
    )
    assert source_path.read_text(encoding="utf-8") == (
        f"source=arxiv\nquery={identifier}\n"
    )
    for path in sorted(target_dir.iterdir()):
        print(f"      {path.name}: {path.stat().st_size} bytes")

    print("[6/6] 重复采集并验证 PDF 复用")
    original_size = downloaded_pdf.stat().st_size
    original_mtime = downloaded_pdf.stat().st_mtime_ns
    second = ArxivSource().fetch(identifier)
    assert second.meta.paper_id == result.meta.paper_id
    assert second.pdf_path == result.pdf_path
    assert downloaded_pdf.stat().st_size == original_size
    assert downloaded_pdf.stat().st_mtime_ns == original_mtime
    paper_dirs = list(Path(config.paths.papers_dir).iterdir())
    assert paper_dirs == [target_dir]
    print(f"      reused={second.pdf_path}")
    print(f"      paper_directories={len(paper_dirs)}")
    print(f"      collected_dir={target_dir}")
    print(f"      will_be_preserved={persistent}")
    print("\n真实 ArxivSource Demo 验收通过。")
    return target_dir


def main() -> None:
    args = _parse_args()
    identifier = _read_identifier(args.identifier)
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-arxiv-config-"
        ) as config_temp:
            config_path = Path(config_temp) / "demo-config.yaml"

            if args.output_root is not None:
                data_root = args.output_root.expanduser().resolve()
                data_root.mkdir(parents=True, exist_ok=True)
                target_dir = _run_demo(
                    identifier,
                    data_root=data_root,
                    config_path=config_path,
                    persistent=True,
                )
                print(f"采集结果已保留: {target_dir}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="paper-rag-arxiv-demo-"
                ) as output_temp:
                    _run_demo(
                        identifier,
                        data_root=Path(output_temp) / "data",
                        config_path=config_path,
                        persistent=False,
                    )
                print("临时采集结果已清理, 默认 data/ 未被修改。")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config


if __name__ == "__main__":
    main()
