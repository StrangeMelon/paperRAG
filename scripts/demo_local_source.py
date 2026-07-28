"""使用真实 PDF 和隔离数据目录演示本地论文采集。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from paper_rag import config as cfg
from paper_rag.utils.ids import make_paper_id, to_safe_dirname


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集你提供的真实 PDF, 并验证 LocalSource 的完整落盘结果。",
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="待采集 PDF 的本地路径",
    )
    parser.add_argument(
        "--title",
        help="可选论文标题; 默认使用 PDF 文件名",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="可选持久化 data_root; 未提供时使用临时目录并自动清理",
    )
    return parser.parse_args()


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
            "PyMuPDF 未安装, 请先执行: uv sync --extra dev --extra ingest"
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
        text = " ".join(document[0].get_text("text").split())
        preview = text[:120] if text else "<no extractable text>"
        return document.page_count, preview
    finally:
        document.close()


def _run_demo(
    source_pdf: Path,
    *,
    title: str | None,
    data_root: Path,
    config_path: Path,
    persistent: bool,
) -> Path:
    print("[1/6] 检查你提供的真实 PDF")
    page_count, preview = _inspect_pdf(source_pdf)
    print(f"      source={source_pdf}")
    print(f"      bytes={source_pdf.stat().st_size} pages={page_count}")
    print(f"      first_page={preview!r}")

    print("[2/6] 加载隔离配置")
    _write_isolated_config(config_path, data_root)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    assert Path(config.paths.data_root) == data_root
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    from paper_rag.ingest.local_source import LocalSource

    print("[3/6] 第一次采集本地 PDF")
    result = LocalSource(title=title).fetch(str(source_pdf))
    expected_id = make_paper_id(pdf_path=source_pdf)
    expected_title = title or source_pdf.stem
    assert result.meta.paper_id == expected_id
    assert result.meta.title == expected_title
    assert result.meta.source == "local"
    print(f"      paper_id={result.meta.paper_id}")
    print(f"      title={result.meta.title}")
    print(f"      pdf_path={result.pdf_path}")

    print("[4/6] 检查规范目录与审计文件")
    target_dir = Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
    copied_pdf = target_dir / "raw.pdf"
    meta_path = target_dir / "meta.json"
    source_path = target_dir / "source.txt"
    assert Path(result.pdf_path) == copied_pdf
    assert copied_pdf.read_bytes() == source_pdf.read_bytes()
    copied_pages, copied_preview = _inspect_pdf(copied_pdf)
    assert copied_pages == page_count
    assert copied_preview == preview
    persisted_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted_meta == result.meta.model_dump(mode="json")
    assert source_path.read_text(encoding="utf-8") == (
        f"source=local\nquery={source_pdf}\n"
    )
    for path in sorted(target_dir.iterdir()):
        print(f"      {path.name}: {path.stat().st_size} bytes")

    print("[5/6] 重复采集同一 PDF")
    second = LocalSource(title=title).fetch(str(source_pdf))
    assert second.meta.paper_id == result.meta.paper_id
    assert second.pdf_path == result.pdf_path
    paper_dirs = list(Path(config.paths.papers_dir).iterdir())
    assert paper_dirs.count(target_dir) == 1
    print(f"      reused={second.pdf_path}")
    print(f"      paper_directories={len(paper_dirs)}")

    print("[6/6] 验证内容哈希与输出策略")
    assert expected_id.startswith("sha1:")
    assert target_dir.is_dir()
    print(f"      content_hash_id={expected_id}")
    print(f"      collected_dir={target_dir}")
    print(f"      will_be_preserved={persistent}")
    print("\n真实 LocalSource Demo 验收通过。")
    return target_dir


def main() -> None:
    args = _parse_args()
    source_pdf = args.pdf_path.expanduser().resolve()
    if not source_pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")

    original_config = os.environ.get("PAPER_RAG_CONFIG")
    try:
        with tempfile.TemporaryDirectory(prefix="paper-rag-local-config-") as config_temp:
            config_path = Path(config_temp) / "demo-config.yaml"

            if args.output_root is not None:
                data_root = args.output_root.expanduser().resolve()
                data_root.mkdir(parents=True, exist_ok=True)
                target_dir = _run_demo(
                    source_pdf,
                    title=args.title,
                    data_root=data_root,
                    config_path=config_path,
                    persistent=True,
                )
                print(f"采集结果已保留: {target_dir}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="paper-rag-local-demo-"
                ) as output_temp:
                    _run_demo(
                        source_pdf,
                        title=args.title,
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
