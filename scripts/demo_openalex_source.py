"""
使用真实 OpenAlex API 演示元数据采集和可选 PDF 下载
主要定位本来就是"元数据增强", 不是主要的 PDF 采集器
"""

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
        description=(
            "查询真实 OpenAlex API, 验证元数据落盘, "
            "并在可用时检查开放获取 PDF。"
        ),
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help=(
            "doi:...、OpenAlex ID 或 OpenAlex 链接; "
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
            "请输入 doi:...、OpenAlex ID 或 OpenAlex 链接: "
        ).strip()

    if not identifier:
        raise ValueError("OpenAlex 标识符不能为空")
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
        raise ValueError(
            f"OpenAlex 返回的内容不是可打开的 PDF: {pdf_path}"
        ) from exc

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
    print("[1/6] 读取你输入的 OpenAlex 标识符")
    print(f"      input={identifier}")

    print("[2/6] 加载隔离配置")
    _write_isolated_config(config_path, data_root)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    assert Path(config.paths.data_root) == data_root
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    from paper_rag.ingest.openalex_source import OpenAlexSource

    print("[3/6] 查询真实 OpenAlex API")
    first = OpenAlexSource().fetch(identifier)
    target_dir = Path(config.paths.papers_dir) / to_safe_dirname(
        first.meta.paper_id
    )
    assert first.meta.paper_id
    assert first.meta.source == "openalex"
    assert first.meta.title
    assert target_dir.is_dir()
    print(f"      paper_id={first.meta.paper_id}")
    print(f"      title={first.meta.title}")
    print(f"      authors={', '.join(first.meta.authors[:5]) or '<none>'}")
    print(f"      year={first.meta.year or '<unknown>'}")
    print(f"      venue={first.meta.venue or '<unknown>'}")
    print(f"      doi={first.meta.doi or '<none>'}")
    print(f"      abstract={((first.meta.abstract or '')[:160])!r}")
    for url in first.meta.urls:
        print(f"      url={url}")

    print("[4/6] 验证标准元数据和审计文件")
    meta_path = target_dir / "meta.json"
    source_path = target_dir / "source.txt"
    assert json.loads(
        meta_path.read_text(encoding="utf-8")
    ) == first.meta.model_dump(mode="json")
    assert source_path.read_text(encoding="utf-8") == (
        f"source=openalex\nquery={identifier}\n"
    )
    print(f"      meta.json={meta_path.stat().st_size} bytes")
    print(f"      source.txt={source_path.stat().st_size} bytes")

    print("[5/6] 检查可选开放获取 PDF")
    pdf_mtime = None
    if first.pdf_path:
        pdf_path = Path(first.pdf_path)
        page_count, preview = _inspect_pdf(pdf_path)
        assert pdf_path == target_dir / "raw.pdf"
        assert pdf_path.stat().st_size > 0
        pdf_mtime = pdf_path.stat().st_mtime_ns
        print("      mode=metadata+pdf")
        print(f"      pdf_path={pdf_path}")
        print(f"      bytes={pdf_path.stat().st_size}")
        print(f"      pages={page_count}")
        print(f"      first_page={preview!r}")
    else:
        assert not (target_dir / "raw.pdf").exists()
        print("      mode=metadata-only")
        print("      OpenAlex 没有提供可下载的开放 PDF")

    print("[6/6] 重复查询并验证输出目录稳定")
    second = OpenAlexSource().fetch(identifier)
    assert second.meta.paper_id == first.meta.paper_id
    assert second.pdf_path == first.pdf_path
    if pdf_mtime is not None:
        assert Path(second.pdf_path).stat().st_mtime_ns == pdf_mtime
    assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
    print(f"      reused_target={target_dir}")
    print("\n真实 OpenAlexSource Demo 验收通过。")
    return target_dir


def main() -> None:
    args = _parse_args()
    identifier = _read_identifier(args.identifier)
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-openalex-config-"
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
                    prefix="paper-rag-openalex-demo-"
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
