"""使用真实 Semantic Scholar Graph API 演示论文采集。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import httpx
import yaml

from paper_rag import config as cfg
from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import to_safe_dirname


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("查询真实 Semantic Scholar API, 验证元数据落盘, 并在可用时检查开放 PDF。"),
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help=("arXiv 链接、doi:...、裸 DOI 或 S2 Paper ID; 未提供时进入交互输入"),
    )
    parser.add_argument(
        "--api-key",
        help="可选 Semantic Scholar API key; 推荐改用 S2_API_KEY 环境变量",
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
        identifier = input("请输入 arXiv 链接、doi:...、裸 DOI 或 S2 Paper ID: ").strip()

    if not identifier:
        raise ValueError("Semantic Scholar 标识符不能为空")
    return identifier


def _resolve_api_key(argument: str | None) -> str | None:
    return argument or os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


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
        raise RuntimeError("PyMuPDF 未安装, 请执行: uv sync --extra dev --extra ingest") from exc

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Semantic Scholar 返回的内容不是可打开的 PDF: {pdf_path}") from exc

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


def _fetch_real_paper(
    identifier: str,
    *,
    api_key: str | None,
) -> FetchResult:
    from paper_rag.ingest.semantic_scholar_source import SemanticScholarSource

    try:
        return SemanticScholarSource(api_key=api_key).fetch(identifier)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {401, 403, 429}:
            raise RuntimeError(
                "Semantic Scholar 拒绝或限制了请求。请设置 S2_API_KEY 后重试。"
            ) from exc
        raise


def _run_demo(
    identifier: str,
    *,
    api_key: str | None,
    data_root: Path,
    config_path: Path,
    persistent: bool,
) -> Path:
    print("[1/6] 读取输入和凭据状态")
    print(f"      input={identifier}")
    print(f"      api_key_configured={api_key is not None}")

    print("[2/6] 加载隔离配置")
    _write_isolated_config(config_path, data_root)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    cfg.load.cache_clear()
    config = cfg.load()
    assert Path(config.paths.data_root) == data_root
    print(f"      data_root={config.paths.data_root}")
    print(f"      persistent={persistent}")

    print("[3/6] 查询真实 Semantic Scholar Graph API")
    result = _fetch_real_paper(identifier, api_key=api_key)
    target_dir = Path(config.paths.papers_dir) / to_safe_dirname(result.meta.paper_id)
    assert result.meta.paper_id
    assert result.meta.source == "semantic_scholar"
    assert result.meta.title
    assert target_dir.is_dir()
    if result.meta.arxiv_id:
        assert result.meta.paper_id.startswith("arxiv:")
    elif result.meta.doi:
        assert result.meta.paper_id.startswith("doi:")
    else:
        assert result.meta.paper_id.startswith("s2:")
    print(f"      paper_id={result.meta.paper_id}")
    print(f"      title={result.meta.title}")
    print(f"      authors={', '.join(result.meta.authors[:5]) or '<none>'}")
    print(f"      year={result.meta.year or '<unknown>'}")
    print(f"      venue={result.meta.venue or '<unknown>'}")
    print(f"      doi={result.meta.doi or '<none>'}")
    print(f"      arxiv_id={result.meta.arxiv_id or '<none>'}")
    print(f"      abstract={((result.meta.abstract or '')[:160])!r}")
    for url in result.meta.urls:
        print(f"      url={url}")

    print("[4/6] 验证标准元数据和审计文件")
    meta_path = target_dir / "meta.json"
    source_path = target_dir / "source.txt"
    assert json.loads(meta_path.read_text(encoding="utf-8")) == result.meta.model_dump(mode="json")
    assert source_path.read_text(encoding="utf-8") == (
        f"source=semantic_scholar\nquery={identifier}\n"
    )
    print(f"      meta.json={meta_path.stat().st_size} bytes")
    print(f"      source.txt={source_path.stat().st_size} bytes")

    print("[5/6] 检查可选开放 PDF")
    if result.pdf_path:
        pdf_path = Path(result.pdf_path)
        page_count, preview = _inspect_pdf(pdf_path)
        assert pdf_path == target_dir / "raw.pdf"
        assert pdf_path.stat().st_size > 0
        print("      mode=metadata+pdf")
        print(f"      pdf_path={pdf_path}")
        print(f"      bytes={pdf_path.stat().st_size}")
        print(f"      pages={page_count}")
        print(f"      first_page={preview!r}")
    else:
        assert not (target_dir / "raw.pdf").exists()
        print("      mode=metadata-only")
        print("      Semantic Scholar 没有提供开放 PDF")

    print("[6/6] 验证输出目录隔离")
    assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
    for path in sorted(target_dir.iterdir()):
        print(f"      {path.name}: {path.stat().st_size} bytes")
    print(f"      collected_dir={target_dir}")
    print(f"      will_be_preserved={persistent}")
    print("\n真实 SemanticScholarSource Demo 验收通过。")
    return target_dir


def main() -> None:
    args = _parse_args()
    identifier = _read_identifier(args.identifier)
    api_key = _resolve_api_key(args.api_key)
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-semantic-scholar-config-"
        ) as config_temp:
            config_path = Path(config_temp) / "demo-config.yaml"

            if args.output_root is not None:
                data_root = args.output_root.expanduser().resolve()
                data_root.mkdir(parents=True, exist_ok=True)
                target_dir = _run_demo(
                    identifier,
                    api_key=api_key,
                    data_root=data_root,
                    config_path=config_path,
                    persistent=True,
                )
                print(f"采集结果已保留: {target_dir}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="paper-rag-semantic-scholar-demo-"
                ) as output_temp:
                    _run_demo(
                        identifier,
                        api_key=api_key,
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
