"""ArxivSource 的无 mock 真实公网集成测试。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import pytest
import yaml

from paper_rag import config as cfg
from paper_rag.utils.ids import make_paper_id, normalize_arxiv, to_safe_dirname

ARXIV_PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
ARXIV_ID = "1706.03762"


def _fitz_module() -> ModuleType:
    try:
        import fitz
    except ImportError:
        pytest.fail(
            "PyMuPDF 未安装; 请执行: uv sync --extra dev --extra ingest",
            pytrace=False,
        )
    return fitz


def _write_config(config_path: Path, data_root: Path) -> None:
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


def _inspect_pdf(fitz: ModuleType, pdf_path: Path) -> tuple[int, str]:
    document = fitz.open(str(pdf_path))
    try:
        assert document.page_count > 0
        text = " ".join(document[0].get_text("text").split())
        assert text
        return document.page_count, text[:160]
    finally:
        document.close()


def test_real_arxiv_source_downloads_and_persists_paper() -> None:
    fitz = _fitz_module()
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-arxiv-real-"
        ) as temp_name:
            demo_root = Path(temp_name)
            data_root = demo_root / "data"
            config_path = demo_root / "config.yaml"

            print("[1/5] 通过真实 YAML 加载隔离配置")
            _write_config(config_path, data_root)
            os.environ["PAPER_RAG_CONFIG"] = str(config_path)
            cfg.load.cache_clear()
            config = cfg.load()
            assert Path(config.paths.data_root) == data_root
            print(f"      data_root={config.paths.data_root}")

            from paper_rag.ingest.arxiv_source import ArxivSource

            print("[2/5] 查询真实 arXiv API 并下载 PDF")
            print(f"      url={ARXIV_PDF_URL}")
            result = ArxivSource().fetch(ARXIV_PDF_URL)
            downloaded_pdf = Path(result.pdf_path)
            expected_id = make_paper_id(arxiv_id=ARXIV_ID)
            assert normalize_arxiv(ARXIV_PDF_URL) == ARXIV_ID
            assert result.meta.paper_id == expected_id
            assert result.meta.arxiv_id == ARXIV_ID
            assert result.meta.title
            assert result.meta.authors
            assert result.meta.year == 2017
            assert result.meta.venue == "arXiv"
            assert result.meta.source == "arxiv"
            entry_url = urlparse(result.meta.urls[0])
            pdf_url = urlparse(result.meta.urls[1])
            assert entry_url.scheme in {"http", "https"}
            assert pdf_url.scheme in {"http", "https"}
            assert entry_url.netloc == "arxiv.org"
            assert pdf_url.netloc == "arxiv.org"
            assert entry_url.path.startswith("/abs/")
            assert pdf_url.path.startswith("/pdf/")
            print(f"      paper_id={result.meta.paper_id}")
            print(f"      title={result.meta.title}")
            print(f"      authors={', '.join(result.meta.authors[:4])}")
            print(f"      downloaded={downloaded_pdf}")

            print("[3/5] 使用 PyMuPDF 打开真实下载结果")
            page_count, preview = _inspect_pdf(fitz, downloaded_pdf)
            assert downloaded_pdf.stat().st_size > 0
            print(f"      bytes={downloaded_pdf.stat().st_size}")
            print(f"      pages={page_count}")
            print(f"      first_page={preview!r}")

            print("[4/5] 验证标准元数据与采集审计文件")
            target_dir = (
                Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
            )
            assert downloaded_pdf == target_dir / "raw.pdf"
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == result.meta.model_dump(mode="json")
            assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
                f"source=arxiv\nquery={ARXIV_PDF_URL}\n"
            )
            assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
            print(f"      target={target_dir}")

            print("[5/5] 验证采集目录隔离")
            assert target_dir.is_dir()
            assert demo_root.is_dir()
            print("真实 ArxivSource 集成测试通过。")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config

    assert not demo_root.exists()
    print("临时目录与配置已清理。")
