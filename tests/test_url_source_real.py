"""UrlSource 访问 ACL Anthology 的无 mock 公网集成测试。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from paper_rag import config as cfg
from paper_rag.utils.ids import make_paper_id, to_safe_dirname

ACL_PDF_URL = "https://aclanthology.org/2025.acl-long.426.pdf"


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
        return document.page_count, text[:120]
    finally:
        document.close()


def test_real_url_source_downloads_acl_anthology_pdf() -> None:
    fitz = _fitz_module()
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(prefix="paper-rag-url-real-") as temp_name:
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

            from paper_rag.ingest.url_source import UrlSource

            print("[2/5] 从 ACL Anthology 发起真实 HTTPS 下载")
            print(f"      url={ACL_PDF_URL}")
            result = UrlSource().fetch(ACL_PDF_URL)
            downloaded_pdf = Path(result.pdf_path)
            print(f"      paper_id={result.meta.paper_id}")
            print(f"      downloaded={downloaded_pdf}")

            print("[3/5] 使用 PyMuPDF 打开下载结果")
            page_count, preview = _inspect_pdf(fitz, downloaded_pdf)
            assert downloaded_pdf.stat().st_size > 0
            print(f"      bytes={downloaded_pdf.stat().st_size}")
            print(f"      pages={page_count}")
            print(f"      first_page={preview!r}")

            print("[4/5] 验证标准元数据与审计文件")
            expected_id = make_paper_id(pdf_path=downloaded_pdf)
            target_dir = (
                Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
            )
            assert result.meta.paper_id == expected_id
            assert result.meta.title == "2025.acl-long.426.pdf"
            assert result.meta.source == "url"
            assert result.meta.urls == [ACL_PDF_URL]
            assert downloaded_pdf == target_dir / "raw.pdf"
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == result.meta.model_dump(mode="json")
            assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
                f"source=url\nquery={ACL_PDF_URL}\n"
            )
            assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
            print(f"      target={target_dir}")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config

    print("[5/5] 临时目录与配置已清理")
    assert not demo_root.exists()
    print("真实 ACL Anthology URL 集成测试通过。")
