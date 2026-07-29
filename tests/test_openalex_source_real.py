"""OpenAlexSource 的无 mock 真实公网集成测试。"""

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

METADATA_ONLY_ID = "W2741809807"
METADATA_ONLY_DOI = "10.7717/peerj.4375"
PDF_WORK_ID = "W3038568908"
PDF_WORK_DOI = "10.1585/pfr.15.2402039"
PDF_URL_FRAGMENT = "jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf"


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
        return document.page_count, text[:160]
    finally:
        document.close()


def test_real_openalex_source_persists_metadata_without_pdf() -> None:
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-openalex-metadata-real-"
        ) as temp_name:
            demo_root = Path(temp_name)
            data_root = demo_root / "data"
            config_path = demo_root / "config.yaml"

            print("[1/4] 通过真实 YAML 加载隔离配置")
            _write_config(config_path, data_root)
            os.environ["PAPER_RAG_CONFIG"] = str(config_path)
            cfg.load.cache_clear()
            config = cfg.load()
            assert Path(config.paths.data_root) == data_root
            print(f"      data_root={config.paths.data_root}")

            from paper_rag.ingest.openalex_source import OpenAlexSource

            print("[2/4] 查询真实 OpenAlex Work 元数据")
            print(f"      work_id={METADATA_ONLY_ID}")
            result = OpenAlexSource().fetch(METADATA_ONLY_ID)
            expected_id = make_paper_id(doi=METADATA_ONLY_DOI)
            target_dir = (
                Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
            )
            assert result.meta.paper_id == expected_id
            assert result.meta.doi == METADATA_ONLY_DOI
            assert "state of OA" in result.meta.title
            assert result.meta.authors
            assert result.meta.year == 2018
            assert result.meta.source == "openalex"
            assert result.pdf_path == ""
            print(f"      paper_id={result.meta.paper_id}")
            print(f"      title={result.meta.title}")
            print(f"      doi={result.meta.doi}")

            print("[3/4] 验证 metadata-only 产物")
            assert target_dir.is_dir()
            assert not (target_dir / "raw.pdf").exists()
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == result.meta.model_dump(mode="json")
            assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
                f"source=openalex\nquery={METADATA_ONLY_ID}\n"
            )
            assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
            print(f"      target={target_dir}")
            print("      pdf_path=<empty>")

            print("[4/4] 确认临时数据目录仍然隔离")
            assert demo_root.is_dir()
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config

    assert not demo_root.exists()
    print("真实 OpenAlex metadata-only 集成测试通过。")


def test_real_openalex_source_downloads_open_pdf() -> None:
    fitz = _fitz_module()
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-openalex-pdf-real-"
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

            from paper_rag.ingest.openalex_source import OpenAlexSource

            print("[2/5] 查询真实 OpenAlex Work 并下载开放 PDF")
            print(f"      work_id={PDF_WORK_ID}")
            result = OpenAlexSource().fetch(PDF_WORK_ID)
            expected_id = make_paper_id(doi=PDF_WORK_DOI)
            target_dir = (
                Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
            )
            downloaded_pdf = Path(result.pdf_path)
            assert result.meta.paper_id == expected_id
            assert result.meta.doi == PDF_WORK_DOI
            assert result.meta.title
            assert result.meta.authors
            assert result.meta.source == "openalex"
            assert any(
                PDF_URL_FRAGMENT in url
                for url in result.meta.urls
            )
            assert downloaded_pdf == target_dir / "raw.pdf"
            print(f"      paper_id={result.meta.paper_id}")
            print(f"      title={result.meta.title}")
            print(f"      pdf_url={result.meta.urls[0]}")

            print("[3/5] 使用 PyMuPDF 打开真实 PDF")
            page_count, preview = _inspect_pdf(fitz, downloaded_pdf)
            assert downloaded_pdf.stat().st_size > 0
            print(f"      bytes={downloaded_pdf.stat().st_size}")
            print(f"      pages={page_count}")
            print(f"      first_page={preview!r}")

            print("[4/5] 验证标准元数据和审计文件")
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == result.meta.model_dump(mode="json")
            assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
                f"source=openalex\nquery={PDF_WORK_ID}\n"
            )
            assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
            print(f"      target={target_dir}")

            print("[5/5] 确认真实 PDF 已写入隔离目录")
            assert target_dir.is_dir()
            print("真实 OpenAlex PDF 集成测试通过。")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config

    assert not demo_root.exists()
    print("临时数据目录与配置已清理。")
