"""LocalSource 的无 mock 真实集成测试。"""

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


def _fitz_module() -> ModuleType:
    try:
        import fitz
    except ImportError:
        pytest.fail(
            "PyMuPDF 未安装; 请执行: uv sync --extra dev --extra ingest",
            pytrace=False,
        )
    return fitz


def _create_real_pdf(fitz: ModuleType, pdf_path: Path) -> None:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Paper RAG LocalSource real integration test",
            fontsize=16,
        )
        document.save(str(pdf_path))
    finally:
        document.close()


def _read_first_page(fitz: ModuleType, pdf_path: Path) -> str:
    document = fitz.open(str(pdf_path))
    try:
        assert document.page_count == 1
        return document[0].get_text("text").strip()
    finally:
        document.close()


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


def test_real_local_source_ingests_pdf_idempotently() -> None:
    fitz = _fitz_module()
    original_config = os.environ.get("PAPER_RAG_CONFIG")

    try:
        with tempfile.TemporaryDirectory(prefix="paper-rag-local-real-") as temp_name:
            demo_root = Path(temp_name)
            source_pdf = demo_root / "uploaded-paper.pdf"
            data_root = demo_root / "data"
            config_path = demo_root / "config.yaml"

            print("[1/5] 创建并打开结构有效的真实 PDF")
            _create_real_pdf(fitz, source_pdf)
            source_text = _read_first_page(fitz, source_pdf)
            assert source_text == "Paper RAG LocalSource real integration test"
            print(f"      source={source_pdf}")
            print(f"      bytes={source_pdf.stat().st_size}")

            print("[2/5] 通过真实 YAML 加载隔离配置")
            _write_config(config_path, data_root)
            os.environ["PAPER_RAG_CONFIG"] = str(config_path)
            cfg.load.cache_clear()
            config = cfg.load()
            assert Path(config.paths.data_root) == data_root
            print(f"      data_root={config.paths.data_root}")

            from paper_rag.ingest.local_source import LocalSource

            print("[3/5] 第一次真实采集")
            first = LocalSource(title="Real Integration Paper").fetch(str(source_pdf))
            expected_id = make_paper_id(pdf_path=source_pdf)
            target_dir = Path(config.paths.papers_dir) / to_safe_dirname(expected_id)
            copied_pdf = target_dir / "raw.pdf"
            assert first.meta.paper_id == expected_id
            assert first.meta.title == "Real Integration Paper"
            assert first.meta.source == "local"
            assert first.meta.urls == [source_pdf.as_uri()]
            assert Path(first.pdf_path) == copied_pdf
            print(f"      paper_id={first.meta.paper_id}")
            print(f"      target={target_dir}")

            print("[4/5] 验证真实产物并重复采集")
            assert copied_pdf.read_bytes() == source_pdf.read_bytes()
            assert _read_first_page(fitz, copied_pdf) == source_text
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == first.meta.model_dump(mode="json")
            assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
                f"source=local\nquery={source_pdf}\n"
            )

            second = LocalSource(title="Real Integration Paper").fetch(str(source_pdf))
            assert second.meta.paper_id == first.meta.paper_id
            assert second.pdf_path == first.pdf_path
            assert list(Path(config.paths.papers_dir).iterdir()) == [target_dir]
            assert json.loads(
                (target_dir / "meta.json").read_text(encoding="utf-8")
            ) == second.meta.model_dump(mode="json")
            print(f"      reused={second.pdf_path}")
    finally:
        cfg.load.cache_clear()
        if original_config is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = original_config

    print("[5/5] 临时目录与配置已清理")
    assert not demo_root.exists()
    print("真实 LocalSource 集成测试通过。")
