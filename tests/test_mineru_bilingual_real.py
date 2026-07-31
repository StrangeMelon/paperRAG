"""MinerU 中英文真实 GPU OCR 的无 mock 集成测试。

需要真实模型、CUDA GPU 和两份真实 PDF; 通过环境变量提供:

    PAPER_RAG_REAL_ENGLISH_PDF=/absolute/english.pdf   # 普通英文 PDF, 无语言元数据
    PAPER_RAG_REAL_CHINESE_PDF=/absolute/chinese.pdf   # 中文扫描件, 人工标注 zh

缺少任一变量或文件时明确失败, 不 skip。单独运行:

    PAPER_RAG_REAL_ENGLISH_PDF=... PAPER_RAG_REAL_CHINESE_PDF=... \
        uv run pytest -vv -s tests/test_mineru_bilingual_real.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from paper_rag import config as cfg
from paper_rag.parse import mineru_local

ENGLISH_ENV = "PAPER_RAG_REAL_ENGLISH_PDF"
CHINESE_ENV = "PAPER_RAG_REAL_CHINESE_PDF"


def _require_real_pdf(env_var: str) -> Path:
    value = os.environ.get(env_var, "").strip()
    if not value:
        pytest.fail(
            f"缺少环境变量 {env_var}; 请设置为真实 PDF 的绝对路径后重跑该真实测试。",
            pytrace=False,
        )
    pdf_path = Path(value).expanduser().resolve()
    if not pdf_path.is_file():
        pytest.fail(f"{env_var} 指向的文件不存在: {pdf_path}", pytrace=False)
    return pdf_path


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
    mineru = {
        "mode": "local",
        "cli": "magic-pdf",
        "method": "ocr",
        "lang": "auto",
        "timeout_sec": 1200,
        "fallback_to_pymupdf": True,
    }
    config_path.write_text(
        yaml.safe_dump({"paths": paths, "mineru": mineru}, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture
def isolated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """使用隔离的业务配置和临时 data_root, 但复用真实 magic-pdf.json 与模型。"""

    data_root = tmp_path / "data"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_root)
    monkeypatch.setenv("PAPER_RAG_CONFIG", str(config_path))
    cfg.load.cache_clear()
    try:
        yield data_root
    finally:
        cfg.load.cache_clear()


def _prepare_pdf(source: Path, work_dir: Path, *, language: str | None) -> Path:
    """把真实 PDF 复制到隔离目录, 可选写出人工语言 meta.json。"""

    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = work_dir / "raw.pdf"
    shutil.copyfile(source, pdf_path)
    if language is not None:
        (work_dir / "meta.json").write_text(
            json.dumps({"language": language}, ensure_ascii=False),
            encoding="utf-8",
        )
    return pdf_path


def test_real_english_pdf_selects_english_model_from_text(
    isolated_config: Path,
    tmp_path: Path,
) -> None:
    source = _require_real_pdf(ENGLISH_ENV)
    print(f"[1/3] 准备英文普通 PDF (无语言元数据): {source}")
    pdf_path = _prepare_pdf(source, tmp_path / "english", language=None)

    print("[2/3] 调用真实 MinerU GPU OCR")
    output_dir = mineru_local.parse_pdf("real-english", pdf_path)

    print("[3/3] 校验标准化产物与语言决策")
    markdown = output_dir / "paper.md"
    assert markdown.is_file()
    assert markdown.stat().st_size > 100
    payload = json.loads((output_dir / "language.json").read_text(encoding="utf-8"))
    print(f"      language={payload}")
    assert payload["mineru_language"] == "en"
    assert payload["source"] == "pdf_text"


def test_real_chinese_scanned_pdf_selects_chinese_model_from_metadata(
    isolated_config: Path,
    tmp_path: Path,
) -> None:
    source = _require_real_pdf(CHINESE_ENV)
    print(f"[1/3] 准备中文扫描 PDF (人工语言元数据 zh): {source}")
    pdf_path = _prepare_pdf(source, tmp_path / "chinese", language="zh")

    print("[2/3] 调用真实 MinerU GPU OCR")
    output_dir = mineru_local.parse_pdf("real-chinese", pdf_path)

    print("[3/3] 校验标准化产物与语言决策")
    markdown = output_dir / "paper.md"
    assert markdown.is_file()
    assert markdown.stat().st_size > 100
    payload = json.loads((output_dir / "language.json").read_text(encoding="utf-8"))
    print(f"      language={payload}")
    assert payload["mineru_language"] == "ch"
    assert payload["source"] == "metadata"
