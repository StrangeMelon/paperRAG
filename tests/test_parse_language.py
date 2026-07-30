"""MinerU OCR 语言决策的纯本地边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_rag.parse.language import resolve_ocr_language


def _write_pdf(path: Path, text: str, *, font_name: str = "helv") -> None:
    import fitz

    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text, fontname=font_name, fontsize=12)
    document.save(path)
    document.close()


def test_metadata_language_has_priority(tmp_path: Path) -> None:
    pdf = tmp_path / "raw.pdf"
    _write_pdf(pdf, "English text " * 20)
    (tmp_path / "meta.json").write_text(
        json.dumps({"language": "zh"}),
        encoding="utf-8",
    )

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "zh"
    assert decision.mineru_language == "ch"
    assert decision.source == "metadata"
    assert decision.reason == "valid_meta_language"


def test_english_text_selects_english_model(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "Retrieval augmented generation research paper. " * 20)

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "en"
    assert decision.mineru_language == "en"
    assert decision.source == "pdf_text"
    assert decision.reason == "latin_text_detected"


def test_chinese_text_selects_bilingual_model(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "检索增强生成能够处理中文学术论文。" * 20, font_name="china-s")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "zh"
    assert decision.mineru_language == "ch"
    assert decision.source == "pdf_text"
    assert decision.reason == "cjk_text_detected"


def test_blank_scanned_pdf_falls_back_without_raising(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf, "")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language is None
    assert decision.mineru_language == "ch"
    assert decision.source == "fallback"
    assert decision.reason == "no_extractable_text"


def test_malformed_metadata_is_ignored_and_pdf_text_is_used(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "Retrieval augmented generation research paper. " * 20)
    (tmp_path / "meta.json").write_text("{broken", encoding="utf-8")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "en"
    assert decision.mineru_language == "en"
    assert decision.source == "pdf_text"
    assert decision.reason == "latin_text_detected"


def test_damaged_pdf_falls_back_to_bilingual_model(tmp_path: Path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a real PDF")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language is None
    assert decision.mineru_language == "ch"
    assert decision.source == "fallback"
    assert decision.reason.startswith("pdf_text_error:")


@pytest.mark.parametrize(
    ("configured_language", "document_language", "mineru_language", "reason"),
    [
        ("ch", "zh", "ch", "forced_ch"),
        ("en", "en", "en", "forced_en"),
    ],
)
def test_forced_language_modes_bypass_auto_detection(
    tmp_path: Path,
    configured_language: str,
    document_language: str,
    mineru_language: str,
    reason: str,
) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a real PDF")

    decision = resolve_ocr_language(pdf, configured_language)

    assert decision.document_language == document_language
    assert decision.mineru_language == mineru_language
    assert decision.source == "forced"
    assert decision.reason == reason


def test_unknown_configured_language_is_rejected(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "English text " * 20)

    with pytest.raises(ValueError, match="unsupported OCR language mode"):
        resolve_ocr_language(pdf, "fr")
