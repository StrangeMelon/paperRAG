import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_PAGES = 5
MAX_CHARACTERS = 20_000
MIN_CJK_CHARACTERS = 20
MIN_LATIN_CHARACTERS = 50
MIN_CJK_RATIO = 0.05


@dataclass(frozen=True)
class OcrLanguageDecision:
    document_language: Literal["zh", "en"] | None
    mineru_language: Literal["ch", "en"]
    source: Literal["forced", "metadata", "pdf_text", "fallback"]
    reason: str
    model_fallback: bool = False


def resolve_ocr_language(
    pdf_path: str | Path,
    configured_language: Literal["auto", "ch", "en"] = "auto",
    *,
    meta_path: str | Path | None = None,
) -> OcrLanguageDecision:
    pdf = Path(pdf_path).expanduser().resolve()

    if configured_language == "ch":
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="forced",
            reason="forced_ch",
        )
    if configured_language == "en":
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="forced",
            reason="forced_en",
        )
    if configured_language != "auto":
        raise ValueError(f"unsupported OCR language mode: {configured_language}")

    resolved_meta_path = (
        Path(meta_path).expanduser().resolve()
        if meta_path is not None
        else pdf.parent / "meta.json"
    )
    metadata_language = _read_metadata_language(resolved_meta_path)
    if metadata_language == "zh":
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="metadata",
            reason="valid_meta_language",
        )
    if metadata_language == "en":
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="metadata",
            reason="valid_meta_language",
        )

    try:
        text = _sample_pdf_text(pdf)
    except Exception as exc:
        return OcrLanguageDecision(
            document_language=None,
            mineru_language="ch",
            source="fallback",
            reason=f"pdf_text_error:{type(exc).__name__}",
        )

    if not text.strip():
        return OcrLanguageDecision(
            document_language=None,
            mineru_language="ch",
            source="fallback",
            reason="no_extractable_text",
        )

    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin_count = sum(char.isascii() and char.isalpha() for char in text)
    language_characters = cjk_count + latin_count
    cjk_ratio = cjk_count / language_characters if language_characters else 0.0

    if cjk_count >= MIN_CJK_CHARACTERS and cjk_ratio >= MIN_CJK_RATIO:
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="pdf_text",
            reason="cjk_text_detected",
        )
    if latin_count >= MIN_LATIN_CHARACTERS:
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="pdf_text",
            reason="latin_text_detected",
        )
    return OcrLanguageDecision(
        document_language=None,
        mineru_language="ch",
        source="fallback",
        reason="insufficient_language_signal",
    )

def _read_metadata_language(
    meta_path: Path,
) -> Literal["zh", "en"] | None:
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    language = payload.get("language")
    return language if language in {"zh", "en"} else None


def _sample_pdf_text(pdf_path: Path) -> str:
    import fitz

    document = fitz.open(str(pdf_path))
    parts: list[str] = []
    characters = 0
    try:
        for page_index in range(min(len(document), MAX_PAGES)):
            page_text = document[page_index].get_text("text") or ""
            remaining = MAX_CHARACTERS - characters
            parts.append(page_text[:remaining])
            characters += len(parts[-1])
            if characters >= MAX_CHARACTERS:
                break
    finally:
        document.close()
    return "\n".join(parts)
