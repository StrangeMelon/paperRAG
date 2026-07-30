"""论文采集领域模型的行为契约测试。"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from types import ModuleType

import pytest
from pydantic import ValidationError


def _schema_module() -> ModuleType:
    return importlib.import_module("paper_rag.ingest.schema")


def test_paper_meta_supports_minimal_metadata() -> None:
    schema = _schema_module()
    before_creation = datetime.now(UTC)

    # 最小论文记录只要求 paper_id 和 title
    meta = schema.PaperMeta(
        paper_id="arxiv:2310.11511",
        title="Self-RAG",
    )

    after_creation = datetime.now(UTC)

    assert meta.paper_id == "arxiv:2310.11511"
    assert meta.title == "Self-RAG"
    assert meta.authors == []
    assert meta.urls == []
    assert meta.extra == {}
    assert meta.year is None
    assert meta.venue is None
    assert meta.source == "unknown"
    assert meta.language is None
    assert before_creation <= meta.fetched_at <= after_creation


# 列表和字典默认值不能被不同论文对象共享
def test_mutable_defaults_are_not_shared() -> None:
    schema = _schema_module()
    first = schema.PaperMeta(paper_id="paper:first", title="First")
    second = schema.PaperMeta(paper_id="paper:second", title="Second")

    first.authors.append("Alice")
    first.urls.append("https://example.com/first")
    first.extra["citation_count"] = 10

    assert second.authors == []
    assert second.urls == []
    assert second.extra == {}


# FetchResult 将论文元数据与本地 PDF 路径组合起来
def test_fetch_result_serializes_nested_metadata() -> None:
    schema = _schema_module()
    meta = schema.PaperMeta(
        paper_id="doi:10.1000/example",
        title="Example Paper",
        authors=["Alice", "Bob"],
        year=2026,
        venue="Example Conference",
        doi="10.1000/example",
        abstract="An example abstract.",
        urls=["https://doi.org/10.1000/example"],
        source="crossref",
        extra={"citation_count": 12},
    )

    result = schema.FetchResult(
        meta=meta,
        pdf_path="/tmp/example.pdf",
    )
    serialized = result.model_dump()

    assert serialized["meta"]["paper_id"] == "doi:10.1000/example"
    assert serialized["meta"]["authors"] == ["Alice", "Bob"]
    assert serialized["meta"]["extra"] == {"citation_count": 12}
    assert serialized["pdf_path"] == "/tmp/example.pdf"


# 缺少 paper_id 或 title 的元数据时, Pydantic必须立即拒绝数据
def test_paper_meta_requires_identity_and_title() -> None:
    schema = _schema_module()

    with pytest.raises(ValidationError):
        schema.PaperMeta(title="Missing paper ID")

    with pytest.raises(ValidationError):
        schema.PaperMeta(paper_id="paper:missing-title")


@pytest.mark.parametrize("language", ["zh", "en", None])
def test_paper_meta_accepts_supported_document_languages(
    language: str | None,
) -> None:
    schema = _schema_module()
    meta = schema.PaperMeta(
        paper_id="paper:language",
        title="Language Paper",
        language=language,
    )

    assert meta.language == language


def test_paper_meta_rejects_unknown_document_language() -> None:
    schema = _schema_module()

    with pytest.raises(ValidationError):
        schema.PaperMeta(
            paper_id="paper:language",
            title="Language Paper",
            language="ch",
        )
