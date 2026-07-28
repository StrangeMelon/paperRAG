"""论文采集器公共接口的行为契约。"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from paper_rag.ingest.schema import FetchResult, PaperMeta


def _paper_source_class() -> type[Any]:
    try:
        module = importlib.import_module("paper_rag.ingest.sources")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.sources":
            raise
        pytest.fail("尚未实现 paper_rag.ingest.sources.PaperSource", pytrace=False)

    return module.PaperSource


def test_paper_source_has_neutral_default_name() -> None:
    paper_source = _paper_source_class()

    assert paper_source.name == "abstract"


def test_paper_source_cannot_be_instantiated_directly() -> None:
    paper_source = _paper_source_class()

    with pytest.raises(TypeError, match="abstract"):
        paper_source()


def test_concrete_source_returns_the_standard_fetch_result() -> None:
    paper_source = _paper_source_class()

    class MemorySource(paper_source):
        name = "memory"

        def fetch(self, identifier: str) -> FetchResult:
            return FetchResult(
                meta=PaperMeta(
                    paper_id=f"memory:{identifier}",
                    title="Test Paper",
                    source=self.name,
                ),
                pdf_path="/tmp/test-paper.pdf",
            )

    result = MemorySource().fetch("example")

    assert result.meta.paper_id == "memory:example"
    assert result.meta.source == "memory"
    assert result.pdf_path == "/tmp/test-paper.pdf"
