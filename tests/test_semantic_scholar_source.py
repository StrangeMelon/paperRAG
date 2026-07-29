"""Semantic Scholar 采集源的边界行为测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import to_safe_dirname

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf"
_PDF_URL = "https://repository.example/paper.pdf"
_PDF_BYTES = b"%PDF-1.7\nsemantic scholar boundary test\n%%EOF\n"


def _source_module() -> ModuleType:
    try:
        return importlib.import_module(
            "paper_rag.ingest.semantic_scholar_source"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.semantic_scholar_source":
            raise
        pytest.fail(
            "尚未实现 paper_rag.ingest.semantic_scholar_source."
            "SemanticScholarSource",
            pytrace=False,
        )


def _isolate_storage(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    storage_root: Path,
) -> None:
    monkeypatch.setattr(
        module,
        "paper_dir",
        lambda paper_id: storage_root / to_safe_dirname(paper_id),
    )


class _FakeResponse:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
        error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.content = content
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


class _FakeClient:
    def __init__(
        self,
        *,
        payload: dict[str, Any],
        calls: list[dict[str, Any]],
        pdf_error: Exception | None,
        timeout: float,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
    ) -> None:
        self._payload = payload
        self._calls = calls
        self._pdf_error = pdf_error
        self._calls.append(
            {
                "timeout": timeout,
                "headers": headers,
                "follow_redirects": follow_redirects,
            }
        )

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self._calls.append({"url": url, "params": params})
        if url.startswith(f"{_BASE_URL}/paper/"):
            return _FakeResponse(payload=self._payload)
        assert url == _PDF_URL
        return _FakeResponse(content=_PDF_BYTES, error=self._pdf_error)


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    payload: dict[str, Any],
    pdf_error: Exception | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def client_factory(
        *,
        timeout: float,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
    ) -> _FakeClient:
        return _FakeClient(
            payload=payload,
            calls=calls,
            pdf_error=pdf_error,
            timeout=timeout,
            headers=headers,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr(module.httpx, "Client", client_factory)
    return calls


def _paper_payload(
    *,
    external_ids: dict[str, Any] | None,
    pdf_url: str | None,
) -> dict[str, Any]:
    return {
        "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
        "title": "  Semantic Scholar Boundary Paper  ",
        "authors": [
            {"authorId": "1", "name": "Alice"},
            {"authorId": "2", "name": "Bob"},
        ],
        "year": 2023,
        "venue": "Boundary Conference",
        "abstract": "A boundary abstract.",
        "externalIds": external_ids,
        "openAccessPdf": (
            {"url": pdf_url, "status": "GREEN"}
            if pdf_url
            else None
        ),
    }


def test_fetch_arxiv_url_downloads_pdf_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _source_module()
    storage_root = tmp_path / "papers"
    _isolate_storage(monkeypatch, module, storage_root)
    payload = _paper_payload(
        external_ids={
            "ArXiv": "2310.12345",
            "DOI": "10.1000/S2-Test",
            "CorpusId": 123456,
        },
        pdf_url=_PDF_URL,
    )
    calls = _install_fake_httpx(monkeypatch, module, payload=payload)
    identifier = "https://arxiv.org/abs/2310.12345v2"

    result = module.SemanticScholarSource(
        api_key="secret-test-key",
        timeout=42,
    ).fetch(identifier)

    target_dir = storage_root / to_safe_dirname("arxiv:2310.12345")
    pdf_path = target_dir / "raw.pdf"
    assert isinstance(result, FetchResult)
    assert result.meta.paper_id == "arxiv:2310.12345"
    assert result.meta.title == "Semantic Scholar Boundary Paper"
    assert result.meta.authors == ["Alice", "Bob"]
    assert result.meta.year == 2023
    assert result.meta.venue == "Boundary Conference"
    assert result.meta.doi == "10.1000/S2-Test"
    assert result.meta.arxiv_id == "2310.12345"
    assert result.meta.abstract == "A boundary abstract."
    assert result.meta.urls == [
        _PDF_URL,
        "https://www.semanticscholar.org/paper/"
        "649def34f8be52c8b66281af98ae884c09aef38b",
    ]
    assert result.meta.source == "semantic_scholar"
    assert result.meta.extra == {
        "externalIds": payload["externalIds"],
        "paperId": payload["paperId"],
    }
    assert result.pdf_path == str(pdf_path)
    assert pdf_path.read_bytes() == _PDF_BYTES
    assert calls == [
        {
            "timeout": 42,
            "headers": {
                "User-Agent": "paper-rag/0.1",
                "x-api-key": "secret-test-key",
            },
            "follow_redirects": False,
        },
        {
            "url": f"{_BASE_URL}/paper/arxiv:2310.12345",
            "params": {"fields": _FIELDS},
        },
        {
            "timeout": 120,
            "headers": None,
            "follow_redirects": True,
        },
        {"url": _PDF_URL, "params": None},
    ]
    assert json.loads(
        (target_dir / "meta.json").read_text(encoding="utf-8")
    ) == result.meta.model_dump(mode="json")
    assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
        f"source=semantic_scholar\nquery={identifier}\n"
    )


def test_fetch_bare_doi_returns_metadata_without_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _source_module()
    storage_root = tmp_path / "papers"
    _isolate_storage(monkeypatch, module, storage_root)
    payload = _paper_payload(
        external_ids={"DOI": "10.1000/S2-Test"},
        pdf_url=None,
    )
    calls = _install_fake_httpx(monkeypatch, module, payload=payload)

    result = module.SemanticScholarSource().fetch("10.1000/S2-Test")

    target_dir = storage_root / to_safe_dirname("doi:10.1000/s2-test")
    assert result.meta.paper_id == "doi:10.1000/s2-test"
    assert result.pdf_path == ""
    assert not (target_dir / "raw.pdf").exists()
    assert calls == [
        {
            "timeout": 30,
            "headers": {"User-Agent": "paper-rag/0.1"},
            "follow_redirects": False,
        },
        {
            "url": f"{_BASE_URL}/paper/DOI:10.1000/s2-test",
            "params": {"fields": _FIELDS},
        },
    ]


def test_fetch_s2_id_falls_back_to_s2_paper_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _source_module()
    storage_root = tmp_path / "papers"
    _isolate_storage(monkeypatch, module, storage_root)
    payload = _paper_payload(external_ids=None, pdf_url=None)
    _install_fake_httpx(monkeypatch, module, payload=payload)

    result = module.SemanticScholarSource().fetch(payload["paperId"])

    expected_id = f"s2:{payload['paperId']}"
    target_dir = storage_root / to_safe_dirname(expected_id)
    assert result.meta.paper_id == expected_id
    assert result.meta.doi is None
    assert result.meta.arxiv_id is None
    assert result.meta.extra == {
        "externalIds": {},
        "paperId": payload["paperId"],
    }
    assert result.pdf_path == ""
    assert (target_dir / "meta.json").is_file()


def test_fetch_reuses_existing_pdf_without_downloading_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _source_module()
    storage_root = tmp_path / "papers"
    target_dir = storage_root / to_safe_dirname("arxiv:2310.12345")
    target_dir.mkdir(parents=True)
    existing_pdf = target_dir / "raw.pdf"
    existing_pdf.write_bytes(b"existing PDF bytes")
    _isolate_storage(monkeypatch, module, storage_root)
    payload = _paper_payload(
        external_ids={"ArXiv": "2310.12345"},
        pdf_url=_PDF_URL,
    )
    calls = _install_fake_httpx(monkeypatch, module, payload=payload)

    result = module.SemanticScholarSource().fetch("arxiv:2310.12345")

    assert Path(result.pdf_path).read_bytes() == b"existing PDF bytes"
    assert calls == [
        {
            "timeout": 30,
            "headers": {"User-Agent": "paper-rag/0.1"},
            "follow_redirects": False,
        },
        {
            "url": f"{_BASE_URL}/paper/arxiv:2310.12345",
            "params": {"fields": _FIELDS},
        },
    ]
