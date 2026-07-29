"""OpenAlex 采集源的边界行为测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import to_safe_dirname

_PDF_BYTES = b"%PDF-1.7\nopenalex boundary test\n%%EOF\n"
_API_URL = "https://api.openalex.org/works/doi:10.1000/openalex-test"
_PDF_URL = "https://publisher.example/openalex-test.pdf"
_LANDING_URL = "https://doi.org/10.1000/openalex-test"
_PRIMARY_PDF_URL = "https://primary.example/openalex-test.pdf"
_LOCATION_PDF_URL = "https://repository.example/openalex-test.pdf"


def _openalex_source_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.ingest.openalex_source")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.openalex_source":
            raise
        pytest.fail(
            "尚未实现 paper_rag.ingest.openalex_source.OpenAlexSource",
            pytrace=False,
        )


def _isolate_paper_storage(
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
        pdf_url: str | None,
        pdf_error: Exception | None,
        calls: list[dict[str, Any]],
        timeout: int,
        follow_redirects: bool = False,
    ) -> None:
        self._payload = payload
        self._pdf_url = pdf_url
        self._pdf_error = pdf_error
        self._calls = calls
        self._calls.append(
            {
                "timeout": timeout,
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
        if url.startswith("https://api.openalex.org/works/"):
            return _FakeResponse(payload=self._payload)
        assert url == self._pdf_url
        return _FakeResponse(
            content=_PDF_BYTES,
            error=self._pdf_error,
        )


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    payload: dict[str, Any],
    pdf_url: str | None,
    pdf_error: Exception | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def client_factory(
        *,
        timeout: int,
        follow_redirects: bool = False,
    ) -> _FakeClient:
        return _FakeClient(
            payload=payload,
            pdf_url=pdf_url,
            pdf_error=pdf_error,
            calls=calls,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr(module.httpx, "Client", client_factory)
    return calls


def _metadata_payload(*, pdf_url: str | None) -> dict[str, Any]:
    return {
        "id": "https://openalex.org/W123456789",
        "ids": {"doi": "https://doi.org/10.1000/openalex-test"},
        "title": "  OpenAlex Boundary Paper  ",
        "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ],
        "publication_year": 2024,
        "primary_location": {
            "source": {"display_name": "Boundary Journal"},
        },
        "best_oa_location": {"pdf_url": pdf_url},
        "open_access": {"oa_url": pdf_url},
        "abstract_inverted_index": {
            "A": [0],
            "boundary": [1],
            "abstract": [2],
        },
    }


def test_fetch_by_doi_downloads_oa_pdf_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _openalex_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    calls = _install_fake_httpx(
        monkeypatch,
        module,
        payload=_metadata_payload(pdf_url=_PDF_URL),
        pdf_url=_PDF_URL,
    )

    result = module.OpenAlexSource().fetch("doi:10.1000/openalex-test")

    target_dir = storage_root / to_safe_dirname("doi:10.1000/openalex-test")
    pdf_path = target_dir / "raw.pdf"
    assert isinstance(result, FetchResult)
    assert result.meta.paper_id == "doi:10.1000/openalex-test"
    assert result.meta.title == "OpenAlex Boundary Paper"
    assert result.meta.authors == ["Alice", "Bob"]
    assert result.meta.year == 2024
    assert result.meta.venue == "Boundary Journal"
    assert result.meta.doi == "10.1000/openalex-test"
    assert result.meta.abstract == "A boundary abstract"
    assert result.meta.urls == [_PDF_URL, "https://openalex.org/W123456789"]
    assert result.meta.source == "openalex"
    assert result.pdf_path == str(pdf_path)
    assert pdf_path.read_bytes() == _PDF_BYTES
    assert calls == [
        {"timeout": 30, "follow_redirects": False},
        {
            "url": _API_URL,
            "params": {"mailto": "paper-rag@example.com"},
        },
        {"timeout": 120, "follow_redirects": True},
        {"url": _PDF_URL, "params": None},
    ]
    assert json.loads(
        (target_dir / "meta.json").read_text(encoding="utf-8")
    ) == result.meta.model_dump(mode="json")
    assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
        "source=openalex\nquery=doi:10.1000/openalex-test\n"
    )


def test_fetch_by_openalex_id_returns_metadata_without_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _openalex_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    payload = _metadata_payload(pdf_url=None)
    payload["ids"] = {}
    calls = _install_fake_httpx(
        monkeypatch,
        module,
        payload=payload,
        pdf_url=None,
    )

    result = module.OpenAlexSource().fetch("W123456789")

    target_dir = storage_root / to_safe_dirname("openalex:W123456789")
    assert result.meta.paper_id == "openalex:W123456789"
    assert result.meta.urls == ["https://openalex.org/W123456789"]
    assert result.pdf_path == ""
    assert not (target_dir / "raw.pdf").exists()
    assert calls == [
        {"timeout": 30, "follow_redirects": False},
        {
            "url": "https://api.openalex.org/works/W123456789",
            "params": {"mailto": "paper-rag@example.com"},
        },
    ]


def test_pdf_download_failure_keeps_metadata_and_returns_empty_pdf_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _openalex_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    pdf_error = httpx.HTTPStatusError(
        "publisher unavailable",
        request=httpx.Request("GET", _PDF_URL),
        response=httpx.Response(503),
    )
    _install_fake_httpx(
        monkeypatch,
        module,
        payload=_metadata_payload(pdf_url=_PDF_URL),
        pdf_url=_PDF_URL,
        pdf_error=pdf_error,
    )

    result = module.OpenAlexSource().fetch("https://openalex.org/W123456789")

    target_dir = storage_root / to_safe_dirname("doi:10.1000/openalex-test")
    assert result.meta.paper_id == "doi:10.1000/openalex-test"
    assert result.meta.title == "OpenAlex Boundary Paper"
    assert result.pdf_path == ""
    assert not (target_dir / "raw.pdf").exists()
    assert (target_dir / "meta.json").is_file()


def test_fetch_prefers_best_oa_pdf_url_over_other_locations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _openalex_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    payload = _metadata_payload(pdf_url=None)
    payload["open_access"] = {"oa_url": _LANDING_URL}
    payload["best_oa_location"] = {"pdf_url": _PDF_URL}
    payload["primary_location"] = {
        "source": {"display_name": "Boundary Journal"},
        "pdf_url": _PRIMARY_PDF_URL,
    }
    payload["locations"] = [{"pdf_url": _LOCATION_PDF_URL}]
    calls = _install_fake_httpx(
        monkeypatch,
        module,
        payload=payload,
        pdf_url=_PDF_URL,
    )

    result = module.OpenAlexSource().fetch("doi:10.1000/openalex-test")

    target_dir = storage_root / to_safe_dirname("doi:10.1000/openalex-test")
    assert result.pdf_path == str(target_dir / "raw.pdf")
    assert result.meta.urls == [
        _PDF_URL,
        _LANDING_URL,
        "https://openalex.org/W123456789",
    ]
    assert calls[-2:] == [
        {"timeout": 120, "follow_redirects": True},
        {"url": _PDF_URL, "params": None},
    ]


def test_fetch_does_not_treat_oa_landing_page_as_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _openalex_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    payload = _metadata_payload(pdf_url=None)
    payload["open_access"] = {"oa_url": _LANDING_URL}
    calls = _install_fake_httpx(
        monkeypatch,
        module,
        payload=payload,
        pdf_url=None,
    )

    result = module.OpenAlexSource().fetch("doi:10.1000/openalex-test")

    target_dir = storage_root / to_safe_dirname("doi:10.1000/openalex-test")
    assert result.pdf_path == ""
    assert result.meta.urls == [
        _LANDING_URL,
        "https://openalex.org/W123456789",
    ]
    assert not (target_dir / "raw.pdf").exists()
    assert calls == [
        {"timeout": 30, "follow_redirects": False},
        {
            "url": _API_URL,
            "params": {"mailto": "paper-rag@example.com"},
        },
    ]
