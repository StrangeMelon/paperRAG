"""PDF URL 采集器的边界行为测试。"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import to_safe_dirname

_PDF_BYTES = b"%PDF-1.7\nURL source boundary test\n%%EOF\n"


def _url_source_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.ingest.url_source")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.url_source":
            raise
        pytest.fail("尚未实现 paper_rag.ingest.url_source.UrlSource", pytrace=False)


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


def _install_http_transport(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    handler: Callable[[httpx.Request], httpx.Response],
) -> dict[str, Any]:
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    client_options: dict[str, Any] = {}

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client_options.update(kwargs)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", client_factory)
    return client_options


def test_fetch_rejects_non_pdf_url_before_network_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _url_source_module()

    def unexpected_client(*args: Any, **kwargs: Any) -> None:
        pytest.fail("非 PDF URL 不应发起网络请求")

    monkeypatch.setattr(module.httpx, "Client", unexpected_client)

    with pytest.raises(ValueError, match="direct PDF URLs only"):
        module.UrlSource().fetch("https://papers.example/article/123")


def test_fetch_follows_redirect_and_persists_downloaded_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _url_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.path == "/paper.pdf":
            return httpx.Response(
                302,
                headers={"Location": "/downloads/final.pdf"},
            )
        return httpx.Response(200, content=_PDF_BYTES)

    client_options = _install_http_transport(monkeypatch, module, handler)
    url = "https://papers.example/paper.pdf?download=1"

    result = module.UrlSource().fetch(url)

    expected_id = f"sha1:{hashlib.sha1(_PDF_BYTES).hexdigest()}"
    target_dir = storage_root / to_safe_dirname(expected_id)
    final_pdf = target_dir / "raw.pdf"

    assert isinstance(result, FetchResult)
    assert result.meta.paper_id == expected_id
    assert result.meta.title == "paper.pdf"
    assert result.meta.source == "url"
    assert result.meta.urls == [url]
    assert result.pdf_path == str(final_pdf)
    assert final_pdf.read_bytes() == _PDF_BYTES
    assert requested_urls == [
        url,
        "https://papers.example/downloads/final.pdf",
    ]
    assert client_options["timeout"] == 120
    assert client_options["follow_redirects"] is True

    persisted_meta = json.loads((target_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted_meta == result.meta.model_dump(mode="json")
    assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
        f"source=url\nquery={url}\n"
    )
    assert [path.name for path in storage_root.iterdir()] == [target_dir.name]


def test_explicit_title_overrides_url_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _url_source_module()
    _isolate_paper_storage(monkeypatch, module, tmp_path / "papers")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_PDF_BYTES)

    _install_http_transport(monkeypatch, module, handler)

    result = module.UrlSource(title="A Deliberate URL Title").fetch(
        "https://papers.example/original-name.pdf"
    )

    assert result.meta.title == "A Deliberate URL Title"


def test_http_error_is_propagated_without_persisting_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _url_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    _install_http_transport(monkeypatch, module, handler)

    with pytest.raises(httpx.HTTPStatusError, match="404"):
        module.UrlSource().fetch("https://papers.example/missing.pdf")

    assert not storage_root.exists()
