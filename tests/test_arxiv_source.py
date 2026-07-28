"""arXiv 采集器的边界行为测试。"""

from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import to_safe_dirname

_PDF_BYTES = b"%PDF-1.7\narXiv source boundary test\n%%EOF\n"


def _arxiv_source_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.ingest.arxiv_source")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.arxiv_source":
            raise
        pytest.fail("尚未实现 paper_rag.ingest.arxiv_source.ArxivSource", pytrace=False)


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


class _FakeResult:
    title = "  Retrieval-Augmented Generation for Testing  "
    authors = [SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")]
    published = datetime(2023, 10, 18, tzinfo=UTC)
    doi = "10.1000/arxiv-test"
    summary = "  A boundary-test abstract.  "
    entry_id = "https://arxiv.org/abs/2310.12345v3"
    pdf_url = "https://arxiv.org/pdf/2310.12345v3"

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def download_pdf(self, *, dirpath: str, filename: str) -> None:
        self._captured["downloads"].append(("result", dirpath, filename))
        (Path(dirpath) / filename).write_bytes(_PDF_BYTES)


def _install_fake_arxiv(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: list[_FakeResult],
    download_mode: str = "client",
) -> dict[str, Any]:
    captured: dict[str, Any] = {
        "client_options": None,
        "search_ids": None,
        "downloads": [],
    }
    fake_arxiv = ModuleType("arxiv")

    class Search:
        def __init__(self, *, id_list: list[str]) -> None:
            captured["search_ids"] = id_list
            self.id_list = id_list

    class Client:
        def __init__(
            self,
            *,
            page_size: int,
            delay_seconds: int,
            num_retries: int,
        ) -> None:
            captured["client_options"] = {
                "page_size": page_size,
                "delay_seconds": delay_seconds,
                "num_retries": num_retries,
            }

        def results(self, search: Search) -> Any:
            assert search.id_list == captured["search_ids"]
            return iter(results)

        def download_pdf(
            self,
            result: _FakeResult,
            *,
            dirpath: str,
            filename: str,
        ) -> None:
            if download_mode == "legacy":
                raise AttributeError("Client.download_pdf is unavailable")
            if download_mode == "unexpected":
                pytest.fail("已有 raw.pdf 时不应重复下载")
            captured["downloads"].append(("client", dirpath, filename))
            (Path(dirpath) / filename).write_bytes(_PDF_BYTES)

    fake_arxiv.Search = Search
    fake_arxiv.Client = Client
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv)
    return captured


def test_fetch_normalizes_versioned_id_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _arxiv_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    captured: dict[str, Any] = {
        "downloads": [],
    }
    result_record = _FakeResult(captured)
    arxiv_calls = _install_fake_arxiv(
        monkeypatch,
        results=[result_record],
    )
    identifier = "https://arxiv.org/abs/2310.12345v3"

    result = module.ArxivSource().fetch(identifier)

    target_dir = storage_root / to_safe_dirname("arxiv:2310.12345")
    pdf_path = target_dir / "raw.pdf"
    assert isinstance(result, FetchResult)
    assert result.meta.paper_id == "arxiv:2310.12345"
    assert result.meta.title == "Retrieval-Augmented Generation for Testing"
    assert result.meta.authors == ["Alice", "Bob"]
    assert result.meta.year == 2023
    assert result.meta.venue == "arXiv"
    assert result.meta.doi == "10.1000/arxiv-test"
    assert result.meta.arxiv_id == "2310.12345"
    assert result.meta.abstract == "A boundary-test abstract."
    assert result.meta.urls == [result_record.entry_id, result_record.pdf_url]
    assert result.meta.source == "arxiv"
    assert result.meta.extra == {"arxiv_version": "v3"}
    assert result.pdf_path == str(pdf_path)
    assert pdf_path.read_bytes() == _PDF_BYTES

    assert arxiv_calls["client_options"] == {
        "page_size": 1,
        "delay_seconds": 10,
        "num_retries": 5,
    }
    assert arxiv_calls["search_ids"] == ["2310.12345"]
    assert arxiv_calls["downloads"] == [
        ("client", str(target_dir), "raw.pdf")
    ]
    assert json.loads((target_dir / "meta.json").read_text(encoding="utf-8")) == (
        result.meta.model_dump(mode="json")
    )
    assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
        f"source=arxiv\nquery={identifier}\n"
    )


def test_fetch_raises_when_arxiv_id_has_no_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _arxiv_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    _install_fake_arxiv(monkeypatch, results=[])

    with pytest.raises(
        ValueError,
        match=re.escape("arxiv id not found: 2310.99999"),
    ):
        module.ArxivSource().fetch("arXiv:2310.99999v2")

    assert not storage_root.exists()


def test_fetch_reuses_existing_pdf_without_downloading_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _arxiv_source_module()
    storage_root = tmp_path / "papers"
    target_dir = storage_root / to_safe_dirname("arxiv:2310.12345")
    target_dir.mkdir(parents=True)
    existing_pdf = target_dir / "raw.pdf"
    existing_pdf.write_bytes(b"existing PDF bytes")
    _isolate_paper_storage(monkeypatch, module, storage_root)
    result_capture: dict[str, Any] = {"downloads": []}
    result_record = _FakeResult(result_capture)
    arxiv_calls = _install_fake_arxiv(
        monkeypatch,
        results=[result_record],
        download_mode="unexpected",
    )

    result = module.ArxivSource().fetch("2310.12345")

    assert Path(result.pdf_path).read_bytes() == b"existing PDF bytes"
    assert arxiv_calls["downloads"] == []


def test_fetch_falls_back_to_result_download_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _arxiv_source_module()
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    result_capture: dict[str, Any] = {"downloads": []}
    result_record = _FakeResult(result_capture)
    _install_fake_arxiv(
        monkeypatch,
        results=[result_record],
        download_mode="legacy",
    )

    result = module.ArxivSource().fetch("2310.12345")

    assert Path(result.pdf_path).read_bytes() == _PDF_BYTES
    assert result_capture["downloads"] == [
        ("result", str(storage_root / "arxiv_2310.12345"), "raw.pdf")
    ]
