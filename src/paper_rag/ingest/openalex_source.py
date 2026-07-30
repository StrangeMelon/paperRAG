"""OpenAlex 论文采集源。

OpenAlex 主要用于补充论文元数据。
只有响应中提供开放获取地址时才尝试下载 PDF。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..utils.ids import make_paper_id, normalize_doi
from ..utils.logger import get_logger
from ..utils.paths import paper_dir
from .metadata import persist_paper_meta
from .schema import FetchResult, PaperMeta
from .sources import PaperSource

log = get_logger(__name__)

_BASE_URL = "https://api.openalex.org"
_MAILTO = "paper-rag@example.com"


class OpenAlexSource(PaperSource):
    """通过 OpenAlex API 采集论文元数据和开放获取 PDF。"""

    name = "openalex"

    def fetch(self, identifier: str) -> FetchResult:
        request_url = _build_request_url(identifier)

        log.info(f"openalex GET {request_url}")

        with httpx.Client(timeout=30) as client:
            response = client.get(
                request_url,
                params={"mailto": _MAILTO},
            )
            response.raise_for_status()
            data = response.json()

        ids = data.get("ids") or {}
        doi = normalize_doi(ids.get("doi") or "")

        if doi:
            paper_id = make_paper_id(doi=doi)
        else:
            openalex_id = data["id"].rsplit("/", 1)[-1]
            paper_id = f"openalex:{openalex_id}"

        target = paper_dir(paper_id)
        target.mkdir(parents=True, exist_ok=True)

        pdf_path = target / "raw.pdf"

        open_access = data.get("open_access") or {}
        landing_url = open_access.get("oa_url")
        pdf_url = _find_pdf_url(data)

        if pdf_url and not pdf_path.exists():
            _download_optional_pdf(
                pdf_url=pdf_url,
                pdf_path=pdf_path,
            )
        elif pdf_path.exists():
            log.info(f"PDF already present, skip download: {pdf_path}")

        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in (data.get("authorships") or [])
            if authorship.get("author")
        ]

        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}
        reported_language = data.get("language")
        language = reported_language if reported_language in {"zh", "en"} else None

        meta = PaperMeta(
            paper_id=paper_id,
            title=(data.get("title") or "").strip(),
            authors=[author for author in authors if author],
            year=data.get("publication_year"),
            venue=source.get("display_name"),
            doi=doi,
            abstract=_decode_abstract(
                data.get("abstract_inverted_index")
            ),
            language=language,
            urls=_unique_urls(
                pdf_url,
                landing_url,
                data.get("id"),
            ),
            source=self.name,
        )

        _persist_meta(
            target=target,
            meta=meta,
            source_query=identifier,
        )

        return FetchResult(
            meta=meta,
            pdf_path=str(pdf_path) if pdf_path.exists() else "",
        )


def _build_request_url(identifier: str) -> str:
    """把 DOI、OpenAlex URL 或裸 OpenAlex ID 转换为 API URL。"""

    if identifier.lower().startswith("doi:"):
        doi = normalize_doi(identifier)
        if doi is None:
            raise ValueError(f"invalid DOI identifier: {identifier}")
        return f"{_BASE_URL}/works/doi:{doi}"

    if identifier.startswith("https://openalex.org/"):
        return identifier.replace(
            "https://openalex.org/",
            f"{_BASE_URL}/works/",
            1,
        )

    return f"{_BASE_URL}/works/{identifier}"


def _find_pdf_url(data: dict[str, Any]) -> str | None:
    """按照 OpenAlex 推荐顺序寻找真正的 PDF 地址。"""

    for location_name in (
        "best_oa_location",
        "primary_location",
    ):
        location = data.get(location_name) or {}
        pdf_url = location.get("pdf_url")
        if pdf_url:
            return pdf_url

    for location in data.get("locations") or []:
        if not location:
            continue

        pdf_url = location.get("pdf_url")
        if pdf_url:
            return pdf_url

    return None


def _unique_urls(*urls: str | None) -> list[str]:
    """保留 URL 顺序并删除重复值和空值。"""

    return list(
        dict.fromkeys(
            url
            for url in urls
            if url
        )
    )


def _download_optional_pdf(
    *,
    pdf_url: str,
    pdf_path: Path,
) -> None:
    """尝试下载开放获取 PDF。

    OpenAlex 的主要价值是元数据。因此 PDF 下载失败时保留元数据采集结果。
    """

    log.info(f"openalex PDF GET {pdf_url}")

    try:
        with httpx.Client(
            timeout=120,
            follow_redirects=True,
        ) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
    except Exception as exc:
        log.warning(f"openalex PDF download failed: {exc}")


def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准元数据和原始采集参数。"""

    meta = persist_paper_meta(target, meta)

    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )
    return meta


def _decode_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str | None:
    """将 OpenAlex 倒排索引格式的摘要恢复为普通文本。"""

    if not inverted_index:
        return None

    position_to_word: dict[int, str] = {}

    for word, positions in inverted_index.items():
        for position in positions:
            position_to_word[position] = word

    if not position_to_word:
        return None

    return " ".join(
        position_to_word[position]
        for position in sorted(position_to_word)
    )
