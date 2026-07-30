"""
Semantic Scholar 论文采集源
通过 Semantic Scholar Graph API, 根据论文标识符获取论文元数据, 并在存在开放访问 PDF 时将 PDF 下载到本地。

"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..utils.ids import make_paper_id, normalize_arxiv, normalize_doi
from ..utils.logger import get_logger
from ..utils.paths import paper_dir
from .metadata import persist_paper_meta
from .schema import FetchResult, PaperMeta
from .sources import PaperSource

log = get_logger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "title,authors,year,venue,abstract,"
    "externalIds,openAccessPdf"
)


class SemanticScholarSource(PaperSource):
    """通过 Semantic Scholar Graph API 采集论文。"""

    name = "semantic_scholar"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30,
    ) -> None:
        self.headers: dict[str, str] = {
            "User-Agent": "paper-rag/0.1",
        }

        if api_key:
            self.headers["x-api-key"] = api_key

        self.timeout = timeout

    def fetch(self, identifier: str) -> FetchResult:
        request_id = self._normalize_id(identifier)

        log.info(f"semantic scholar fetch id={request_id}")

        with httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
        ) as client:
            response = client.get(
                f"{_BASE_URL}/paper/{request_id}",
                params={"fields": _FIELDS},
            )
            response.raise_for_status()
            data = response.json()

        external_ids = data.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")

        if arxiv_id or doi:
            paper_id = make_paper_id(
                arxiv_id=arxiv_id,
                doi=doi,
            )
        else:
            paper_id = f"s2:{data['paperId']}"

        target = paper_dir(paper_id)
        target.mkdir(parents=True, exist_ok=True)

        pdf_path = target / "raw.pdf"
        open_access_pdf = data.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url")

        if pdf_url and not pdf_path.exists():
            _download_pdf(
                pdf_url=pdf_url,
                pdf_path=pdf_path,
            )
        elif pdf_path.exists():
            log.info(f"PDF already present, skip download: {pdf_path}")
        else:
            log.warning(
                f"no openAccessPdf for {request_id}; "
                "pdf_path will be empty"
            )

        authors = [
            author["name"]
            for author in (data.get("authors") or [])
            if author.get("name")
        ]

        meta = PaperMeta(
            paper_id=paper_id,
            title=(data.get("title") or "").strip(),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            doi=doi,
            arxiv_id=arxiv_id,
            abstract=data.get("abstract"),
            urls=[
                url
                for url in (
                    pdf_url,
                    (
                        "https://www.semanticscholar.org/paper/"
                        f"{data.get('paperId')}"
                    ),
                )
                if url
            ],
            source=self.name,
            extra={
                "externalIds": external_ids,
                "paperId": data.get("paperId"),
            },
        )

        meta = _persist_meta(
            target=target,
            meta=meta,
            source_query=identifier,
        )

        return FetchResult(
            meta=meta,
            pdf_path=str(pdf_path) if pdf_path.exists() else "",
        )

    def _normalize_id(self, identifier: str) -> str:
        """将不同形式的论文标识符转换为 S2 查询标识符。"""

        arxiv_id = normalize_arxiv(identifier)

        if identifier.lower().startswith("arxiv:"):
            return (
                f"arxiv:{arxiv_id}"
                if arxiv_id
                else identifier
            )

        doi = normalize_doi(identifier)

        if identifier.lower().startswith("doi:"):
            return f"DOI:{doi}" if doi else identifier

        if doi:
            return f"DOI:{doi}"

        if arxiv_id:
            return f"arxiv:{arxiv_id}"

        return identifier


def _download_pdf(
    *,
    pdf_url: str,
    pdf_path: Path,
) -> None:
    """从 Semantic Scholar 返回的开放地址下载 PDF。"""

    log.info(f"semantic scholar PDF GET {pdf_url}")

    with httpx.Client(
        timeout=120,
        follow_redirects=True,
    ) as client:
        response = client.get(pdf_url)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)


def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准元数据和原始采集参数。"""

    meta = persist_paper_meta(meta=meta, target=target)

    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )
    return meta
