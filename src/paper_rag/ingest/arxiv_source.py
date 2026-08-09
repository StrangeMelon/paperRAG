"""arXiv 论文采集源。"""

from __future__ import annotations

from pathlib import Path

from ..utils.ids import make_paper_id, normalize_arxiv, split_arxiv_version
from ..utils.logger import get_logger
from ..utils.paths import paper_dir
from .metadata import persist_paper_meta
from .schema import FetchResult, PaperMeta
from .sources import PaperSource

log = get_logger(__name__)


class ArxivSource(PaperSource):
    """通过 arXiv API 获取论文元数据和 PDF。"""

    name = "arxiv"

    def fetch(self, identifier: str) -> FetchResult:
        try:
            import arxiv
        except ImportError as exc:
            raise RuntimeError(
                "arxiv package not installed. "
                "Run: uv sync --extra ingest"
            ) from exc

        normalized = normalize_arxiv(identifier) or identifier
        _, requested_version = split_arxiv_version(identifier)

        log.info(
            f"arxiv fetch id={normalized} version={requested_version}"
        )

        client = arxiv.Client(
            page_size=1,
            delay_seconds=10,
            num_retries=5,
        )
        search = arxiv.Search(id_list=[normalized])

        try:
            result = next(client.results(search))
        except StopIteration as exc:
            raise ValueError(
                f"arxiv id not found: {normalized}"
            ) from exc

        paper_id = make_paper_id(arxiv_id=normalized)
        target = paper_dir(paper_id)
        target.mkdir(parents=True, exist_ok=True)

        pdf_path = target / "raw.pdf"

        if pdf_path.exists():
            log.info(f"PDF already present, skip download: {pdf_path}")
        else:
            log.info(f"downloading PDF -> {pdf_path}")
            _download_pdf(
                client=client,
                result=result,
                target=target,
                pdf_path=pdf_path,
            )

        meta = PaperMeta(
            paper_id=paper_id,
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            year=result.published.year if result.published else None,
            venue="arXiv",
            doi=result.doi,
            arxiv_id=normalized,
            abstract=(result.summary or "").strip(),
            urls=[result.entry_id, result.pdf_url],
            source=self.name,
            extra=(
                {"arxiv_version": requested_version}
                if requested_version
                else {}
            ),
        )

        meta = _persist_meta(
            target=target,
            meta=meta,
            source_query=identifier,
        )

        return FetchResult(
            meta=meta,
            pdf_path=str(pdf_path),
        )


def _download_pdf(
    *,
    client: object,
    result: object,
    target: Path,
    pdf_path: Path,
) -> None:
    """兼容不同版本 arxiv 包提供的 PDF 下载接口。"""

    try:
        client.download_pdf(
            result,
            dirpath=str(target),
            filename="raw.pdf",
        )
    except (AttributeError, TypeError):
        if hasattr(result, "download_pdf"):
            result.download_pdf(
                dirpath=str(target),
                filename="raw.pdf",
            )
            return

        import httpx

        with httpx.Client(
            timeout=120,
            follow_redirects=True,
        ) as http_client:
            response = http_client.get(result.pdf_url)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)


def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准化元数据和原始采集参数。"""

    meta = persist_paper_meta(target, meta)

    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )

    return meta
