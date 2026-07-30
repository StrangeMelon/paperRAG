"""直接 PDF URL 采集源。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ..utils.ids import make_paper_id
from ..utils.logger import get_logger
from ..utils.paths import paper_dir
from .metadata import persist_paper_meta
from .schema import FetchResult, PaperMeta
from .sources import PaperSource

log = get_logger("ingest.url")


class UrlSource(PaperSource):
    """从 HTTP 或 HTTPS 直链下载 PDF。"""

    name = "url"

    def __init__(self, title: str | None = None) -> None:
        self.title = title

    def fetch(self, identifier: str) -> FetchResult:
        url = identifier
        parsed_url = urlparse(url)

        if not parsed_url.path.lower().endswith(".pdf"):
            raise ValueError(
                "UrlSource currently supports direct PDF URLs only. "
                "For HTML pages use a separate web-reading ingest path."
            )

        log.info(f"url fetch GET {url}")

        # raise_for_status() 在任何文件落盘前执行。
        with httpx.Client(
            timeout=120,
            follow_redirects=True,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content

        # make_paper_id() 接受文件路径, 因此先写入隔离临时文件。
        # TemporaryDirectory 退出时会自动删除, 不污染 papers_dir。
        with tempfile.TemporaryDirectory(
            prefix="paper-rag-url-source-"
        ) as temp_name:
            temporary_pdf = Path(temp_name) / "raw.pdf"
            temporary_pdf.write_bytes(content)
            paper_id = make_paper_id(pdf_path=temporary_pdf)

        target = paper_dir(paper_id)
        target.mkdir(parents=True, exist_ok=True)

        final_pdf = target / "raw.pdf"
        if not final_pdf.exists():
            final_pdf.write_bytes(content)
            log.info(f"downloaded PDF -> {final_pdf}")
        else:
            log.info("PDF already present, skip write")

        filename = Path(unquote(parsed_url.path)).name
        meta = PaperMeta(
            paper_id=paper_id,
            title=self.title or filename,
            urls=[url],
            source=self.name,
        )

        meta = persist_paper_meta(target, meta)

        (target / "source.txt").write_text(
            f"source={self.name}\nquery={url}\n",
            encoding="utf-8",
        )

        return FetchResult(
            meta=meta,
            pdf_path=str(final_pdf),
        )
