"""本地 PDF 采集源。

根据文件内容计算稳定的 paper_id, 将 PDF 复制到规范目录,
并保存标准化元数据与来源记录。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..utils.ids import make_paper_id
from ..utils.logger import get_logger
from ..utils.paths import paper_dir
from .schema import FetchResult, PaperMeta
from .sources import PaperSource

log = get_logger("ingest.local")


class LocalSource(PaperSource):
    """从本地文件系统采集 PDF。"""

    name = "local"

    def __init__(self, title: str | None = None) -> None:
        self.title = title

    def fetch(self, identifier: str) -> FetchResult:
        src = Path(identifier).expanduser().resolve()

        if not src.is_file():
            raise FileNotFoundError(f"PDF not found: {src}")

        # 本地文件没有稳定的外部 ID, 因此使用文件内容 SHA-1。
        paper_id = make_paper_id(pdf_path=src)

        target = paper_dir(paper_id)
        target.mkdir(parents=True, exist_ok=True)

        pdf_path = target / "raw.pdf"

        # 已经存在规范副本时不重复复制。
        if not pdf_path.exists():
            shutil.copy2(src, pdf_path)
            log.info(f"copied PDF -> {pdf_path}")
        else:
            log.info("PDF already present, skip copy")

        meta = PaperMeta(
            paper_id=paper_id,
            title=self.title or src.stem,
            source=self.name,
            urls=[f"file://{src}"],
        )

        (target / "meta.json").write_text(
            json.dumps(
                meta.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (target / "source.txt").write_text(
            f"source={self.name}\nquery={identifier}\n",
            encoding="utf-8",
        )

        return FetchResult(
            meta=meta,
            pdf_path=str(pdf_path),
        )
