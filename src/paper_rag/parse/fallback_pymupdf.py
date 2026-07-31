"""基于 PyMuPDF 的纯文本兜底解析器。

当 MinerU 不可用或解析失败时, 将 PDF 按页提取为最小 Markdown。
该解析器不提取图片、表格、公式或章节层级。
"""

from __future__ import annotations

from pathlib import Path

from ..utils.logger import get_logger
from ..utils.paths import parsed_dir

log = get_logger(__name__)


def parse_pdf(
    paper_id: str,
    pdf_path: str | Path,
) -> Path:
    """将 PDF 按页提取为 parsed/<paper_id>/paper.md。"""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF 未安装, 请执行: uv sync --extra ingest"
        ) from exc

    output_dir = parsed_dir(paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "paper.md"
    document = fitz.open(str(pdf_path))
    parts: list[str] = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            text = text.replace("\x00", "").strip()

            parts.append(
                f"\n\n<!-- page {page_number} -->\n\n{text}"
            )
    finally:
        document.close()

    markdown = "\n".join(parts).strip()
    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    log.info(
        f"PyMuPDF fallback wrote {markdown_path} "
        f"({markdown_path.stat().st_size} bytes)"
    )

    return output_dir
