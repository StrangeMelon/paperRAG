"""解析后端调度、降级和结果有效性检查。"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

from .. import config as cfg
from ..utils.logger import get_logger
from ..utils.paths import parsed_dir

log = get_logger(__name__)
_PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)


class ParseError(RuntimeError):
    """所有解析后端均未产生可用正文。"""


def _has_meaningful_markdown(output_dir: Path) -> bool:
    markdown_path = output_dir / "paper.md"
    if not markdown_path.is_file():
        return False
    markdown = markdown_path.read_text(encoding="utf-8")
    return bool(_PAGE_MARKER_RE.sub("", markdown).strip())


def _write_status(
    output_dir: Path,
    *,
    paper_id: str,
    status: str,
    parser: str | None,
    reason: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parse_status.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "status": status,
                "parser": parser,
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_pdf(paper_id: str, pdf_path: str | Path) -> tuple[Path, str]:
    """返回标准化目录和实际解析后端名称。"""

    config = cfg.load()
    status_dir = parsed_dir(paper_id)
    mineru_reason = ""

    if config.mineru.mode == "local":
        mineru_local = importlib.import_module("paper_rag.parse.mineru_local")
        try:
            output_dir = mineru_local.parse_pdf(paper_id, pdf_path)
            if not _has_meaningful_markdown(output_dir):
                raise mineru_local.MineruError("MinerU produced no meaningful text")
            _write_status(
                output_dir,
                paper_id=paper_id,
                status="succeeded",
                parser="mineru",
                reason="",
            )
            return output_dir, "mineru"
        except mineru_local.MineruError as exc:
            mineru_reason = str(exc)
            log.warning(f"mineru failed: {exc}")
            if not config.mineru.fallback_to_pymupdf:
                _write_status(
                    status_dir,
                    paper_id=paper_id,
                    status="failed",
                    parser="mineru",
                    reason=mineru_reason,
                )
                raise

    fallback = importlib.import_module("paper_rag.parse.fallback_pymupdf")
    try:
        output_dir = fallback.parse_pdf(paper_id, pdf_path)
    except Exception as exc:
        reason = f"pymupdf_failed:{type(exc).__name__}:{exc}"
        _write_status(
            status_dir,
            paper_id=paper_id,
            status="failed",
            parser="pymupdf",
            reason=reason,
        )
        raise ParseError(reason) from exc

    if not _has_meaningful_markdown(output_dir):
        reason = "pymupdf_produced_no_meaningful_text"
        _write_status(
            output_dir,
            paper_id=paper_id,
            status="failed",
            parser="pymupdf",
            reason=reason,
        )
        raise ParseError(reason)

    status = "degraded" if mineru_reason else "succeeded"
    _write_status(
        output_dir,
        paper_id=paper_id,
        status=status,
        parser="pymupdf",
        reason=mineru_reason,
    )
    return output_dir, "pymupdf"
