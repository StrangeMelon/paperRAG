"""标准论文元数据的持久化与人工标注保护。"""

from __future__ import annotations

import json
from pathlib import Path

from ..utils.logger import get_logger
from .schema import PaperMeta

log = get_logger(__name__)

# 防止在重复采集时覆盖已有的人工语言标记, persist_paper_meta 会在写入 meta.json 前检查已有文件
def persist_paper_meta(target: Path, meta: PaperMeta) -> PaperMeta:
    """写入 meta.json, 并保留已有的非空人工语言标记。"""

    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "meta.json"
    existing_language: str | None = None

    if meta_path.is_file():
        try:
            existing = PaperMeta.model_validate_json(
                meta_path.read_text(encoding="utf-8")
            )
            existing_language = existing.language
        except Exception as exc:
            log.warning(
                f"invalid existing metadata will be replaced: "
                f"{type(exc).__name__}: {exc}"
            )

    if existing_language is not None:
        meta = meta.model_copy(update={"language": existing_language})

    meta_path.write_text(
        json.dumps(
            meta.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta
