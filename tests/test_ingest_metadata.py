from __future__ import annotations

import json
from pathlib import Path

from paper_rag.ingest.metadata import persist_paper_meta
from paper_rag.ingest.schema import PaperMeta


def _meta(language: str | None) -> PaperMeta:
    return PaperMeta(
        paper_id="paper:manual-language",
        title="Manual Language",
        language=language,
        source="local",
    )


def test_persist_paper_meta_writes_new_language(tmp_path: Path) -> None:
    saved = persist_paper_meta(tmp_path, _meta("zh"))
    payload = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))

    assert saved.language == "zh"
    assert payload["language"] == "zh"


def test_persist_paper_meta_preserves_existing_manual_language(
    tmp_path: Path,
) -> None:
    persist_paper_meta(tmp_path, _meta("zh"))

    saved = persist_paper_meta(tmp_path, _meta(None))

    assert saved.language == "zh"
    assert json.loads(
        (tmp_path / "meta.json").read_text(encoding="utf-8")
    )["language"] == "zh"


# 测试写入损坏的 meta.json
def test_persist_paper_meta_replaces_malformed_existing_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "meta.json").write_text("{broken", encoding="utf-8")    # 构造一个损坏的

    saved = persist_paper_meta(tmp_path, _meta("en"))

    assert saved.language == "en"
    assert json.loads(
        (tmp_path / "meta.json").read_text(encoding="utf-8")
    )["language"] == "en"
