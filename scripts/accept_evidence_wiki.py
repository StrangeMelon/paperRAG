#!/usr/bin/env python3
"""Real SQLite acceptance for evidence-to-Wiki enrichment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml


def _write_config(workdir: Path) -> Path:
    data = workdir / "data"
    config_path = workdir / "acceptance.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_root": str(data),
                    "papers_dir": str(data / "papers"),
                    "parsed_dir": str(data / "parsed"),
                    "index_dir": str(data / "index"),
                    "sqlite_path": str(data / "index" / "papers.sqlite"),
                    "bm25_path": str(data / "index" / "bm25.pkl"),
                    "models_dir": str(data / "index" / "models"),
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _run(workdir: Path) -> dict:
    os.environ["PAPER_RAG_CONFIG"] = str(_write_config(workdir))

    from paper_rag import config as cfg
    from paper_rag.store import sqlite_store
    from paper_rag.wiki import store as wstore
    from paper_rag.wiki.context import resolve_evidence_wiki_context
    from paper_rag.wiki.schema import WikiEntry, WikiLabel

    cfg.load.cache_clear()
    sqlite_store._ENGINE = None
    target = WikiEntry(
        entry_id="concept:state-space-model",
        name="State Space Model",
        category="method",
        definition="A state space sequence model used as background. [chunk:wiki-source]",
        definition_language="en",
        labels=[WikiLabel(text="State Space Model", language="en", kind="primary")],
    )
    duplicate = WikiEntry(
        entry_id="concept:ssm",
        name="SSM",
        category="method",
        definition="Duplicate entry.",
        definition_language="en",
        labels=[WikiLabel(text="SSM", language="en", kind="primary")],
        key_papers=["paper:one"],
        evidence_chunks=["chunk:c1"],
    )
    wstore.upsert_entry(target, reason="acceptance target")
    wstore.upsert_entry(duplicate, reason="acceptance duplicate")
    wstore.merge_entries(duplicate.entry_id, target.entry_id, reason="acceptance redirect")

    result = resolve_evidence_wiki_context(
        "How does the method work?",
        [{"chunk_id": "chunk:c1", "paper_id": "paper:one"}],
        max_entries=1,
    )
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert set(entry) == {"name", "definition"}
    assert "[chunk:" not in entry["definition"]

    sqlite_store.get_engine().dispose()
    sqlite_store._ENGINE = None
    return {"status": "accepted", "entry": entry}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args(argv)
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
        result = _run(args.workdir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="paper-rag-wiki-") as temp_dir:
            result = _run(Path(temp_dir))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
