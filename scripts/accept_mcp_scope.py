#!/usr/bin/env python3
"""Real SQLite acceptance for strict MCP paper scope validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml


def _write_config(workdir: Path) -> Path:
    config_path = workdir / "acceptance.yaml"
    data_root = workdir / "data"
    raw = {
        "paths": {
            "data_root": str(data_root),
            "papers_dir": str(data_root / "papers"),
            "parsed_dir": str(data_root / "parsed"),
            "index_dir": str(data_root / "index"),
            "sqlite_path": str(data_root / "index" / "papers.sqlite"),
            "bm25_path": str(data_root / "index" / "bm25.pkl"),
            "models_dir": str(data_root / "index" / "models"),
        }
    }
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _run(workdir: Path) -> dict[str, object]:
    config_path = _write_config(workdir)
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)

    from sqlmodel import Session

    from paper_rag import config as cfg
    from paper_rag.mcp.errors import InvalidPaperScopeError
    from paper_rag.rag.evidence_retrieval import Principal, validate_paper_scope
    from paper_rag.store import sqlite_store

    cfg.load.cache_clear()
    sqlite_store._ENGINE = None
    for paper_id, status, user_id in (
        ("paper:public", "done", "system"),
        ("paper:owned", "done", "user-a"),
        ("paper:pending", "indexed", "user-a"),
        ("paper:private", "done", "user-b"),
    ):
        sqlite_store.upsert_paper(
            {"paper_id": paper_id, "title": paper_id},
            status=status,
        )
        with Session(sqlite_store.get_engine()) as session:
            paper = session.get(sqlite_store.Paper, paper_id)
            assert paper is not None
            paper.user_id = user_id
            session.add(paper)
            session.commit()

    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    accepted_scope = validate_paper_scope(
        ["paper:public", "paper:owned", "paper:public"],
        principal=principal,
    )
    try:
        validate_paper_scope(
            ["paper:public", "paper:missing", "paper:pending", "paper:private"],
            principal=principal,
        )
    except InvalidPaperScopeError as exc:
        atomic_failure_ids = exc.paper_ids
    else:
        raise AssertionError("mixed invalid paper scope unexpectedly passed")

    database_path = Path(cfg.load().paths.sqlite_path)
    sqlite_store.get_engine().dispose()
    sqlite_store._ENGINE = None
    return {
        "accepted_scope": accepted_scope,
        "atomic_failure_ids": atomic_failure_ids,
        "database_exists": database_path.is_file(),
        "status": "accepted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args(argv)

    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        result = _run(args.workdir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="paper-rag-mcp-scope-") as temp_dir:
            result = _run(Path(temp_dir))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
