"""真实验收参考文献 chunk 的检索降权与证据隔离策略。

本脚本不 mock SQLite、FTS5、Qdrant、embedding、reranker、检索 pipeline、
abstain 或 evidence selection。它在独立目录中创建真实解析产物和 embedded
Qdrant collection, 验证普通问题、参考文献问题和 reference-only 三条路径。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "demo-reference-policy-data"
sys.path.insert(0, str(REPO_ROOT))


def _write_config() -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        "data_root": str(WORK_ROOT / "data"),
        "papers_dir": str(WORK_ROOT / "data/papers"),
        "parsed_dir": str(WORK_ROOT / "data/parsed"),
        "index_dir": str(WORK_ROOT / "data/index"),
        "sqlite_path": str(WORK_ROOT / "data/index/papers.sqlite"),
        "bm25_path": str(WORK_ROOT / "data/index/bm25.pkl"),
        "models_dir": str(REPO_ROOT / "data/index/models"),
    }
    raw["qdrant"]["local_path"] = str(WORK_ROOT / "qdrant")
    raw["qdrant"]["collection_chunks"] = "reference_policy_chunks"
    raw["qdrant"]["collection_wiki"] = "reference_policy_wiki"
    raw["vision"]["enabled"] = False
    raw["rag"]["enable_reflect"] = False
    raw["rag"]["intent"]["enabled"] = False
    config_path = WORK_ROOT / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return config_path


def _write_parsed_paper(paper_id: str, markdown: str) -> Path:
    from paper_rag.utils.ids import to_safe_dirname

    parsed_dir = WORK_ROOT / "data/parsed" / to_safe_dirname(paper_id)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "paper.md").write_text(markdown, encoding="utf-8")
    (parsed_dir / "language.json").write_text(
        json.dumps({"document_language": "en"}),
        encoding="utf-8",
    )
    return parsed_dir


def _prepare_corpus() -> tuple[list[dict], dict[str, list[dict]]]:
    from paper_rag.chunk.builder import build_chunks
    from paper_rag.store import sqlite_store

    papers = {
        "accept:mixed": (
            "Adaptive Evidence Retrieval",
            """# Method

The adaptive evidence gate improves retrieval quality by scoring source passages,
rejecting bibliographic-only matches, and applying a calibrated evidence threshold.
The contextual encoder keeps method evidence while suppressing citation-list noise.

# References

[1] Smith, J. Adaptive Evidence Gate Improves Retrieval Quality by Scoring Source
Passages. Journal of Retrieval Systems, 2024.

[2] Doe, A. Calibrated Evidence Thresholds for Contextual Retrieval. 2023.
""",
        ),
        "accept:reference-only": (
            "Bibliographic Record Only",
            """# References

[1] Smith, J. Adaptive Evidence Gate Improves Retrieval Quality by Scoring Source
Passages and Rejecting Bibliographic Matches. Journal of Retrieval Systems, 2024.
""",
        ),
    }

    chunks_by_paper: dict[str, list[dict]] = {}
    all_chunks: list[dict] = []
    for paper_id, (title, markdown) in papers.items():
        parsed_dir = _write_parsed_paper(paper_id, markdown)
        sections, chunks = build_chunks(paper_id, parsed_dir, title=title)
        sqlite_store.upsert_paper(
            {"paper_id": paper_id, "title": title, "authors": ["Acceptance Test"]},
            status="done",
        )
        sqlite_store.upsert_sections_and_chunks(paper_id, sections, chunks)
        chunks_by_paper[paper_id] = chunks
        all_chunks.extend(chunks)
    return all_chunks, chunks_by_paper


def _index_corpus(chunks: list[dict], paper_ids: list[str]) -> None:
    from paper_rag.embed import bge_m3
    from paper_rag.retrieve import fts5
    from paper_rag.store import qdrant_store

    vectors = bge_m3.encode([str(chunk["context_text"]) for chunk in chunks])
    assert len(vectors) == len(chunks)
    assert qdrant_store.upsert_chunks(chunks, vectors) == len(chunks)
    for paper_id in paper_ids:
        assert fts5.sync_paper(paper_id) > 0


def _chunk_summary(chunks: list[dict]) -> list[dict[str, Any]]:
    from paper_rag.retrieve.reference_policy import is_reference_chunk

    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "paper_id": chunk.get("paper_id"),
            "section": chunk.get("section"),
            "is_references": is_reference_chunk(chunk),
            "score_rerank_raw": chunk.get("score_rerank_raw"),
            "score_effective": chunk.get("score_effective"),
            "reference_penalized": chunk.get("reference_penalized"),
        }
        for chunk in chunks
    ]


def _verify_storage(chunks_by_paper: dict[str, list[dict]]) -> dict[str, Any]:
    from paper_rag.retrieve import dense, fts5
    from paper_rag.retrieve.reference_policy import is_reference_chunk
    from paper_rag.store import sqlite_store

    mixed = chunks_by_paper["accept:mixed"]
    reference_ids = {chunk["chunk_id"] for chunk in mixed if is_reference_chunk(chunk)}
    assert reference_ids, "builder did not mark a References chunk"

    for chunk_id in reference_ids:
        stored = sqlite_store.get_chunk(chunk_id)
        assert stored is not None
        assert json.loads(stored.metadata_json)["is_references"] is True

    fts_hits = fts5.search("Smith adaptive evidence gate", paper_ids=["accept:mixed"])
    assert any(hit["chunk_id"] in reference_ids and is_reference_chunk(hit) for hit in fts_hits), (
        "FTS5 did not return reference metadata"
    )

    qdrant_chunks = dense.retrieve(
        "Smith adaptive evidence gate",
        top_k=10,
        paper_ids=["accept:mixed"],
    )
    assert any(
        chunk["chunk_id"] in reference_ids and is_reference_chunk(chunk) for chunk in qdrant_chunks
    ), "Qdrant did not preserve reference metadata"
    return {
        "reference_chunk_ids": sorted(reference_ids),
        "fts5_hits": _chunk_summary(fts_hits),
        "qdrant_chunk_count": len(qdrant_chunks),
    }


def _verify_pipeline() -> dict[str, Any]:
    from paper_rag.retrieve.pipeline import retrieve_round_with_rewrite
    from paper_rag.retrieve.reference_policy import is_reference_chunk

    diagnostics: dict[str, Any] = {}
    timings: dict[str, float] = {}
    chunks, _rewrite, reported = retrieve_round_with_rewrite(
        "How does the adaptive evidence gate improve retrieval quality?",
        ["accept:mixed"],
        5,
        rewrite_enabled=False,
        timings=timings,
        diagnostics=diagnostics,
    )
    assert reported is timings
    assert chunks
    references = [chunk for chunk in chunks if is_reference_chunk(chunk)]
    bodies = [chunk for chunk in chunks if not is_reference_chunk(chunk)]
    assert references and bodies, "real retrieval must surface both body and reference candidates"
    assert all(chunk.get("reference_penalized") is True for chunk in references)
    assert all(chunk.get("score_rerank_raw") is not None for chunk in chunks), (
        "real BGE reranker did not produce raw scores"
    )
    assert diagnostics["reference_policy"]["intent"] is False
    return {
        "chunks": _chunk_summary(chunks),
        "timings_ms": timings,
        "reference_policy": diagnostics["reference_policy"],
    }


def _retrieve_evidence(query: str, paper_ids: list[str]):
    from paper_rag.rag.evidence_retrieval import Principal, retrieve_evidence

    return retrieve_evidence(
        query,
        paper_ids=paper_ids,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="acceptance", user_id="system"),
    )


def _verify_evidence_paths() -> dict[str, Any]:
    from paper_rag.retrieve.reference_policy import is_reference_chunk

    ordinary = _retrieve_evidence(
        "How does the adaptive evidence gate improve retrieval quality?",
        ["accept:mixed"],
    )
    assert ordinary.evidence_chunks, "ordinary question lost all body evidence"
    assert not any(is_reference_chunk(chunk) for chunk in ordinary.evidence_chunks)
    assert ordinary.trace["reference_policy"]["excluded_chunk_ids"]

    explicit = _retrieve_evidence(
        "Which papers are cited? Smith adaptive evidence gate improves retrieval quality 2024",
        ["accept:mixed"],
    )
    assert explicit.trace["reference_policy"]["intent"] is True
    assert any(is_reference_chunk(chunk) for chunk in explicit.evidence_chunks), (
        "explicit reference question did not receive bibliographic evidence"
    )
    explicit_references = [
        chunk for chunk in explicit.candidate_chunks if is_reference_chunk(chunk)
    ]
    assert explicit_references
    assert all(chunk.get("reference_penalized") is False for chunk in explicit_references)

    reference_only = _retrieve_evidence(
        "How does the adaptive evidence gate improve retrieval quality?",
        ["accept:reference-only"],
    )
    assert reference_only.candidate_chunks, "reference-only corpus was not actually retrieved"
    assert all(is_reference_chunk(chunk) for chunk in reference_only.candidate_chunks)
    assert reference_only.public_decision == "no_evidence"
    assert reference_only.internal_decision == "no_evidence"
    assert reference_only.evidence_chunks == []
    assert reference_only.trace["abstain"]["reason"] == "reference_only"

    return {
        "ordinary": {
            "decision": ordinary.public_decision,
            "candidates": _chunk_summary(ordinary.candidate_chunks),
            "evidence": _chunk_summary(ordinary.evidence_chunks),
            "excluded_chunk_ids": ordinary.trace["reference_policy"]["excluded_chunk_ids"],
        },
        "explicit_reference": {
            "decision": explicit.public_decision,
            "candidates": _chunk_summary(explicit.candidate_chunks),
            "evidence": _chunk_summary(explicit.evidence_chunks),
        },
        "reference_only": {
            "decision": reference_only.public_decision,
            "candidates": _chunk_summary(reference_only.candidate_chunks),
            "evidence": _chunk_summary(reference_only.evidence_chunks),
            "abstain": reference_only.trace["abstain"],
        },
    }


def main() -> int:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    config_path = _write_config()
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    os.environ["PAPER_RAG_FORCE_LOCAL_REWRITE"] = "1"

    from paper_rag import config as cfg
    from paper_rag.store import qdrant_store
    from scripts import init_store

    cfg.load.cache_clear()
    init_store.main()

    chunks, chunks_by_paper = _prepare_corpus()
    _index_corpus(chunks, list(chunks_by_paper))
    report = {
        "storage": _verify_storage(chunks_by_paper),
        "pipeline": _verify_pipeline(),
        "evidence_paths": _verify_evidence_paths(),
    }
    report_path = WORK_ROOT / "acceptance-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    qdrant_store.close_client()

    print("REFERENCE POLICY ACCEPTANCE PASSED")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
