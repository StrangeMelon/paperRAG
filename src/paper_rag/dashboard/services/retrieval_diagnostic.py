"""On-demand retrieval diagnostics for the pipeline monitoring page."""

from __future__ import annotations

from typing import Any

from ...retrieve.pipeline import retrieve_round_with_rewrite

_TIMING_KEYS = (
    "query_rewrite_ms",
    "dense_ms",
    "sparse_ms",
    "rrf_ms",
    "rerank_ms",
    "diversify_ms",
)


def run_retrieval_diagnostic(
    query: str,
    *,
    paper_ids: list[str] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Run one retrieval round and retain every intermediate ranked list."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    chunks, rewrite, _ = retrieve_round_with_rewrite(
        query,
        paper_ids,
        top_k,
        timings=timings,
        diagnostics=diagnostics,
    )
    diagnostics.setdefault("rewrite", rewrite)
    total = sum(float(timings.get(key, 0.0)) for key in _TIMING_KEYS)
    timings["retrieval_total_ms"] = round(total, 1)
    return {
        "query": query,
        "chunks": chunks,
        "timings_ms": timings,
        "stages": _stage_payloads(diagnostics, timings),
    }


def _stage_payloads(diagnostics: dict[str, Any], timings: dict[str, float]) -> dict[str, dict]:
    rewrite = diagnostics.get("rewrite") or {}
    return {
        "Query Rewrite": {
            "timing_ms": timings.get("query_rewrite_ms", 0.0),
            "rewritten_queries": list(rewrite.get("dense_queries") or []),
            "bm25_query": str(rewrite.get("bm25_query") or ""),
        },
        "Dense": _ranked_stage("dense", timings.get("dense_ms", 0.0), diagnostics),
        "Sparse": _ranked_stage("sparse", timings.get("sparse_ms", 0.0), diagnostics),
        "RRF": _ranked_stage("rrf", timings.get("rrf_ms", 0.0), diagnostics),
        "Rerank": _ranked_stage("rerank", timings.get("rerank_ms", 0.0), diagnostics),
        "Diversify": _ranked_stage("diversify", timings.get("diversify_ms", 0.0), diagnostics),
    }


def _ranked_stage(name: str, timing_ms: float, diagnostics: dict[str, Any]) -> dict[str, Any]:
    items = [dict(item) for item in diagnostics.get(name, [])]
    # Multiple rewritten queries can report the same chunk. Keep its strongest score
    # while preserving the stage's ranked order for a readable diagnostic view.
    unique: dict[str, dict] = {}
    for item in items:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id not in unique:
            unique[chunk_id] = item
            continue
        if _score(item) > _score(unique[chunk_id]):
            unique[chunk_id] = item
    ordered = list(unique.values())
    if name != "diversify":
        ordered.sort(key=_score, reverse=True)
    return {"timing_ms": round(float(timing_ms), 1), "items": ordered}


def _score(item: dict[str, Any]) -> float:
    for key in ("score_rerank", "score_rrf", "score_bm25", "score_dense", "score"):
        try:
            if item.get(key) is not None:
                return float(item[key])
        except (TypeError, ValueError):
            continue
    return float("-inf")
