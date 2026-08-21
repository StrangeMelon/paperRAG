"""Reference-chunk identification and query-intent policy."""

from __future__ import annotations

import json
import re

_REFERENCE_SECTIONS = frozenset({"references", "bibliography", "参考文献"})
_REFERENCE_QUERY_PATTERNS = (
    re.compile(r"参考\s*文\s*献"),
    re.compile(r"文献\s*列表"),
    re.compile(r"被引\s*(?:工作|论文|文献)"),
    re.compile(r"引用(?:了)?(?:哪些|什么)(?:论文|文献|工作)"),
    re.compile(r"\bbibliograph(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\breference\s+list\b", re.IGNORECASE),
    re.compile(r"\bcitation\s+list\b", re.IGNORECASE),
    re.compile(r"\bcited\s+(?:papers?|works?|references?)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:papers?|works?)\b[^?]*\bcited\b", re.IGNORECASE),
)


def chunk_metadata(chunk: dict) -> dict:
    """Return a normalized metadata mapping from retrieval or SQLite-shaped chunks."""
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)

    raw_metadata = chunk.get("metadata_json")
    if not isinstance(raw_metadata, str):
        return {}
    try:
        decoded = json.loads(raw_metadata)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _normalize_section(value: object) -> str:
    return "".join(str(value or "").casefold().split())


def is_reference_chunk(chunk: dict, *, legacy_section_fallback: bool = True) -> bool:
    """Return whether a retrieval result belongs to a references section."""
    metadata = chunk_metadata(chunk)
    if isinstance(metadata.get("is_references"), bool):
        return metadata["is_references"]
    if not legacy_section_fallback:
        return False
    return _normalize_section(chunk.get("section")) in _REFERENCE_SECTIONS


def detect_reference_intent(query: str) -> bool:
    """Return whether the query explicitly asks for bibliographic references."""
    normalized = " ".join(str(query or "").split())
    return any(pattern.search(normalized) for pattern in _REFERENCE_QUERY_PATTERNS)


def _ranking_score(chunk: dict) -> float:
    for field in ("score_rerank", "score_rrf", "score_dense", "score", "score_bm25"):
        value = chunk.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def apply_reference_ranking(
    chunks: list[dict],
    *,
    reference_intent: bool,
    penalty: float,
    enabled: bool = True,
    legacy_section_fallback: bool = True,
) -> list[dict]:
    """Copy and sort chunks by a reference-aware effective relevance score."""
    ranked: list[dict] = []
    for chunk in chunks:
        item = dict(chunk)
        raw_score = _ranking_score(item)
        rerank_score = item.get("score_rerank")
        if isinstance(rerank_score, (int, float)):
            item["score_rerank_raw"] = float(rerank_score)
        penalized = bool(
            enabled
            and not reference_intent
            and is_reference_chunk(item, legacy_section_fallback=legacy_section_fallback)
        )
        item["score_effective"] = raw_score * penalty if penalized else raw_score
        item["reference_penalized"] = penalized
        ranked.append(item)
    ranked.sort(key=lambda item: item["score_effective"], reverse=True)
    return ranked


def filter_answer_evidence(
    chunks: list[dict],
    *,
    reference_intent: bool,
    enabled: bool = True,
    exclude_from_evidence: bool = True,
    legacy_section_fallback: bool = True,
) -> list[dict]:
    """Remove bibliographic chunks from ordinary answer evidence."""
    if not enabled or reference_intent or not exclude_from_evidence:
        return list(chunks)
    return [
        chunk
        for chunk in chunks
        if not is_reference_chunk(chunk, legacy_section_fallback=legacy_section_fallback)
    ]


__all__ = [
    "apply_reference_ranking",
    "chunk_metadata",
    "detect_reference_intent",
    "filter_answer_evidence",
    "is_reference_chunk",
]
