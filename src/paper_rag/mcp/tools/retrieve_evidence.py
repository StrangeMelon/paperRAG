"""Adapter for the paper_retrieve_evidence MCP tool."""

from __future__ import annotations

import json
from typing import Any

from ...rag.evidence_retrieval import Principal, RetrievalExecution, retrieve_evidence
from ..runtime import McpRuntime
from ..schemas import (
    EvidenceItem,
    RetrieveEvidenceAbstained,
    RetrieveEvidenceInput,
    RetrieveEvidenceSuccess,
    WikiItem,
)
from ..trace_store import RetrievalTraceStore

_MAX_EVIDENCE_TEXT_CHARS = 6000
_MAX_PUBLIC_RESPONSE_CHARS = 16000


def _clip_text(text: str, limit: int = _MAX_EVIDENCE_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _evidence_item(chunk: dict[str, Any]) -> EvidenceItem:
    chunk_id = str(chunk.get("chunk_id") or "")
    return EvidenceItem(
        citation=f"[chunk:{chunk_id}]",
        paper_id=str(chunk.get("paper_id") or ""),
        title=str(chunk.get("title") or ""),
        section=chunk.get("section"),
        page=chunk.get("page"),
        modality=str(chunk.get("modality") or "text"),
        text=_clip_text(str(chunk.get("text") or chunk.get("raw_snippet") or "")),
    )


def _response_size(response: dict[str, Any]) -> int:
    return len(json.dumps(response, ensure_ascii=False, separators=(",", ":")))


def _enforce_response_budget(response: dict[str, Any]) -> dict[str, Any]:
    wiki = response.get("wiki") or []
    while wiki and _response_size(response) > _MAX_PUBLIC_RESPONSE_CHARS:
        wiki.pop()
    evidence = response.get("evidence") or []
    while len(evidence) > 1 and _response_size(response) > _MAX_PUBLIC_RESPONSE_CHARS:
        evidence.pop()
    if evidence and _response_size(response) > _MAX_PUBLIC_RESPONSE_CHARS:
        excess = _response_size(response) - _MAX_PUBLIC_RESPONSE_CHARS
        text = evidence[0].get("text", "")
        keep = max(3, len(text) - excess - 3)
        evidence[0]["text"] = text[:keep].rstrip() + "..."
    return response


def build_public_response(execution: RetrievalExecution) -> dict[str, Any]:
    """Serialize only the compact evidence contract sent to an MCP host."""
    if execution.public_decision == "no_evidence":
        return RetrieveEvidenceAbstained(
            retrieval_id=execution.retrieval_id,
        ).model_dump(exclude_none=True)

    evidence = [_evidence_item(chunk) for chunk in execution.evidence_chunks]
    wiki = [
        WikiItem(
            name=str(item.get("name") or ""),
            definition=str(item.get("definition") or ""),
        )
        for item in execution.wiki_entries
    ]
    response = RetrieveEvidenceSuccess(
        decision=execution.public_decision,
        retrieval_id=execution.retrieval_id,
        evidence=evidence,
        wiki=wiki,
    )
    return _enforce_response_budget(response.model_dump(exclude_none=True))


async def paper_retrieve_evidence(
    args: RetrieveEvidenceInput | dict[str, Any],
    *,
    runtime: McpRuntime,
    trace_store: RetrievalTraceStore,
    principal: Principal,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute the evidence service from an async MCP tool boundary."""
    parsed = (
        args
        if isinstance(args, RetrieveEvidenceInput)
        else RetrieveEvidenceInput.model_validate(args)
    )
    execution = await runtime.run_sync(
        retrieve_evidence,
        parsed.query,
        paper_ids=parsed.paper_ids,
        max_evidence=parsed.max_evidence,
        include_wiki=parsed.include_wiki,
        wiki_max_entries=parsed.wiki_max_entries,
        principal=principal,
        timeout=timeout,
    )
    trace_store.put(execution, principal)
    return build_public_response(execution)


__all__ = ["build_public_response", "paper_retrieve_evidence"]
