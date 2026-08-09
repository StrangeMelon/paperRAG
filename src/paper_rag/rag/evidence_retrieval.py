"""Shared evidence-retrieval domain service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .. import config as cfg
from ..mcp.errors import InvalidPaperScopeError, PaperRagToolError, RetrievalUnavailableError
from ..store.sqlite_store import get_papers_by_ids
from ..utils.logger import get_logger

log = get_logger("rag.evidence_retrieval")


class PaperRecord(Protocol):
    paper_id: str
    status: str
    user_id: str | None


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    is_admin: bool = False


@dataclass
class RetrievalExecution:
    retrieval_id: str
    public_decision: str
    internal_decision: str
    candidate_chunks: list[dict]
    evidence_chunks: list[dict]
    wiki_entries: list[dict]
    allowed_chunk_ids: list[str]
    trace: dict[str, Any]


@dataclass(frozen=True)
class RetrievalDependencies:
    validate_scope: Callable[[list[str] | None, Principal], list[str] | None]
    resolve_wiki: Callable[[str, list[str] | None], dict[str, Any]]
    classify_intent: Callable[[str], dict[str, Any]]
    retrieve_round: Callable[[str, list[str] | None, int, dict[str, Any]], Any]
    reflect: Callable[[str, str], dict[str, Any]]
    decide_abstain: Callable[[list[dict]], dict[str, Any]]
    select_evidence: Callable[[str, list[dict], str | None, int], tuple[list[dict], dict]]
    new_retrieval_id: Callable[[], str]
    resolve_evidence_wiki: Callable[[str, list[dict], int], list[dict] | dict[str, Any]] | None = (
        None
    )


def _is_accessible(paper: PaperRecord, principal: Principal) -> bool:
    return principal.is_admin or paper.user_id in {None, "system", principal.user_id}


def validate_paper_scope(
    paper_ids: list[str] | None,
    *,
    principal: Principal,
    get_papers_fn: Callable[[list[str]], list[PaperRecord]] = get_papers_by_ids,
) -> list[str] | None:
    """Validate the complete evidence scope before any retrieval-side work."""
    if paper_ids is None:
        return None
    if not paper_ids:
        raise InvalidPaperScopeError([])

    papers_by_id = {paper.paper_id: paper for paper in get_papers_fn(paper_ids)}
    invalid_ids: list[str] = []
    for paper_id in dict.fromkeys(paper_ids):
        paper = papers_by_id.get(paper_id)
        if paper is None or paper.status != "done" or not _is_accessible(paper, principal):
            invalid_ids.append(paper_id)

    if invalid_ids:
        raise InvalidPaperScopeError(invalid_ids)
    return list(dict.fromkeys(paper_ids))


def _resolve_wiki_safe(question: str, paper_ids: list[str] | None) -> dict[str, Any]:
    try:
        from ..wiki.context import resolve_wiki_context

        return resolve_wiki_context(question, paper_ids=paper_ids)
    except Exception as exc:
        log.warning(f"wiki context resolve failed (non-fatal): {exc}")
        return {"role": "background_not_evidence", "fingerprint": "", "entries": []}


def _retrieve_round_default(
    query: str,
    paper_ids: list[str] | None,
    top_k: int,
    wiki_context: dict[str, Any],
):
    from ..retrieve.pipeline import retrieve_round_with_rewrite

    return retrieve_round_with_rewrite(
        query,
        paper_ids,
        top_k,
        wiki_context=wiki_context,
    )


def _reflect_default(question: str, evidence: str) -> dict[str, Any]:
    from .reflect import reflect

    return reflect(question, evidence)


def _decide_abstain_default(chunks: list[dict]) -> dict[str, Any]:
    from .abstain import decide

    abstain_cfg = cfg.load().rag.abstain
    return decide(
        chunks,
        enabled=abstain_cfg.enabled,
        threshold_low=abstain_cfg.threshold_low,
        threshold_high=abstain_cfg.threshold_high,
        min_chunks=abstain_cfg.min_chunks,
    )


def _select_evidence_default(
    question: str,
    chunks: list[dict],
    intent: str | None,
    limit: int,
) -> tuple[list[dict], dict]:
    from .evidence_select import select_evidence

    return select_evidence(question, chunks, intent=intent, max_chunks=limit)


def _new_retrieval_id() -> str:
    from ..observability import new_trace_id

    return f"r_{new_trace_id()}"


def _resolve_evidence_wiki_default(
    question: str,
    evidence_chunks: list[dict],
    max_entries: int,
) -> list[dict]:
    from ..wiki.context import resolve_evidence_wiki_context

    return resolve_evidence_wiki_context(
        question,
        evidence_chunks,
        max_entries=max_entries,
    ).get("entries", [])


def build_default_dependencies() -> RetrievalDependencies:
    from .intent_classifier import classify

    return RetrievalDependencies(
        validate_scope=lambda paper_ids, principal: validate_paper_scope(
            paper_ids,
            principal=principal,
        ),
        resolve_wiki=_resolve_wiki_safe,
        classify_intent=classify,
        retrieve_round=_retrieve_round_default,
        reflect=_reflect_default,
        decide_abstain=_decide_abstain_default,
        select_evidence=_select_evidence_default,
        new_retrieval_id=_new_retrieval_id,
        resolve_evidence_wiki=_resolve_evidence_wiki_default,
    )


def _split_retrieval_result(result: Any) -> tuple[list[dict], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        chunks, rewrite = result
        return list(chunks), dict(rewrite or {})
    return list(result or []), {}


def _retrieve_loop(
    question: str,
    paper_ids: list[str] | None,
    *,
    top_k: int,
    max_iter: int,
    enable_reflect: bool,
    wiki_context: dict[str, Any],
    dependencies: RetrievalDependencies,
) -> tuple[list[dict], list[dict], list[dict], str]:
    from ..retrieve.format import format_evidence

    candidates: dict[str, dict] = {}
    iterations: list[dict] = []
    rewrites: list[dict] = []
    current_query = question
    stopped_by = "max_iters"

    for iteration in range(max_iter):
        result = dependencies.retrieve_round(
            current_query,
            paper_ids,
            top_k,
            wiki_context,
        )
        chunks, rewrite = _split_retrieval_result(result)
        if rewrite:
            rewrites.append(rewrite)
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in candidates:
                candidates[chunk_id] = chunk

        if not chunks:
            iterations.append({"query": current_query, "n_retrieved": 0, "reflect": None})
            stopped_by = "no_evidence"
            break

        if enable_reflect and iteration < max_iter - 1:
            reflection = dependencies.reflect(question, format_evidence(chunks))
            iterations.append(
                {"query": current_query, "n_retrieved": len(chunks), "reflect": reflection}
            )
            if reflection.get("sufficiency") == "sufficient":
                stopped_by = "answered"
                break
            follow_up = str(reflection.get("follow_up") or "").strip()
            if follow_up:
                current_query = follow_up
                continue
            stopped_by = "answered"
            break

        iterations.append({"query": current_query, "n_retrieved": len(chunks), "reflect": None})
        stopped_by = "answered"
        break

    return list(candidates.values()), iterations, rewrites, stopped_by


def retrieve_evidence(
    query: str,
    *,
    paper_ids: list[str] | None,
    max_evidence: int,
    include_wiki: bool,
    wiki_max_entries: int,
    principal: Principal,
    dependencies: RetrievalDependencies | None = None,
) -> RetrievalExecution:
    """Run the synchronous evidence pipeline without generating an answer."""
    deps = dependencies or build_default_dependencies()
    retrieval_id = deps.new_retrieval_id()
    validated_scope = deps.validate_scope(paper_ids, principal)

    wiki_context = deps.resolve_wiki(query, validated_scope)
    intent = deps.classify_intent(query)
    rag_cfg = cfg.load().rag
    top_k = int(intent["top_k"])
    max_iter = min(int(intent["max_iter"]), int(rag_cfg.max_inner_iters))
    try:
        candidates, iterations, rewrites, stopped_by = _retrieve_loop(
            query,
            validated_scope,
            top_k=top_k,
            max_iter=max_iter,
            enable_reflect=bool(rag_cfg.enable_reflect),
            wiki_context=wiki_context,
            dependencies=deps,
        )
    except PaperRagToolError:
        raise
    except Exception as exc:
        raise RetrievalUnavailableError(details={"stage": "retrieve"}) from exc
    final_candidates = candidates[: top_k * 2]

    if not final_candidates:
        internal_decision = "no_chunks"
        abstain = {
            "decision": "no_chunks",
            "evidence_score": 0.0,
            "n_chunks": 0,
        }
        evidence_chunks: list[dict] = []
        evidence_selection: dict[str, Any] = {}
    else:
        abstain = deps.decide_abstain(final_candidates)
        internal_decision = str(abstain["decision"])
        if internal_decision == "no_evidence":
            evidence_chunks = []
            evidence_selection = {}
        else:
            evidence_chunks, evidence_selection = deps.select_evidence(
                query,
                final_candidates,
                intent.get("intent"),
                max_evidence,
            )

    wiki_entries: list[dict] = []
    if include_wiki and evidence_chunks and wiki_max_entries > 0 and deps.resolve_evidence_wiki:
        try:
            result = deps.resolve_evidence_wiki(query, evidence_chunks, wiki_max_entries)
            wiki_entries = (
                result.get("entries", []) if isinstance(result, dict) else list(result or [])
            )
        except Exception as exc:
            log.warning(f"evidence wiki enrichment failed (non-fatal): {exc}")

    public_decision = (
        "no_evidence" if internal_decision in {"no_chunks", "no_evidence"} else internal_decision
    )
    allowed_chunk_ids = [
        str(chunk["chunk_id"]) for chunk in evidence_chunks if chunk.get("chunk_id")
    ]
    trace = {
        "query": query,
        "paper_scope": validated_scope,
        "intent": intent,
        "rewrites": rewrites,
        "iters": iterations,
        "stopped_by": stopped_by,
        "abstain": abstain,
        "evidence_selection": evidence_selection,
        "evidence_chunk_ids": allowed_chunk_ids,
        "wiki_context": wiki_context,
        "include_wiki": include_wiki,
        "wiki_max_entries": wiki_max_entries,
        "wiki_entries": wiki_entries,
    }
    return RetrievalExecution(
        retrieval_id=retrieval_id,
        public_decision=public_decision,
        internal_decision=internal_decision,
        candidate_chunks=final_candidates,
        evidence_chunks=evidence_chunks,
        wiki_entries=wiki_entries,
        allowed_chunk_ids=allowed_chunk_ids,
        trace=trace,
    )


__all__ = [
    "Principal",
    "RetrievalDependencies",
    "RetrievalExecution",
    "build_default_dependencies",
    "retrieve_evidence",
    "validate_paper_scope",
]
