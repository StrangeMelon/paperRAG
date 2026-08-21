"""Evidence retrieval domain service contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from paper_rag.mcp.errors import InvalidPaperScopeError, RetrievalUnavailableError
from paper_rag.rag import evidence_retrieval as retrieval
from paper_rag.rag.evidence_retrieval import (
    Principal,
    RetrievalDependencies,
    retrieve_evidence,
    validate_paper_scope,
)


def _paper(paper_id: str, *, status: str = "done", user_id: str = "system"):
    return SimpleNamespace(paper_id=paper_id, status=status, user_id=user_id)


def test_validate_scope_allows_none_without_store_access() -> None:
    called = False

    def get_papers(_paper_ids):
        nonlocal called
        called = True
        return []

    assert (
        validate_paper_scope(
            None,
            principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            get_papers_fn=get_papers,
        )
        is None
    )
    assert called is False


def test_validate_scope_rejects_empty_list_without_store_access() -> None:
    called = False

    def get_papers(_paper_ids):
        nonlocal called
        called = True
        return []

    with pytest.raises(InvalidPaperScopeError) as exc_info:
        validate_paper_scope(
            [],
            principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            get_papers_fn=get_papers,
        )

    assert exc_info.value.paper_ids == []
    assert called is False


def test_validate_scope_is_atomic_and_reports_every_invalid_id() -> None:
    requested = ["paper:ok", "paper:missing", "paper:private", "paper:pending"]
    papers = [
        _paper("paper:ok"),
        _paper("paper:private", user_id="user-b"),
        _paper("paper:pending", status="indexed", user_id="user-a"),
    ]

    with pytest.raises(InvalidPaperScopeError) as exc_info:
        validate_paper_scope(
            requested,
            principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            get_papers_fn=lambda paper_ids: papers,
        )

    assert exc_info.value.paper_ids == ["paper:missing", "paper:private", "paper:pending"]


def test_validate_scope_deduplicates_only_after_all_ids_are_valid() -> None:
    requested = ["paper:one", "paper:two", "paper:one"]

    result = validate_paper_scope(
        requested,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        get_papers_fn=lambda paper_ids: [
            _paper("paper:one"),
            _paper("paper:two", user_id="user-a"),
        ],
    )

    assert result == ["paper:one", "paper:two"]


def test_admin_principal_can_access_other_users_papers() -> None:
    result = validate_paper_scope(
        ["paper:private"],
        principal=Principal(tenant_id="tenant-a", user_id="admin", is_admin=True),
        get_papers_fn=lambda paper_ids: [_paper("paper:private", user_id="user-b")],
    )

    assert result == ["paper:private"]


def _dependencies(**overrides) -> RetrievalDependencies:
    values = {
        "validate_scope": lambda paper_ids, principal: paper_ids,
        "resolve_wiki": lambda question, paper_ids: {"entries": [], "fingerprint": ""},
        "classify_intent": lambda question: {
            "intent": "reasoning",
            "top_k": 2,
            "max_iter": 2,
            "rrf_k": 60,
        },
        "retrieve_round": lambda query, paper_ids, top_k, wiki_context: [],
        "reflect": lambda question, evidence: {
            "sufficiency": "sufficient",
            "follow_up": "",
        },
        "decide_abstain": lambda chunks: {
            "decision": "confident",
            "evidence_score": 0.9,
        },
        "select_evidence": lambda question, chunks, intent, limit: (
            chunks[:limit],
            {"selected_chunk_ids": [chunk["chunk_id"] for chunk in chunks[:limit]]},
        ),
        "new_retrieval_id": lambda: "r_fixed",
    }
    values.update(overrides)
    return RetrievalDependencies(**values)


def _chunk(chunk_id: str, *, paper_id: str = "paper:one", score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": "Paper",
        "section": "Method",
        "page": 1,
        "modality": "text",
        "text": f"evidence {chunk_id}",
        "score_rerank": score,
    }


def test_retrieve_evidence_validates_scope_before_any_model_or_store_work() -> None:
    downstream_calls: list[str] = []

    def reject_scope(paper_ids, principal):
        raise InvalidPaperScopeError(["paper:bad"])

    dependencies = _dependencies(
        validate_scope=reject_scope,
        resolve_wiki=lambda *args: downstream_calls.append("wiki"),
        classify_intent=lambda *args: downstream_calls.append("intent"),
        retrieve_round=lambda *args: downstream_calls.append("retrieve"),
    )

    with pytest.raises(InvalidPaperScopeError):
        retrieve_evidence(
            "question",
            paper_ids=["paper:bad"],
            max_evidence=4,
            include_wiki=True,
            wiki_max_entries=3,
            principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            dependencies=dependencies,
        )

    assert downstream_calls == []


def test_retrieve_evidence_iterates_on_reflection_and_deduplicates_candidates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=3, enable_reflect=True)),
    )
    queries: list[str] = []
    rounds = [
        ([_chunk("c1"), _chunk("c2")], {"dense_queries": ["question"]}),
        ([_chunk("c2"), _chunk("c3", paper_id="paper:two")], {"dense_queries": ["follow"]}),
    ]

    def retrieve_round(query, paper_ids, top_k, wiki_context):
        queries.append(query)
        return rounds.pop(0)

    reflects = [
        {"sufficiency": "insufficient", "follow_up": "follow"},
    ]
    execution = retrieve_evidence(
        "question",
        paper_ids=["paper:one", "paper:two"],
        max_evidence=2,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=retrieve_round,
            reflect=lambda question, evidence: reflects.pop(0),
        ),
    )

    assert queries == ["question", "follow"]
    assert [chunk["chunk_id"] for chunk in execution.candidate_chunks] == ["c1", "c2", "c3"]
    assert [chunk["chunk_id"] for chunk in execution.evidence_chunks] == ["c1", "c2"]
    assert execution.allowed_chunk_ids == ["c1", "c2"]
    assert execution.public_decision == "confident"
    assert execution.trace["rewrites"] == [
        {"dense_queries": ["question"]},
        {"dense_queries": ["follow"]},
    ]
    for iteration in execution.trace["iters"]:
        assert iteration["retrieval_latency_ms"] >= 0
        assert iteration["iteration_latency_ms"] >= iteration["retrieval_latency_ms"]
    assert execution.trace["iters"][0]["reflect_latency_ms"] >= 0


def test_retrieve_evidence_records_zero_result_retrieval_latency(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=1, enable_reflect=True)),
    )

    execution = retrieve_evidence(
        "question",
        paper_ids=None,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(),
    )

    iteration = execution.trace["iters"][0]
    assert iteration["n_retrieved"] == 0
    assert iteration["retrieval_latency_ms"] >= 0
    assert iteration["iteration_latency_ms"] >= iteration["retrieval_latency_ms"]


def test_retrieve_evidence_no_chunks_returns_empty_public_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=3, enable_reflect=True)),
    )

    execution = retrieve_evidence(
        "question",
        paper_ids=None,
        max_evidence=4,
        include_wiki=True,
        wiki_max_entries=3,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(),
    )

    assert execution.public_decision == "no_evidence"
    assert execution.internal_decision == "no_chunks"
    assert execution.evidence_chunks == []
    assert execution.wiki_entries == []


def test_retrieve_evidence_abstention_does_not_leak_low_relevance_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=3, enable_reflect=True)),
    )
    low_chunks = [_chunk("low", score=0.01)]

    execution = retrieve_evidence(
        "question",
        paper_ids=None,
        max_evidence=4,
        include_wiki=True,
        wiki_max_entries=3,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=lambda *args: low_chunks,
            decide_abstain=lambda chunks: {"decision": "no_evidence", "evidence_score": 0.01},
        ),
    )

    assert execution.public_decision == "no_evidence"
    assert execution.candidate_chunks == low_chunks
    assert execution.evidence_chunks == []
    assert execution.allowed_chunk_ids == []


def test_regular_query_filters_references_before_abstain_and_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=1, enable_reflect=False)),
    )
    reference = {
        **_chunk("ref", score=0.99),
        "section": "References",
        "metadata": {"is_references": True},
    }
    body = _chunk("body", score=0.4)
    abstain_inputs: list[list[str]] = []

    def decide_abstain(chunks):
        abstain_inputs.append([chunk["chunk_id"] for chunk in chunks])
        return {"decision": "confident", "evidence_score": 0.4}

    execution = retrieve_evidence(
        "How does the method work?",
        paper_ids=None,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=lambda *args: [reference, body],
            decide_abstain=decide_abstain,
        ),
    )

    assert [chunk["chunk_id"] for chunk in execution.candidate_chunks] == ["ref", "body"]
    assert abstain_inputs == [["body"]]
    assert [chunk["chunk_id"] for chunk in execution.evidence_chunks] == ["body"]
    assert execution.trace["reference_policy"]["excluded_chunk_ids"] == ["ref"]


def test_regular_query_with_reference_only_candidates_returns_no_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=1, enable_reflect=False)),
    )
    reference = {
        **_chunk("ref", score=0.99),
        "section": "References",
        "metadata": {"is_references": True},
    }

    execution = retrieve_evidence(
        "How does the method work?",
        paper_ids=None,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(retrieve_round=lambda *args: [reference]),
    )

    assert execution.internal_decision == "no_evidence"
    assert execution.public_decision == "no_evidence"
    assert execution.candidate_chunks == [reference]
    assert execution.evidence_chunks == []
    assert execution.trace["abstain"]["reason"] == "reference_only"


def test_regular_reference_only_round_does_not_call_reflect(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=2, enable_reflect=True)),
    )
    reference = {
        **_chunk("ref", score=0.99),
        "section": "References",
        "metadata": {"is_references": True},
    }

    execution = retrieve_evidence(
        "How does the method work?",
        paper_ids=None,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=lambda *args: [reference],
            reflect=lambda *args: pytest.fail("reference-only evidence must not reach reflect"),
        ),
    )

    assert execution.public_decision == "no_evidence"
    assert execution.trace["stopped_by"] == "reference_only"


def test_reference_intent_is_forwarded_across_reflection_rounds(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=2, enable_reflect=True)),
    )
    calls: list[tuple[str, bool | None]] = []

    def retrieve_round(
        query,
        paper_ids,
        top_k,
        wiki_context,
        timings=None,
        *,
        reference_intent=None,
    ):
        calls.append((query, reference_intent))
        return [
            {
                **_chunk(f"ref-{len(calls)}"),
                "section": "References",
                "metadata": {"is_references": True},
            }
        ]

    reflections = [{"sufficiency": "insufficient", "follow_up": "Which specific works?"}]
    execution = retrieve_evidence(
        "Which papers are cited by this article?",
        paper_ids=None,
        max_evidence=2,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=retrieve_round,
            reflect=lambda question, evidence: reflections.pop(0),
        ),
    )

    assert calls == [
        ("Which papers are cited by this article?", True),
        ("Which specific works?", True),
    ]
    assert [chunk["chunk_id"] for chunk in execution.evidence_chunks] == ["ref-1", "ref-2"]


def test_retrieve_evidence_attaches_post_wiki_only_to_selected_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=3, enable_reflect=True)),
    )
    calls: list[tuple[str, list[dict], int]] = []

    def resolve_post(question, evidence, max_entries):
        calls.append((question, evidence, max_entries))
        return [{"name": "RAG", "definition": "Background"}]

    execution = retrieve_evidence(
        "question",
        paper_ids=None,
        max_evidence=2,
        include_wiki=True,
        wiki_max_entries=1,
        principal=Principal(tenant_id="tenant-a", user_id="user-a"),
        dependencies=_dependencies(
            retrieve_round=lambda *args: ([_chunk("c1")], {}),
            resolve_evidence_wiki=resolve_post,
        ),
    )

    assert len(calls) == 1
    assert [chunk["chunk_id"] for chunk in calls[0][1]] == ["c1"]
    assert calls[0][2] == 1
    assert execution.wiki_entries == [{"name": "RAG", "definition": "Background"}]


def test_retrieve_evidence_maps_retrieval_infrastructure_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval.cfg,
        "load",
        lambda: SimpleNamespace(rag=SimpleNamespace(max_inner_iters=3, enable_reflect=True)),
    )

    with pytest.raises(RetrievalUnavailableError):
        retrieve_evidence(
            "question",
            paper_ids=None,
            max_evidence=2,
            include_wiki=False,
            wiki_max_entries=0,
            principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            dependencies=_dependencies(
                retrieve_round=lambda *args: (_ for _ in ()).throw(RuntimeError("stores down"))
            ),
        )
