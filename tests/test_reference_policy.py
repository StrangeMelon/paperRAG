"""Reference-chunk policy behavior contracts."""

from __future__ import annotations

import pytest

from paper_rag.retrieve import reference_policy
from paper_rag.retrieve.reference_policy import detect_reference_intent, is_reference_chunk


@pytest.mark.parametrize(
    "chunk",
    [
        {"metadata": {"is_references": True}, "section": "Methods"},
        {"metadata_json": '{"is_references": true}', "section": "Methods"},
        {"section": " References "},
        {"section": "BIBLIOGRAPHY"},
        {"section": "参 考 文 献"},
    ],
)
def test_is_reference_chunk_accepts_metadata_json_and_legacy_sections(chunk: dict) -> None:
    assert is_reference_chunk(chunk) is True


@pytest.mark.parametrize(
    "chunk",
    [
        {},
        {"metadata": {"is_references": False}, "section": "References"},
        {"metadata_json": "not-json", "section": "Methods"},
        {"section": "Reference Architecture"},
        {"section": "Related Work"},
    ],
)
def test_is_reference_chunk_rejects_non_reference_chunks(chunk: dict) -> None:
    assert is_reference_chunk(chunk) is False


def test_is_reference_chunk_can_disable_legacy_section_fallback() -> None:
    assert is_reference_chunk({"section": "References"}, legacy_section_fallback=False) is False


@pytest.mark.parametrize(
    "query",
    [
        "这篇论文的参考文献有哪些?",
        "作者引用了哪些论文?",
        "请列出文献列表",
        "论文讨论了哪些被引工作",
        "Show me the bibliography.",
        "Which papers are cited by this article?",
        "List the cited works",
        "What is included in the reference list?",
    ],
)
def test_detect_reference_intent_accepts_explicit_bibliographic_questions(query: str) -> None:
    assert detect_reference_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "回答时请引用证据",
        "请引用论文中的实验结果",
        "Cite evidence for every factual claim.",
        "How is citation precision evaluated?",
        "Explain the reference architecture.",
        "Related work 对这个方法有什么影响?",
        "",
    ],
)
def test_detect_reference_intent_rejects_citation_format_and_domain_questions(query: str) -> None:
    assert detect_reference_intent(query) is False


def test_apply_reference_ranking_uses_effective_score_without_mutating_input() -> None:
    ranker = getattr(reference_policy, "apply_reference_ranking", None)
    assert ranker is not None, "reference ranking policy is not implemented"
    chunks = [
        {
            "chunk_id": "ref",
            "score_rerank": 0.9,
            "metadata": {"is_references": True},
        },
        {"chunk_id": "body", "score_rerank": 0.4, "section": "Methods"},
    ]

    ranked = ranker(chunks, reference_intent=False, penalty=0.15)

    assert [chunk["chunk_id"] for chunk in ranked] == ["body", "ref"]
    reference = ranked[1]
    assert reference["score_rerank_raw"] == 0.9
    assert reference["score_effective"] == pytest.approx(0.135)
    assert reference["reference_penalized"] is True
    assert ranked[0]["score_effective"] == 0.4
    assert chunks[0] == {
        "chunk_id": "ref",
        "score_rerank": 0.9,
        "metadata": {"is_references": True},
    }


def test_apply_reference_ranking_bypasses_penalty_for_reference_intent() -> None:
    ranker = getattr(reference_policy, "apply_reference_ranking", None)
    assert ranker is not None, "reference ranking policy is not implemented"
    chunks = [
        {
            "chunk_id": "ref",
            "score_rerank": 0.9,
            "metadata": {"is_references": True},
        },
        {"chunk_id": "body", "score_rerank": 0.4},
    ]

    ranked = ranker(chunks, reference_intent=True, penalty=0.15)

    assert [chunk["chunk_id"] for chunk in ranked] == ["ref", "body"]
    assert ranked[0]["score_effective"] == 0.9
    assert ranked[0]["reference_penalized"] is False


def test_filter_answer_evidence_excludes_references_only_for_regular_queries() -> None:
    filter_evidence = getattr(reference_policy, "filter_answer_evidence", None)
    assert filter_evidence is not None, "reference evidence filter is not implemented"
    chunks = [
        {"chunk_id": "ref", "metadata": {"is_references": True}},
        {"chunk_id": "body", "section": "Methods"},
    ]

    regular = filter_evidence(chunks, reference_intent=False)
    explicit = filter_evidence(chunks, reference_intent=True)
    disabled = filter_evidence(chunks, reference_intent=False, enabled=False)

    assert [chunk["chunk_id"] for chunk in regular] == ["body"]
    assert [chunk["chunk_id"] for chunk in explicit] == ["ref", "body"]
    assert [chunk["chunk_id"] for chunk in disabled] == ["ref", "body"]
