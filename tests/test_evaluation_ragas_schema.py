"""Contracts for the RAGAS-only Golden Set and report models."""

from __future__ import annotations

import json

import pytest


def _write_golden(path, cases: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ragas-eval.v1",
                "corpus": {"selection": "all_indexed"},
                "test_cases": cases,
            }
        ),
        encoding="utf-8",
    )


def test_load_ragas_golden_set_uses_an_independent_schema(tmp_path) -> None:
    from paper_rag.evaluation.ragas_schema import load_ragas_golden_set

    path = tmp_path / "ragas.json"
    _write_golden(
        path,
        [
            {
                "id": "r1",
                "query": "What is RAG?",
                "paper_ids": ["p1"],
                "reference_answer": "RAG retrieves evidence before generation.",
                "reference_chunk_ids": ["c1"],
                "expected_abstain": False,
                "tags": ["factual", "en"],
            }
        ],
    )

    cases, digest = load_ragas_golden_set(path)

    assert cases[0].id == "r1"
    assert cases[0].reference_chunk_ids == ["c1"]
    assert len(digest) == 64


def test_load_ragas_golden_set_rejects_duplicate_ids(tmp_path) -> None:
    from paper_rag.evaluation.ragas_schema import RagasGoldenSetError, load_ragas_golden_set

    case = {
        "id": "duplicate",
        "query": "q",
        "paper_ids": ["p1"],
        "reference_answer": "a",
        "reference_chunk_ids": ["c1"],
        "expected_abstain": False,
        "tags": ["factual"],
    }
    path = tmp_path / "ragas.json"
    _write_golden(path, [case, case])

    with pytest.raises(RagasGoldenSetError, match="duplicate"):
        load_ragas_golden_set(path)


def test_load_ragas_golden_set_allows_explicit_abstain_case_without_reference(tmp_path) -> None:
    from paper_rag.evaluation.ragas_schema import load_ragas_golden_set

    path = tmp_path / "ragas.json"
    _write_golden(
        path,
        [
            {
                "id": "negative",
                "query": "Unknown topic",
                "paper_ids": [],
                "expected_abstain": True,
                "tags": ["negative"],
            }
        ],
    )

    cases, _ = load_ragas_golden_set(path)

    assert cases[0].expected_abstain is True
    assert cases[0].reference_answer is None
