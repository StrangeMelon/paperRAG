"""MCP public schema contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_rag.mcp.schemas import (
    EvidenceItem,
    RetrieveEvidenceAbstained,
    RetrieveEvidenceInput,
    RetrieveEvidenceSuccess,
    WikiItem,
)


def test_retrieve_input_defaults_and_trims_query() -> None:
    args = RetrieveEvidenceInput(query="  What is RAG?  ")

    assert args.query == "What is RAG?"
    assert args.paper_ids is None
    assert args.max_evidence == 4
    assert args.include_wiki is True
    assert args.wiki_max_entries == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", " "),
        ("query", "x" * 2001),
        ("paper_ids", [f"paper:{index}" for index in range(21)]),
        ("max_evidence", 0),
        ("max_evidence", 9),
        ("wiki_max_entries", -1),
        ("wiki_max_entries", 6),
    ],
)
def test_retrieve_input_rejects_contract_violations(field: str, value: object) -> None:
    kwargs = {"query": "question", field: value}

    with pytest.raises(ValidationError):
        RetrieveEvidenceInput(**kwargs)


def test_success_response_is_minimal_and_excludes_none() -> None:
    response = RetrieveEvidenceSuccess(
        decision="confident",
        retrieval_id="r_123",
        evidence=[
            EvidenceItem(
                citation="[chunk:c1]",
                paper_id="paper:1",
                title="Paper One",
                section=None,
                page=None,
                modality="text",
                text="Evidence text.",
            )
        ],
        wiki=[WikiItem(name="RAG", definition="Retrieval augmented generation.")],
    )

    assert response.model_dump(exclude_none=True) == {
        "decision": "confident",
        "retrieval_id": "r_123",
        "evidence": [
            {
                "citation": "[chunk:c1]",
                "paper_id": "paper:1",
                "title": "Paper One",
                "modality": "text",
                "text": "Evidence text.",
            }
        ],
        "wiki": [{"name": "RAG", "definition": "Retrieval augmented generation."}],
    }


def test_abstained_response_omits_wiki() -> None:
    response = RetrieveEvidenceAbstained(retrieval_id="r_none")

    assert response.model_dump(exclude_none=True) == {
        "decision": "no_evidence",
        "retrieval_id": "r_none",
        "evidence": [],
    }
