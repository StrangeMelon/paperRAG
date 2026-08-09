"""Public MCP input and output schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrieveEvidenceInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] | None = Field(default=None, max_length=20)
    max_evidence: int = Field(default=4, ge=1, le=8)
    include_wiki: bool = True
    wiki_max_entries: int = Field(default=3, ge=0, le=5)

    @field_validator("query")
    @classmethod
    def _strip_non_empty_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class EvidenceItem(BaseModel):
    citation: str
    paper_id: str
    title: str
    section: str | None = None
    page: int | None = None
    modality: str
    text: str


class WikiItem(BaseModel):
    name: str
    definition: str


class RetrieveEvidenceSuccess(BaseModel):
    decision: Literal["confident", "weak_evidence"]
    retrieval_id: str
    evidence: list[EvidenceItem]
    wiki: list[WikiItem]


class RetrieveEvidenceAbstained(BaseModel):
    decision: Literal["no_evidence"] = "no_evidence"
    retrieval_id: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
