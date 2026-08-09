"""Domain failures exposed by the MCP adapter as tool errors."""

from __future__ import annotations

from typing import Any


class PaperRagToolError(Exception):
    code = "retrieval_unavailable"
    default_message = "Paper retrieval is unavailable."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidPaperScopeError(PaperRagToolError):
    code = "invalid_paper_scope"
    default_message = "One or more paper IDs do not exist, are inaccessible, or are not indexed."

    def __init__(self, paper_ids: list[str]):
        self.paper_ids = list(paper_ids)
        super().__init__(details={"paper_ids": self.paper_ids})


class RetrievalBusyError(PaperRagToolError):
    code = "busy"
    default_message = "Paper retrieval service is at capacity."

    def __init__(self, *, retry_after: int):
        self.retry_after = retry_after
        super().__init__(details={"retry_after": retry_after})


class PermissionDeniedError(PaperRagToolError):
    code = "permission_denied"
    default_message = "You do not have permission to access this retrieval trace."


class RetrievalExpiredError(PaperRagToolError):
    code = "retrieval_expired"
    default_message = "The retrieval trace does not exist or has expired."

    def __init__(self, retrieval_id: str):
        self.retrieval_id = retrieval_id
        super().__init__(details={"retrieval_id": retrieval_id})


class RetrievalTimeoutError(PaperRagToolError):
    code = "timeout"
    default_message = "Paper retrieval exceeded its time budget."


class EmbeddingUnavailableError(PaperRagToolError):
    code = "embedding_unavailable"
    default_message = "The query embedding service is unavailable."


class StoreUnavailableError(PaperRagToolError):
    code = "store_unavailable"
    default_message = "A required retrieval store is unavailable."


class RetrievalUnavailableError(PaperRagToolError):
    code = "retrieval_unavailable"
    default_message = "Paper retrieval is temporarily unavailable."
