"""MCP domain error contracts."""

import pytest

from paper_rag.mcp.errors import (
    InvalidPaperScopeError,
    PermissionDeniedError,
    RetrievalBusyError,
    RetrievalExpiredError,
    RetrievalTimeoutError,
)


def test_invalid_scope_error_has_stable_public_payload() -> None:
    error = InvalidPaperScopeError(["paper:missing"])

    assert error.to_payload() == {
        "code": "invalid_paper_scope",
        "message": "One or more paper IDs do not exist, are inaccessible, or are not indexed.",
        "details": {"paper_ids": ["paper:missing"]},
    }


def test_busy_error_exposes_retry_after() -> None:
    error = RetrievalBusyError(retry_after=5)

    assert error.to_payload() == {
        "code": "busy",
        "message": "Paper retrieval service is at capacity.",
        "details": {"retry_after": 5},
    }


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (PermissionDeniedError(), "permission_denied"),
        (RetrievalExpiredError("r_expired"), "retrieval_expired"),
        (RetrievalTimeoutError(), "timeout"),
    ],
)
def test_other_tool_errors_have_stable_codes(error, code: str) -> None:
    assert error.to_payload()["code"] == code
