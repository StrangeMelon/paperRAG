"""FastMCP stdio server and profile-specific tool registration."""

import argparse
import json
import os
from typing import Annotated, Any

from pydantic import Field

from .. import config as cfg
from ..rag.evidence_retrieval import Principal
from .errors import PaperRagToolError
from .resource_guards import configure_resource_guards
from .runtime import McpRuntime
from .trace_store import RetrievalTraceStore

DEFAULT_TOOL_DESCRIPTION = (
    "Retrieve a compact, citation-ready evidence set from indexed academic papers. "
    "The tool performs query understanding, rewriting, hybrid retrieval, reranking, "
    "sufficiency reflection, abstention, and Wiki background enrichment. It does not "
    "generate the final answer. Wiki content is background context only and must not "
    "be cited; factual claims must cite tokens from evidence."
)


def available_tool_names(profile: str) -> list[str]:
    if profile not in {"default", "admin"}:
        raise ValueError(f"unknown MCP profile: {profile}")
    names = ["paper_retrieve_evidence"]
    if profile == "admin":
        names.append("paper_get_retrieval_trace")
    return names


def _principal(profile: str) -> Principal:
    return Principal(
        tenant_id=os.getenv("PAPER_RAG_TENANT_ID", "system"),
        user_id=os.getenv("PAPER_RAG_USER_ID", "system"),
        is_admin=profile == "admin",
    )


def _tool_error(exc: PaperRagToolError):
    from mcp.server.fastmcp.exceptions import ToolError

    return ToolError(json.dumps(exc.to_payload(), ensure_ascii=False))


def create_server(
    *,
    profile: str | None = None,
    runtime: McpRuntime | None = None,
    trace_store: RetrievalTraceStore | None = None,
    principal: Principal | None = None,
    retrieval_timeout_sec: float | None = None,
):
    """Create a FastMCP server with exactly the selected profile's tools."""
    from mcp.server.fastmcp import FastMCP

    selected_profile = profile or cfg.load().mcp.profile
    available_tool_names(selected_profile)
    mcp = FastMCP("paper-rag", instructions=DEFAULT_TOOL_DESCRIPTION)
    mcp_runtime = runtime or _runtime_from_config()
    mcp_trace_store = trace_store or _trace_store_from_config()
    caller = principal or _principal(selected_profile)
    configure_resource_guards(cfg.load().mcp.resources.model_dump())

    from .tools.retrieve_evidence import paper_retrieve_evidence

    @mcp.tool(name="paper_retrieve_evidence", description=DEFAULT_TOOL_DESCRIPTION)
    async def _paper_retrieve_evidence(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        paper_ids: Annotated[list[str] | None, Field(max_length=20)] = None,
        max_evidence: Annotated[int, Field(ge=1, le=8)] = 4,
        include_wiki: bool = True,
        wiki_max_entries: Annotated[int, Field(ge=0, le=5)] = 3,
    ) -> dict[str, Any]:
        try:
            return await paper_retrieve_evidence(
                {
                    "query": query,
                    "paper_ids": paper_ids,
                    "max_evidence": max_evidence,
                    "include_wiki": include_wiki,
                    "wiki_max_entries": wiki_max_entries,
                },
                runtime=mcp_runtime,
                trace_store=mcp_trace_store,
                principal=caller,
                timeout=(
                    retrieval_timeout_sec
                    if retrieval_timeout_sec is not None
                    else cfg.load().mcp.retrieval_timeout_sec
                ),
            )
        except PaperRagToolError as exc:
            raise _tool_error(exc) from exc

    if selected_profile == "admin":
        from .tools.retrieval_trace import paper_get_retrieval_trace

        @mcp.tool(
            name="paper_get_retrieval_trace",
            description="Admin-only diagnostic access to a short-lived retrieval trace.",
        )
        async def _paper_get_retrieval_trace(retrieval_id: str) -> dict[str, Any]:
            try:
                return await paper_get_retrieval_trace(
                    retrieval_id,
                    trace_store=mcp_trace_store,
                    principal=caller,
                )
            except PaperRagToolError as exc:
                raise _tool_error(exc) from exc

    return mcp


def _runtime_from_config() -> McpRuntime:
    mcp_cfg = cfg.load().mcp
    return McpRuntime(
        max_running=mcp_cfg.max_running_retrievals,
        max_queued=mcp_cfg.max_queued_retrievals,
        admission_timeout=mcp_cfg.admission_timeout_sec,
        thread_tokens=mcp_cfg.thread_tokens,
    )


def _trace_store_from_config() -> RetrievalTraceStore:
    mcp_cfg = cfg.load().mcp
    return RetrievalTraceStore(
        ttl_sec=mcp_cfg.trace_ttl_sec,
        max_entries=mcp_cfg.trace_max_entries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paper-rag MCP stdio server")
    parser.add_argument("--profile", choices=("default", "admin"))
    parser.add_argument("--retrieval-timeout-sec", type=float)
    args = parser.parse_args(argv)
    create_server(
        profile=args.profile,
        retrieval_timeout_sec=args.retrieval_timeout_sec,
    ).run(transport="stdio")
    return 0


__all__ = ["DEFAULT_TOOL_DESCRIPTION", "available_tool_names", "create_server", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
