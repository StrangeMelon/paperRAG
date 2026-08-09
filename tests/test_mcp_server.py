"""MCP server profile and tool catalog contracts."""

from paper_rag.mcp.server import DEFAULT_TOOL_DESCRIPTION, available_tool_names, create_server


def test_default_profile_exposes_only_public_retrieval_tool() -> None:
    assert available_tool_names("default") == ["paper_retrieve_evidence"]


def test_admin_profile_adds_trace_tool_only() -> None:
    assert available_tool_names("admin") == [
        "paper_retrieve_evidence",
        "paper_get_retrieval_trace",
    ]


def test_tool_description_sets_evidence_boundary() -> None:
    assert "does not generate the final answer" in DEFAULT_TOOL_DESCRIPTION
    assert "background" in DEFAULT_TOOL_DESCRIPTION.lower()
    assert "must not be cited" in DEFAULT_TOOL_DESCRIPTION


def test_fastmcp_schema_exposes_only_public_constrained_arguments() -> None:
    server = create_server(profile="default")
    tools = server._tool_manager.list_tools()

    assert [tool.name for tool in tools] == ["paper_retrieve_evidence"]
    properties = tools[0].parameters["properties"]
    assert set(properties) == {
        "query",
        "paper_ids",
        "max_evidence",
        "include_wiki",
        "wiki_max_entries",
    }
    assert properties["query"]["maxLength"] == 2000
    assert properties["paper_ids"]["anyOf"][0]["maxItems"] == 20
    assert properties["max_evidence"]["minimum"] == 1
    assert properties["max_evidence"]["maximum"] == 8


def test_fastmcp_admin_profile_physically_registers_trace_tool() -> None:
    server = create_server(profile="admin")

    assert [tool.name for tool in server._tool_manager.list_tools()] == [
        "paper_retrieve_evidence",
        "paper_get_retrieval_trace",
    ]
