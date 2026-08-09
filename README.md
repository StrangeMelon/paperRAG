# Paper RAG Agent

面向学术论文的本地 RAG 与 Agentic RAG 后端项目。

项目支持从 arXiv、PDF URL 和本地 PDF 获取论文，经过解析、切块、嵌入和索引后，
使用 SQLite、Qdrant、混合检索和带引用约束的问答链路回答问题。

## 开发环境

本项目使用 Python 3.10+ 与 uv 管理依赖。

```bash
uv sync --extra dev

运行测试：

uv run pytest -q
```

## MCP Server

Install the MCP extra and run the stdio server:

```bash
uv run --extra mcp paper-rag-mcp
```

The default profile exposes `paper_retrieve_evidence`; use `--profile admin` for
the admin-only retrieval trace tool. LangChain MCP adapters can discover it with:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "paper-rag": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "--extra", "mcp", "paper-rag-mcp"],
    }
})
tools = await client.get_tools()
```
