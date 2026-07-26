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