# 源文件与验证清单

## 使用方法

这是课程的外部记忆索引。每次讲解前，按阶段查找唯一的“下一个文件”；完成后在
`LEARNING_STATE.md` 记录源文件与重建文件的 SHA-256。源码一律以
`/home/user_kyh/paper-rag-agent-main` 为基准，运行时不得依赖该目录。

## 阶段文件映射

| 阶段 | 基准文件或目录 | 主要验证 |
| --- | --- | --- |
| P1 | `pyproject.toml`、`.env.example`、`Makefile`、`config/`、Docker/Compose | `make smoke`、配置加载 |
| P2 | `src/paper_rag/__init__.py`、`config.py`、`utils/`、`ingest/schema.py`、`ingest/dedup.py` | `tests/test_pure.py` |
| P3 | `store/sqlite_store.py`、`store/qdrant_store.py`、`scripts/init_store.py` | 存储初始化、`tests/test_skeleton.py` |
| P4 | `ingest/`、`parse/`、`chunk/` | `tests/test_m5_fixes.py`、切块检查 |
| P5 | `embed/`、`store/ingest_pipeline.py`、ingest 脚本 | ingest 状态机与索引重建 |
| P6 | `retrieve/` | `tests/test_retrieve_pure.py`、`tests/test_m5_p1.py` |
| P7 | `rag/` | abstain、citation、memory、stream 测试 |
| P8 | `tools/`、`observability/`、`validate/`、`tests/eval/` | eval、chaos、性能与工具测试 |
| P9 | `discovery/`、`wiki/` | discovery/wiki/closed-loop 测试 |
| P10 | `vision/`、`deliver/`、`feedback/`、`proactive/` | vision/deliver/feedback/proactive 测试 |
| P11 | DeerFlow Paper RAG 相关后端文件与测试 | Gateway/Harness integration 测试 |
| P12 | 部署、运行、维护脚本与运维文档 | smoke、secret scan、完整验收 |

## Paper RAG 包目录索引

- `chunk/`：`section_splitter.py`、`text_chunker.py`、`contextual.py`、`builder.py`、`sanity.py`、`multimodal_chunker.py`。
- `deliver/`：`_common.py`、`survey_md.py`、`pptx.py`、`docx.py`、`latex_bib.py`、`pdf.py`、`dispatch.py`。
- `discovery/`：`sources.py`、`ranker.py`、`store.py`、`trace.py`、`runner.py`。
- `embed/`：`bge_m3.py`。
- `feedback/`：`events.py`、`store.py`、`collector.py`。
- `ingest/`：`schema.py`、`dedup.py`、`arxiv_source.py`、`local_source.py`、`url_source.py`、`openalex_source.py`、`semantic_scholar_source.py`、`sources.py`。
- `observability/`：`metrics.py`、`trace.py`。
- `parse/`：`dispatcher.py`、`fallback_pymupdf.py`、`mineru_local.py`。
- `proactive/`：`_db.py`、`subscriptions.py`、`inbox.py`、`paper_access.py`、`matcher.py`、`digest.py`、`stale.py`、`auto_ingest_hook.py`、`webhook.py`、`cron_runner.py`。
- `rag/`：`llm.py`、`qa_simple.py`、`intent_classifier.py`、`query_rewrite.py`、`reflect.py`、`evidence_select.py`、`citation_check.py`、`abstain.py`、`qa_agentic.py`、`qa_cache.py`、`history.py`、`research_memory.py`、`async_api.py`、`qa_stream.py`。
- `retrieve/`：`dense.py`、`sparse_bm25.py`、`fts5.py`、`hybrid.py`、`rerank.py`、`pipeline.py`、`format.py`。
- `store/`：`sqlite_store.py`、`qdrant_store.py`、`ingest_pipeline.py`。
- `tools/`：`_schema.py`、`paper_index.py`、`paper_search.py`、`paper_section.py`、`paper_qa.py`、`paper_compare.py`、`paper_discover.py`、`wiki_lookup.py`、`bibtex_export.py`。
- `utils/`：`ids.py`、`paths.py`、`logger.py`、`hf_cache.py`。
- `validate/`：`metadata_paths.py`。
- `vision/`：`schema.py`、`cache.py`、`api.py`、`local.py`、`enrich.py`。
- `wiki/`：`schema.py`、`normalize.py`、`store.py`、`concept_extractor.py`、`consistency.py`、`flow.py`、`queue.py`、`review_queue.py`、`triggers.py`、`context.py`、`usage.py`。

每个包的 `__init__.py` 也必须重建；它们会紧跟在该包第一个实现文件之前处理。

## DeerFlow 仅限项目相关的文件

- `packages/harness/deerflow/community/paper_rag/__init__.py`
- `packages/harness/deerflow/community/paper_rag/tools.py`
- `packages/harness/deerflow/subagents/builtins/paper_research.py`
- `app/gateway/routers/paper_rag.py`
- `app/gateway/routers/feedback.py`
- `app/gateway/routers/metrics.py`
- 与这些文件直接相关的 Router 注册、Gateway app、认证/中间件配置及
  `test_paper_rag_harness_adapter.py`、`test_paper_rag_integration.py` 等测试。
