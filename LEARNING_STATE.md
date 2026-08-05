# 学习状态

> 本文件只保留新会话恢复课程所需的当前状态、关键约束和下一步。逐课实现细节、
> 诊断过程、测试输出与提交注解已归档到 `docs/learning/LEARNING_HISTORY.md`；恢复
> 课程时默认不读取归档，只有追溯历史决策时才查阅。

## 当前定位

- 工作目录：`/home/user_kyh/paper-rag-agent-rebuild`。
- 只读基准：`/home/user_kyh/paper-rag-agent-main`，不得写入新代码。
- **P1-P7 核心后端已全部完成并收口**：采集、解析、切块、嵌入与入库、检索、
  RAG/QA 均已建成；`ask`、单篇/批量入库 CLI 和 Vision 视觉增强也已完成。
- 当前处于**正式语料批量入库与后续能力选题阶段**。用户已用 `--limit 3` 在正式库
  成功试入三篇中文 PDF；唯一既定操作是移除 `--limit` 后继续全量批跑。批跑支持
  断点续传，可分多晚完成。
- 全量入库完成后的候选课程：**元数据补全**、**评测层**、**入库质量清单**。
  用户提出明确的新需求时，以新需求决定下一课，不擅自插入遗留问题。

## 已完成能力

- **采集与解析**：本地 PDF、URL、arXiv、OpenAlex、Semantic Scholar 边界；
  PyMuPDF 兜底与 MinerU CUDA OCR；中英文语言识别、降级与失败记账。
- **切块**：章节、文本、上下文、页码、多模态和质量检查已贯通。文本块保持原文
  偏移可回切；MinerU layout 为图表补页码/图注；参考文献块保留并打标。
- **嵌入与入库**：BGE-M3、SQLite 溯源、Qdrant 向量、三级幂等去重、force 重建、
  FTS5 增量同步、状态记录与单篇失败隔离。
- **检索**：dense + FTS5/BM25 + RRF 混合检索 + cross-encoder 重排 + 论文多样化；
  `retrieve_round()` 已接查询改写，`format_evidence()` 输出 `[chunk:<id>]` 证据令牌。
- **RAG/QA**：LLM、双语查询改写、意图分类、证据选择、弃答、反思、引用校验、
  `qa_simple`、`qa_agentic`、`qa_stream` 全部完成。Agentic 链路含 trace 与进程内指标；
  流式链路输出结构化事件并支持域外问题零 LLM 短路。
- **CLI**：`scripts/ask.py` 支持默认、`--agentic`、`--stream` 与 `--no-llm`；
  `scripts/ingest_one.py` 和 `scripts/ingest_batch.py` 支持 dry-run、limit、force、
  逐篇异常隔离、JSON 报告和幂等续跑。CLI 会自行加载 `.env` 且不覆盖已导出变量。
- **Vision**：图表视觉摘要已接入 builder/ingest，支持中英文提示词、语言相关缓存键、
  失败不入索引与缓存幂等；生产模型为 GLM-4.6V，`vision.enabled=true`。

## 当前运行检查点

- 2026-08-05 记录：正式 Qdrant 1.18.3 服务运行中，`paper_chunks`、
  `wiki_entries` collection 和 SQLite 四表已初始化。
- 正式库 `--limit 3` 结果：三篇全部 `done`，共约 99.8 秒；两篇论文分别 79/73
  chunks，一篇征文通知 7 chunks 且被如实标为 `mineru+broken`。这说明正式 PDF
  目录可能混有非论文文件，全量结束后需按报告中的 `parsed_with` 清查。
- 批量入口会写逐篇状态、块数、耗时和错误；默认报告位于
  `<data_root>/ingest_batch_report.json`。新库必须先运行 `scripts/init_store.py`。
- Embedded Qdrant 会持有目录锁；父进程准备数据后若需启动 CLI 子进程，必须先调用
  `qdrant_store.close_client()`，否则子进程检索可能静默为空。
- 元数据补全已约定为独立课程，递进方案是：首页抽 arXiv/DOI 查权威源 -> 标题模糊
  搜索并做相似度校验 -> 中文首页结构化解析与 LLM 兜底。Semantic Scholar 标题
  搜索前还需增加 `/paper/search` 方法。

## 关键提交索引

- P5：BGE-M3 `a91a30b`；入库管线 `5201259`。
- P6：dense `45c86c0`；FTS5 `aeec63d`；BM25 `e5ce271`；hybrid `3ecc6c8`；
  rerank `4140338`；format/pipeline `11d80d9`。
- P7：LLM `92662d2`；query rewrite `46be7c3`；intent `762d353`；
  evidence select `83b9d30`；abstain `395cf7f`；reflect `2a32494`；
  citation check `170c12d`；qa_simple `d52f466`；qa_agentic/observability
  `cfe67fd` + `9175749`；qa_stream `e976dc2`；ADR-0002 `c8d109a`。
- CLI：ask `cb4f414`；单篇/批量入库 `dd40519`。
- Vision：功能 `1f37a10`；课程记录 `434cfb5`；纯逻辑测试隔离 `70de121`。
- 更早阶段的逐文件提交和验收证据见 `docs/learning/LEARNING_HISTORY.md`。

## 必须保留的设计约束

- 目标规模是 20,000 篇、约 1,000,000 chunks。生产 Qdrant 走远程服务模式，
  embedded 只用于 Demo；`rank_bm25` 仅作小规模后备和评测对照，并受
  `retrieve.bm25_max_chunks` 保护。
- 使用 `uv`，Python `>=3.10,<3.14`。常用完整环境同步命令：
  `uv sync --extra dev --extra embed --extra ingest --extra mineru`；四个 extra
  都要保留，避免卸掉开发或服务依赖。
- MinerU 生产默认 `.venv/bin/magic-pdf`、CUDA GPU、强制 OCR；模型保存在项目
  `data/index/mineru_models/`。MinerU 失败可降级 PyMuPDF；扫描件无正文必须记失败，
  批处理继续，空结果不能伪装成功。
- 语料中英混合。领域语言值统一为 `zh | en | None`，供应商值 `ch | en` 只存在于
  MinerU 边界。语言判断失败不终止流程；后续新模块必须检查基准中的英文隐式假设，
  明确设计中文扩展。
- LLM 通过 OpenAI-compatible API；`.env` 提供 `OPENAI_BASE_URL`、
  `OPENAI_API_KEY`、`CHAT_MODEL`。Qwen 思考型模型的非流式配置需
  `llm.extra_body: {enable_thinking: false}`。
- 主引擎保持同步；异步只在未来网关边界用 `anyio.to_thread.run_sync` 包装。
  公共 `llm.py` 保持非流式，流式实现只位于 `qa_stream.py`。
- Citation 的硬协议是 `[chunk:<id>]`。引用必须先校验合法 ID，再检测并清理可疑的
  数字/作者年份形态；中文全角引用同样受检。
- 弃答阈值 0.21/0.48 尚未用中英混合评测集重校。低质量 BM25/RRF 信号或字段缺失
  时 fail open，并通过 `signal_quality` 暴露降级状态。
- CJK 词面匹配采用 bigram，与 ADR-0001/0002 一致；单字查询没有 unigram 命中，
  由稠密检索兜底。FTS5 原地 UPDATE 行数自愈和多样化补位超单篇限额仍是已知边界。

## 待处理问题

- `src/paper_rag/ingest/arxiv_source.py` 保留一份未提交的 Task 6 元数据持久化迁移
  diff，必须独立处理，不能混入其他提交。arXiv 真实请求缺显式 timeout，未来另开
  `fix(ingest): 为 arXiv 真实请求增加超时`。
- Semantic Scholar 因缺 API key 尚未完成真实验收；
  `scripts/demo_semantic_scholar_source.py` 为未跟踪文件，真实集成测试尚未创建。
- `AGENTS.md`、`CLAUDE.md`、`scripts/demo_semantic_scholar_source.py` 当前按未跟踪
  文件保留，助手不得擅自提交。恢复时始终以实际 `git status` 为准。
- Vision 本地 Qwen2.5-VL 兜底尚未做真实 GPU 验收，默认 `fallback_local=false`。
  “视觉摘要 vs 仅图注”的检索收益对比归入评测层。
- 全量 Ruff 的既有历史问题位于 `scripts/demo_qdrant_store.py`、
  `src/paper_rag/ingest/arxiv_source.py`、`src/paper_rag/store/qdrant_store.py`；
  后续课只检查改动路径，不顺带清理。
- `data/`、`demo-*-data/`、PDF、模型、数据库与 `.env` 都是运行产物或秘密，不提交、
  不删除。

## 协作规则

- 助手直接创建/修改所有代码文件，包括正式功能、测试和 `scripts/`，并自行完成
  开发阶段 RED/GREEN、聚焦测试和 Ruff 自查；用户不手抄代码。
- Git 分工：仅 `LEARNING_STATE.md` 与 `docs/learning/LEARNING_HISTORY.md` 由助手用
  独立 `docs(course)` 提交；所有功能、测试、配置和 Demo 文件由用户提交。助手收尾
  时提供精确 `git add` 清单和 Conventional Commit message。
- 每次只讲解和实现一个项目文件，测试可先作为该文件的验收契约。新模块开始前必须
  先讲“为什么需要、承担什么职责”，再讲“怎么实现、基准有哪些英文假设、中文如何
  扩展”；方案经用户确认后才写代码。
- 代码完成后，最终测试和真实 Demo 由用户亲自运行。助手先给出命令、预期输出和关键
  不变量，再根据用户实跑结果对比收口。
- 新包必须同时纳入 `__init__.py`。提交前用 `git archive HEAD` 解包到 `/tmp` 后运行
  聚焦测试，验证克隆态不依赖未提交文件。
- Commit message 使用 `<type>(<scope>): <中文摘要>`。

## 强制验收协议

存在依赖边界或外部副作用时，顺序固定为：

1. 边界测试：可 mock，只证明接口和失败路径。
2. 生产实现。
3. 真实 Demo：`scripts/demo_*.py`，使用真实服务/数据/API，隔离数据，包含断言，
   失败时非零退出。
4. 真实集成测试：`tests/test_*_real.py`，无 mock，使用
   `uv run pytest -vv -s` 单独运行；缺服务或密钥必须明确失败，不能 skip 后宣称通过。
5. Checkpoint：聚焦测试、排除真实用例的全量测试、真实 Demo/集成测试和 Ruff 均通过。

纯函数可以只做单元测试，但应在后续真实链路 Demo 中覆盖。SQLite 使用真实临时库，
Qdrant 使用真实服务或隔离 embedded collection，LLM/嵌入/MinerU 使用真实配置和模型。
全量测试必须通过 `uv run python -m pytest` 启动，避免 `scripts.init_store` 导入失败。
真实验收命令交付前，还要在剥离相关已导出变量的干净 shell 中复跑，确认 `.env` 加载
不是被父进程环境偶然掩盖。

## 新会话恢复协议

1. 进入重建目录，完整读取本文件和 `AGENTS.md`；无需默认读取历史归档。
2. 只读执行 `git status --short` 和 `git log -5 --oneline`，以文件系统实际状态为准。
3. 不得 reset、checkout、clean、覆盖或删除任何现有改动与运行数据。
4. 基准仓库只读；待处理问题不在课次中间插队，除非用户明确要求。
5. 从“当前定位”的下一步或用户最新明确需求继续。

## 每次课结束必须更新

- 当前阶段、完成文件、公开接口和关键行为。
- 助手开发自查与用户真实验收的命令、结果和失败原因。
- 新设计决策、已知边界、待复习概念和下一目标。
- 详细过程写入历史归档；本文件只更新恢复所需的结论。
