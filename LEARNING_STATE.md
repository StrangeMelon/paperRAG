# 学习状态

## 当前定位

- 上下文恢复点：2026-07-29（arXiv 与 OpenAlex 采集 checkpoint 已建立）
- 当前阶段：P4（采集、解析和切块）
- 当前课次：P4.6（Semantic Scholar 采集）
- 上一个确认完成文件：`tests/test_openalex_source_real.py`
- 当前待处理事项：由助手创建 `tests/test_semantic_scholar_source.py`，先定义
  Semantic Scholar 采集边界
- 边界测试确认 RED 后的下一个文件：
  `src/paper_rag/ingest/semantic_scholar_source.py`
- 源文件基准：`/home/user_kyh/paper-rag-agent-main`
- 重建目录：`/home/user_kyh/paper-rag-agent-rebuild`

## 已确认的约束

- 使用与基准仓库相同的 Python 依赖、嵌入/重排模型、Qdrant、Docker 和
  OpenAI-compatible LLM 配置。
- 按可运行的依赖顺序重建，而非不可恢复的历史提交顺序。
- 核心 RAG 优先，随后扩展 Discovery、Wiki、反馈、主动 Agent、交付物和 DeerFlow。
- 代码逐文件讲解和复制；不讲 DeerFlow 前端及无关上游后端功能。
- 所有教学文本使用中文。
- 使用 `uv` 管理虚拟环境、依赖和 `uv.lock`，不使用 Conda。
- 以运行行为、公开接口、测试和评测为等价标准，不要求源文件字节级一致。
- 为兼容 `uv` 的通用解析，项目 Python 支持范围为 `>=3.10,<3.14`，并在
  `[tool.uv]` 中允许 MinerU 所需的预发布依赖。

## 协作规则

- 用户负责创建/修改 `src/paper_rag/` 下的正式功能文件，并执行所有 Git 命令。
- 助手直接创建/修改测试文件和 `scripts/demo_*.py`，无需逐次申请写入权限；助手不得
  代写正式功能文件。
- 用户亲自运行安装、测试、Demo 和服务命令，除非明确要求助手代为执行。
- 助手可在用户明确要求“更新进度”时维护并单独提交本文件，但不得提交其他文件。
- 每次只讲解和实现一个项目文件；测试可以作为该文件的前置验收契约。
- Git message 使用 Conventional Commits：`<type>(<scope>): <中文摘要>`。

## P4 固化顺序

P4 必须按完整数据流推进，不得因为解析器可以直接读取现成 PDF 而跳过真实采集：

1. **采集抽象契约**：`tests/test_ingest_sources.py` ->
   `src/paper_rag/ingest/sources.py`。
2. **本地 PDF 采集**：边界测试 -> `src/paper_rag/ingest/local_source.py` -> 真实 Demo ->
   无 mock 集成测试。验证文件复制、内容哈希 ID、标准目录、`meta.json`、`source.txt`
   与重复执行幂等性。
3. **PDF URL 采集**：边界测试 -> `src/paper_rag/ingest/url_source.py` -> 真实 HTTP Demo ->
   无 mock 集成测试。验证重定向、下载、HTTP 错误、PDF 落盘和临时文件清理。
4. **arXiv 采集**：边界测试 -> `src/paper_rag/ingest/arxiv_source.py` -> 真实 arXiv Demo ->
   无 mock 集成测试。验证 ID 归一化、元数据、版本信息、PDF 下载和重复执行。
5. **OpenAlex 采集**：边界测试 -> `src/paper_rag/ingest/openalex_source.py` -> 真实 API Demo
   -> 无 mock 集成测试。验证 DOI/Work 查询、倒排摘要还原和开放 PDF 行为。
6. **Semantic Scholar 采集**：边界测试 ->
   `src/paper_rag/ingest/semantic_scholar_source.py` -> 真实 API Demo -> 无 mock 集成测试。
   验证多种标识符、元数据和开放 PDF 行为。
7. **解析**：完成上述采集 checkpoint 后，依次进入 `src/paper_rag/parse/__init__.py`、
   PyMuPDF 降级解析、MinerU 本地解析和解析调度器。
8. **切块**：解析 checkpoint 完成后，依次实现 section splitter、text chunker、
   contextual chunk、builder、sanity 和 multimodal chunker。

阶段门禁：五类具体采集器的真实 Demo、无 mock 集成测试和 Ruff 检查没有全部通过前，
不得把当前课次推进到解析；解析未完成真实验收前，不得进入切块。

## 强制验收协议

对存在依赖边界或外部副作用的功能，必须按以下顺序学习、实现和验收：

1. **边界测试**：在生产代码前可使用 mock，明确输入、输出、异常分支与依赖调用
   契约。它只证明接口设计，不是功能完成证据。
2. **生产实现**：实现最小功能，使边界测试通过。
3. **真实 Demo**：在生产实现后新增可直接运行的 `scripts/demo_*.py`。它必须使用
   真实服务、真实数据和公共 API，逐步打印数据流与结果，并通过断言和非零退出码
   表示失败。Demo 必须使用隔离的临时数据或专用 collection，并负责清理，不能污染
   正式运行数据。
4. **真实集成测试**：新增无 mock 的 `tests/test_*_real.py`（也可放在
   `tests/integration/`），验证与 Demo 相同的关键不变量。使用
   `uv run pytest -vv -s <目标文件>` 单独运行，以显示每个用例和输出；服务、模型或
   密钥未准备好时必须明确失败，不能 skip 后宣称验收完成。
5. **checkpoint**：只有边界测试、真实 Demo、真实集成测试和适用的 Ruff 检查均通过，
   才能提交该功能的 Git checkpoint。

适用示例：SQLite 使用真实临时数据库文件；Qdrant 使用真实 Docker 或 embedded
实例和隔离 collection；LLM、嵌入、MinerU 等使用真实配置、真实模型或真实 API。
纯函数可仅使用单元测试，但应在调用它的真实链路 Demo 中得到覆盖。

## 已完成的验证

- 已确认基准工作区是无 Git 历史的源码快照。
- 已创建独立 Git 工作区和 `docs/learning/`、`docs/superpowers/plans/` 目录。
- `pyproject.toml` 已创建并通过 TOML 解析；`paper_rag` 包入口可导入并输出
  `0.1.0.dev0`。
- `README.md` 已创建；`uv sync --extra dev` 成功并生成 `uv.lock`。
- `src/paper_rag/utils/ids.py` 已实现；`tests/test_ids.py` 已通过（5 passed）。
- `src/paper_rag/utils/logger.py` 已实现并通过手工日志验证。
- `src/paper_rag/utils/hf_cache.py` 已实现；`tests/test_hf_cache.py` 已通过。
- `config/default.yaml`、`src/paper_rag/config.py` 和 `tests/test_config.py` 已实现；
  配置测试通过（3 passed），Ruff 检查通过。
- `src/paper_rag/utils/paths.py` 与 `tests/test_paths.py` 已实现；聚焦测试通过
  （3 passed），基础模块回归测试及 Ruff 检查通过。
- `src/paper_rag/ingest/__init__.py`、`schema.py` 与领域模型测试已完成；
  `tests/test_ingest_schema.py` 通过（4 passed）。
- `src/paper_rag/ingest/dedup.py` 与测试已完成；聚焦测试通过（2 passed）。
- P2 阶段全量 `uv run pytest -q` 通过，`uv run ruff check src tests` 通过。
- `src/paper_rag/store/__init__.py` 已创建并通过导入与 Ruff 验证。
- `src/paper_rag/store/sqlite_store.py` 已按三个 TDD 切片完成：论文与状态、
  ingest 步骤与跨来源查重、Section/Chunk 快照及旧库迁移。
- `tests/test_sqlite_store.py` 最终通过（9 passed）；全量 `uv run pytest -q`
  通过，SQLite 实现与测试的聚焦 Ruff 检查通过。
- `src/paper_rag/store/qdrant_store.py` 边界测试通过（12 passed）；真实 Docker
  Qdrant Demo 已验证 1024 维向量写入、Cosine 排序、metadata 过滤、按论文删除和
  collection 清理；无 mock 的 `tests/test_qdrant_store_real.py` 已通过。
- Qdrant 真实 Demo 暴露并修复了 Loguru 不解析 `%s` 的日志格式问题；修复后 Demo、
  边界测试与 Ruff 均重新通过。
- `scripts/init_store.py` 与 `tests/test_init_store.py` 已完成，边界测试通过（4 passed）；
  其中 SQLite 用真实临时数据库验证，Qdrant collection 与主流程顺序使用边界 fake。
- `scripts/init_store.py` 已连接真实 SQLite 与 Qdrant 连续运行两次，验证初始化流程的
  幂等行为；无 mock 的 `tests/test_init_store_real.py` 已纳入初始化 checkpoint。
- 存储初始化文件已提交为 `6d0b8e2 feat(store): 实现存储初始化入口与真实验收`，P3
  阶段结束，课程进入 P4。
- `src/paper_rag/ingest/sources.py` 已建立 `PaperSource.fetch()` 抽象契约；接口测试通过。
- `src/paper_rag/ingest/local_source.py` 已完成：使用真实本地 PDF 验证内容哈希 ID、
  `raw.pdf`、`meta.json`、`source.txt` 和重复采集幂等性；边界测试 4 项及无 mock
  集成测试均通过，并提交为 `71d1b95 feat(ingest): 实现本地PDF采集与真实验收`。
- `src/paper_rag/ingest/url_source.py` 已完成：边界测试通过 4 项；真实 Demo 与无 mock
  集成测试均直接采集 ACL Anthology 的
  `https://aclanthology.org/2025.acl-long.426.pdf`，验证公网 HTTPS 下载、PDF 有效性、
  内容哈希 ID 和标准落盘，并提交为
  `9644520 feat(ingest): 实现PDF URL采集与真实验收`。
- `src/paper_rag/ingest/arxiv_source.py` 已完成：边界测试、交互式真实 arXiv Demo 和
  无 mock 公网集成测试均通过；验证 arXiv ID 与版本归一化、稳定 `paper_id`、真实
  元数据、PDF 下载、标准落盘和重复采集复用，并提交为
  `698d611 feat(ingest): 实现 arXiv PDF 采集器与真实验收`。
- `src/paper_rag/ingest/openalex_source.py` 已完成：边界测试、交互式真实 OpenAlex Demo
  和无 mock 公网集成测试均通过。真实测试分别验证 metadata-only 与 metadata+PDF
  两条路径，并使用 PyMuPDF 打开 J-STAGE 的真实 PDF。
- OpenAlex 真实测试发现 `open_access.oa_url` 可能是 DOI 落地页而非 PDF；已通过新增
  RED 用例修正为依次读取 `best_oa_location.pdf_url`、
  `primary_location.pdf_url`、`locations[].pdf_url`，同时保留落地页作为元数据 URL，
  不再错误下载 DOI 页面。OpenAlex 已提交为
  `3f4de48 feat(ingest): 实现 OpenAlex 采集与真实验收`。

## 待处理问题

- 下一项是 Semantic Scholar 真实采集；完成其边界测试、生产实现、真实 Demo、无 mock
  集成测试和 Ruff 检查前不得进入解析。
- `AGENTS.md` 为用户未跟踪文件，助手不得擅自纳入课程提交。
- `demo-local-data/`、`demo-url-data/`、`demo-arxiv-data/` 和
  `demo-openalex-data/` 是用户选择保留的真实采集结果，不得纳入 Git。

## Git 状态说明

- `909c8f2 feat(core): 初始化项目骨架与基础工具`
- `0deae55 fix(utils): 修复模型缓存模块的导入顺序`
- `d4978f0 feat(config): 实现类型化配置加载`
- `d2958c9 docs(config): 补充数据目录配置说明`
- `826e5c0 feat(utils): 实现运行时路径管理`
- `d8ce9a5 feat(ingest): 定义论文采集的数据模型`
- `6d8d2ca feat(ingest): 实现论文采集去重判断`
- `2f510c3 chore(style): 规范基础工具与测试文件格式`
- `24bd07b feat(store): 实现SQLite 元数据与内容存储`
- `1262c23 chore(style): 修正注释中的易混淆标点`
- `7d86734 feat(store): 实现 Qdrant 向量存储与真实验收`
- `6d0b8e2 feat(store): 实现存储初始化入口与真实验收`
- `71d1b95 feat(ingest): 实现本地PDF采集与真实验收`
- `9644520 feat(ingest): 实现PDF URL采集与真实验收`
- `698d611 feat(ingest): 实现 arXiv PDF 采集器与真实验收`
- `3f4de48 feat(ingest): 实现 OpenAlex 采集与真实验收`
- 课程状态提交不得包含业务文件。

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
