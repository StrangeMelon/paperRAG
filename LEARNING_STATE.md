# 学习状态

## 当前定位

- 上下文恢复点：2026-07-28（P3 存储初始化 checkpoint 已建立）
- 当前阶段：P4（采集、解析和切块）
- 当前课次：P4.1（解析子系统包入口）
- 上一个确认完成文件：`tests/test_init_store_real.py`
- 当前待处理事项：创建 `src/paper_rag/parse/__init__.py`，建立解析模块边界
- 包入口验证后的下一个文件：`tests/test_parse_fallback.py`
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

- 用户负责创建/修改所有项目文件，并亲自运行安装、测试、服务和 Git 命令。
- 除非用户明确要求助手执行某项操作，否则助手只提供代码、命令、预期结果和讲解。
- 助手可在用户明确要求“更新进度”时维护并单独提交本文件，但不得提交其他文件。
- 每次只讲解和实现一个项目文件；测试可以作为该文件的前置验收契约。
- Git message 使用 Conventional Commits：`<type>(<scope>): <中文摘要>`。

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

## 待处理问题

- 当前工作区中的 `src/paper_rag/store/sqlite_store.py` 与 `tests/test_sqlite_store.py`
  修改不属于本次课程状态提交，必须保留且不得混入后续 checkpoint。
- P4 从解析子系统包入口开始；随后先为 PyMuPDF 降级解析器编写边界测试，再实现解析器，
  最后用真实生成的 PDF Demo 和无 mock 集成测试验收。
- `AGENTS.md` 为用户未跟踪文件，助手不得擅自纳入课程提交。

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
- 课程状态提交不得包含业务文件。

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
