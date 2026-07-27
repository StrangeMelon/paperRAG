# 学习状态

## 当前定位

- 上下文恢复点：2026-07-27（P2 阶段门禁通过）
- 当前阶段：P3（SQLite + Qdrant 双存储）
- 当前课次：P3.1（存储包入口）
- 上一个确认完成文件：`src/paper_rag/ingest/dedup.py`
- 当前待处理文件：`src/paper_rag/store/__init__.py`
- 存储包入口完成后的下一个文件：`tests/test_sqlite_store.py`
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

## 待处理问题

- `hf_cache.py`、`test_ids.py`、`test_paths.py` 存在已经过门禁验证的注释、格式和
  文件末尾换行调整，进入 P3 前由用户单独提交。
- `AGENTS.md` 为用户未跟踪文件，助手不得擅自纳入课程提交。
- 下一步创建 `src/paper_rag/store/__init__.py`，随后以测试驱动方式实现
  SQLite 元数据存储。

## Git 状态说明

- `909c8f2 feat(core): 初始化项目骨架与基础工具`
- `0deae55 fix(utils): 修复模型缓存模块的导入顺序`
- `d4978f0 feat(config): 实现类型化配置加载`
- `d2958c9 docs(config): 补充数据目录配置说明`
- `826e5c0 feat(utils): 实现运行时路径管理`
- `d8ce9a5 feat(ingest): 定义论文采集的数据模型`
- `6d8d2ca feat(ingest): 实现论文采集去重判断`

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
