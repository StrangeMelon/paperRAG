# 学习状态

## 当前定位

- 上下文恢复点：2026-07-31（双语 OCR Task 10–14 已提交；真实模型已下载、`mineru_doctor.py
  --strict` 严格通过；中英文真实 GPU OCR Demo 与无 mock 集成测试已由用户在真实 GPU 上
  运行通过并提交）
- 当前阶段：P4（采集、解析和切块）
- 当前课次：P4.9（MinerU 本地 GPU OCR 解析器与中英文语言路由）
- 上一个确认完成文件：`tests/test_mineru_bilingual_real.py`（Task 14，提交 `2e147af`）
- 当前待处理事项：唯一下一步是实施计划 Task 15——解析调度器
  `src/paper_rag/parse/dispatcher.py`。助手先写 `tests/test_parse_dispatcher.py` 边界
  测试并观察 RED；用户实现 `dispatcher.py`（MinerU→PyMuPDF 降级、空结果拒绝、单篇失败
  隔离，保留 `(Path, parser_name)` 接口）；助手再写 `scripts/demo_parse_dispatcher.py`
  真实降级 Demo。Task 15 完成后方可进入切块模块。
- 分工更新（2026-07-31）：`scripts/` 下所有文件均由助手直接写入并自测，不再限于
  `scripts/demo_*.py`；`src/paper_rag/` 正式功能文件仍由用户编写。
- Semantic Scholar 恢复点：取得 API key 后运行
  `scripts/demo_semantic_scholar_source.py`，再创建并运行无 mock 真实集成测试
- 源文件基准：`/home/user_kyh/paper-rag-agent-main`
- 重建目录：`/home/user_kyh/paper-rag-agent-rebuild`

## 新会话恢复协议

新会话不要重新进行需求访谈，也不要从基准仓库重新开始。用户可以把下面这句话作为新会话
的第一条消息：

```text
请继续 /home/user_kyh/paper-rag-agent-rebuild 的后端重建课程。先完整读取
LEARNING_STATE.md、双语 OCR 设计文档和实施计划，再检查 Git 状态；不要回退任何现有改动，
从状态文件记录的唯一下一步继续。
```

新会话中的助手必须按以下顺序恢复：

1. 把 `/home/user_kyh/paper-rag-agent-rebuild` 作为实际工作目录；
   `/home/user_kyh/paper-rag-agent-main` 只作为源码基准，不能把新代码写入基准仓库。
2. 完整读取本文件；若重建目录中存在 `AGENTS.md`，同时读取并遵守它。
3. 完整读取
   `docs/superpowers/specs/2026-07-30-bilingual-mineru-language-routing-design.md` 和
   `docs/superpowers/plans/2026-07-30-bilingual-mineru-language-routing.md`。
4. 执行只读的 `git status --short` 和 `git log -5 --oneline`，以实际文件系统为准；不得
   reset、checkout、清理或覆盖任何未提交文件和 Demo 数据。
5. 如果实施计划仍是未跟踪文件，先告诉用户执行：
   `git add docs/superpowers/plans/2026-07-30-bilingual-mineru-language-routing.md`，再执行
   `git commit -m "docs(parse): 规划中英文 OCR 语言路由实施步骤"`。如果计划已经提交，
   直接跳过这一步。
6. 随后从实施计划 Task 15 开始。助手先写 `tests/test_parse_dispatcher.py` 边界测试
   （MinerU 成功→`mineru`；MinerU 抛错且 PyMuPDF 有正文→`pymupdf` 且状态 `degraded`；
   PyMuPDF 只有页标记→`ParseError` 且状态 `failed`；禁用 fallback 时重抛 MinerU 错误），
   用户运行观察 RED。
7. RED 确认后，由用户亲自实现 `src/paper_rag/parse/dispatcher.py`；助手写测试和
   `scripts/demo_parse_dispatcher.py`，不代写 `src/paper_rag/` 正式功能文件。
8. Task 1–14 已完成：真实模型已下载到 `data/index/mineru_models/`，`mineru_doctor.py
   --strict` 退出码 0，中英文真实 GPU OCR 已通过。Task 15 解析调度器完成后方可进入
   切块模块。

恢复时继续遵守已有分工：助手写测试和 `scripts/demo_*.py`，用户写正式功能代码并执行
安装、测试、Demo、服务和所有业务 Git 命令。只有用户明确要求“更新进度”时，助手可以
单独修改并提交 `LEARNING_STATE.md`。

## 已确认的约束

- 使用与基准仓库相同的 Python 依赖、嵌入/重排模型、Qdrant、Docker 和
  OpenAI-compatible LLM 配置。
- 按可运行的依赖顺序重建，而非不可恢复的历史提交顺序。
- 核心 RAG 优先，随后扩展 Discovery、Wiki、反馈、主动 Agent、交付物和 DeerFlow。
- 代码逐文件讲解和复制；不讲 DeerFlow 前端及无关上游后端功能。
- 所有教学文本使用中文。
- 使用 `uv` 管理虚拟环境、依赖和 `uv.lock`，不使用 Conda。
- 以运行行为、公开接口、测试和评测为等价标准，不要求源文件字节级一致。
- MinerU 生产默认使用 `.venv/bin/magic-pdf`、强制 `ocr` 模式和 CUDA GPU；真实模型
  必须下载到重写项目自己的 `data/index/mineru_models/`，不得引用原项目模型目录。
- 论文集合同时包含中文和英文论文；纯扫描 PDF 由人工在同目录 `meta.json` 顶层标注
  `language: "zh" | "en"`，普通 PDF 不要求人工标注。
- 应用配置使用 `mineru.lang: auto | ch | en`；`auto` 模式优先读取人工元数据，其次使用
  PyMuPDF 采样文字判断，无法判断时回退 PaddleOCR 中英文通用模型 `ch`。
- 领域元数据使用 `zh/en`，只在 MinerU 适配边界映射为供应商参数 `ch/en`；自动模式必须
  准备中英文两套 OCR 权重。
- 语言判断失败不能终止流程；英文权重缺失但中文权重完整时降级到 `ch`。MinerU 失败时
  普通文字 PDF 可降级 PyMuPDF；扫描件仍无正文时记录单篇失败，批处理继续下一篇，并且
  空结果不得伪装成成功。
- 为兼容 `uv` 的通用解析，项目 Python 支持范围为 `>=3.10,<3.14`，并在
  `[tool.uv]` 中允许 MinerU 所需的预发布依赖。

## 协作规则

- 用户负责创建/修改 `src/paper_rag/` 下的正式功能文件，并执行所有 Git 命令。
- 助手直接创建/修改测试文件和 `scripts/` 下所有脚本，无需逐次申请写入权限；助手不得
  代写 `src/paper_rag/` 正式功能文件。
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

临时门禁例外：2026-07-29 用户暂时没有 Semantic Scholar API key，并明确要求先跳过。
因此课程可以继续解析，但 Semantic Scholar 不能标记为完整 checkpoint；在最终后端验收前
必须回补真实 Demo、无 mock 集成测试、Ruff 检查和独立真实验收提交。

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
- `src/paper_rag/ingest/semantic_scholar_source.py` 已完成边界实现：支持 arXiv URL、
  `arxiv:`、`doi:`、裸 DOI 和 S2 Paper ID；根据 API 返回的 ArXiv、DOI、S2 ID
  优先级生成稳定 `paper_id`，映射标准元数据，并支持开放 PDF 下载与已有 PDF 复用。
- `tests/test_semantic_scholar_source.py` 边界测试通过（4 passed），覆盖 API key 请求头、
  标识符归一化、metadata-only、S2 ID 降级和 PDF 幂等复用；已提交为
  `e45e6ac feat(ingest): 实现 Semantic Scholar 采集边界`。
- `src/paper_rag/parse/__init__.py` 已创建并通过包入口测试，保持解析后端的延迟导入
  边界。
- `src/paper_rag/parse/fallback_pymupdf.py` 已实现：使用真实 PyMuPDF 创建和读取 PDF，
  按页生成 `paper.md` 与页标记，清理 NUL 字符并保证重复解析稳定；边界测试和允许用户
  选择本地 PDF 的 `scripts/demo_fallback_pymupdf.py` 已通过真实验收。
- `src/paper_rag/parse/mineru_local.py` 已完成四个内部切片：运行时缓存环境、CLI 定位、
  错误分类与诊断数据结构；MinerU 原始 Markdown/图片/layout 发现和标准化；真实
  `subprocess.run()` 解析调度、超时、非零退出和空产物拒绝；以及 Doctor 的依赖、模型
  目录、布局权重、按语言的 OCR 权重、CLI 版本和报告汇总。Task 10 目标回归中
  `tests/test_mineru_local.py` 已通过（25 passed）。
- `scripts/demo_mineru_local.py` 已创建，支持交互输入本地 PDF、默认强制 OCR、隔离或
  持久化输出，并检查标准化 Markdown、figures、layout 与原始产物；Ruff 和 `--help`
  启动检查通过，但尚未运行真实解析。
- MinerU 依赖已通过
  `uv sync --extra dev --extra ingest --extra mineru` 安装；实际 CLI 是
  `.venv/bin/magic-pdf`，安装版本为 `magic-pdf 1.3.12`。
- 用户已在宿主环境确认 PyTorch `2.13.0+cu130`、CUDA 13.0 可用，并识别到
  `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`。
- `config/default.yaml` 已将 MinerU CLI 改为 `magic-pdf`、解析模式改为 `ocr`；
  `config/magic-pdf.json` 已配置 `device-mode: cuda`、项目内模型目录、
  `doclayout_yolo`，并保持表格和公式识别关闭。`tests/test_mineru_gpu_config.py`
  已通过（2 passed）。
- MinerU Doctor 的第四段边界测试已追加到 `tests/test_mineru_local.py`，覆盖依赖导入、
  模型根目录、布局权重、按语言选择的 OCR 检测/识别权重、CLI 版本和报告汇总；用户
  已确认最终 GREEN 为 `23 passed, 13 warnings`。
- `scripts/mineru_doctor.py` 已按五个 TDD 切片完成：JSON 输出、人类可读报告、
  `--strict` 退出码、`--try-parse` 成功/失败结构，以及终端试解析结果。用户已确认
  `tests/test_mineru_doctor_script.py` 全部通过（7 passed），Ruff 与 `--help` 入口也已通过。
- 真实 Doctor 已运行：CLI、完整 MinerU 依赖、OpenCV、配置文件和 CLI 版本检查通过；
  因模型目录不存在及 OCR 语言尚未配置而准确返回 `ok: False`，`--strict` 退出码为 `1`。
- 用户补充语料为中英文混合集合，并确认纯扫描件有人工语言元数据、普通 PDF 由系统自动
  判断；无法判断时使用 `ch`，且采用单篇失败隔离和 PyMuPDF 降级策略。
- 双语设计文档
  `docs/superpowers/specs/2026-07-30-bilingual-mineru-language-routing-design.md` 已确认并提交为
  `e12284d docs(parse): 设计中英文 OCR 语言路由与降级策略`。
- 双语实施计划
  `docs/superpowers/plans/2026-07-30-bilingual-mineru-language-routing.md` 已生成、自审并提交为
  `ddff85b docs(parse): 规划中英文 OCR 语言路由实施步骤`。计划共 15 个逐文件任务，
  固定官方模型仓库修订 `a4f6a8d29a4d96730f90ea174a9322e842b93552`，明确七个布局、
  LayoutReader 和中英文 OCR 文件。
- Task 1 已提交为 `13dd3a6 feat(config): 支持 MinerU OCR 语言自动路由`：
  `mineru.lang` 配置值域为 `auto | ch | en`，默认值改为 `auto`。
- Task 2 已提交为 `8afa910 feat(ingest): 为论文元数据添加语言字段`：
  `PaperMeta.language` 支持领域语言 `zh/en/None` 并拒绝供应商值 `ch`。
- Task 3 已提交为 `381f748 feat(ingest): 保留论文人工语言标注`：
  新增 `src/paper_rag/ingest/metadata.py`，统一写入 `meta.json` 并保留已有人工语言。
- Task 4 已提交为 `3931b6a refactor(ingest): 统一本地论文元数据持久化`：
  本地 PDF 采集器改用统一元数据持久化函数，重复采集保留人工语言。
- Task 5 已提交为 `e193b7f refactor(ingest): 统一 URL 论文元数据持久化`：
  URL PDF 采集器改用统一元数据持久化函数。
- Task 7 已提交为 `d8b2673 refactor(ingest): 保留 OpenAlex 论文语言元数据`：
  OpenAlex 采集器映射可信 `zh/en` 语言并保留人工标注。
- Task 8 已提交为 `7a88b37 refactor(ingest): 统一 Semantic Scholar 元数据持久化`：
  Semantic Scholar 采集器改用统一元数据持久化函数；真实 API 验收仍等待 API key。
- Task 9 已提交为 `5abecae feat(parse): 实现中英文 OCR 语言自动判断`：
  `resolve_ocr_language()` 支持强制 `ch/en`、人工元数据优先、PyMuPDF 文本采样和失败
  回退到 `ch`，对应测试覆盖中英文、扫描件、损坏元数据与损坏 PDF。
- Task 10 已提交为 `8ce42b1 feat(parse): 诊断 Mineru 中英文OCR权重`：Doctor 的
  `auto` 模式展开检查 `ch/en` 四个 OCR 权重，新增 OCR 权重路径/可用性函数和
  LayoutReader 两个文件检查，并接入 `diagnose()`。本轮复核
  `tests/test_mineru_local.py` 为 25 passed，`tests/test_mineru_doctor_script.py` 为
  7 passed，Task 10 目标 Ruff 与 `git diff --check` 均通过；但只包含提交 `8ce42b1` 的
  `/tmp` 干净快照中 `tests/test_mineru_local.py` 为 24 passed、1 failed，聚合测试因
  `config/magic-pdf.json` 未被提交而缺少 `models-dir` 检查。生产实现未发现对应行为错误，
  测试必须改为自建临时配置后再确认 Task 10 checkpoint。
- Task 10 测试可复现性已修复并提交为 `7207705 feat(parse): 诊断 MinerU 中英文 OCR 权重`：
  Doctor 聚合测试改为用 `tmp_path` 自建临时 `config/magic-pdf.json` 并 monkeypatch
  `cfg.PROJECT_ROOT`，不再依赖未跟踪配置；干净快照复核通过。
- Task 11 已提交为 `768548d feat(parse): 按论文选择 MinerU OCR 语言`：新增
  `_select_available_ocr_language()`，CLI 不再接收 `auto`；英文权重缺失且中文可用时降级
  `ch` 并写出含 `model_fallback` 的 `language.json`。`test_mineru_local.py` 与
  `test_parse_language.py` 共 35 passed，目标 Ruff 通过。
- Task 12 已提交为 `ba4d0fb feat(parse): 增加 MinerU 双语模型下载入口`：
  `scripts/download_mineru_models.py` 固定官方修订 `a4f6a8d…`，含 7 个权重的远端/本地映射、
  大小校验、`.part` 原子替换与复用；`tests/test_download_mineru_models.py` 3 passed，Ruff
  与 `--help` 通过。（脚本由助手编写，修复了用户初稿 `cached = Path` 的断裂调用。）
- Task 13 真实模型已下载到 `data/index/mineru_models/`：LayoutReader `model.safetensors`
  约 713MB、`ch/en` 四套 OCR 权重、YOLO 布局等共 7 个文件均非空；`mineru_doctor.py
  --strict` 全部 `[OK]`、退出码 0；`data/` 未进入 Git。Task 13 为 runtime-only，无代码提交。
- Task 14 已提交为 `2e147af test(parse): 验收 MinerU 中英文 GPU OCR`：
  `scripts/demo_mineru_local.py` 增加 `--lang auto|ch|en`、打印逐篇语言路由并校验
  `language.json`；`tests/test_mineru_bilingual_real.py` 为无 mock 双语真实测试，缺环境
  变量时明确失败不 skip。用户已在真实 GPU 上运行 Demo 与集成测试通过：英文
  `en/pdf_text`、中文 `ch/metadata`，`paper.md` 非空、`language.json` 记录完整。

## 待处理问题

- 下一步实施 Task 15 解析调度器 `src/paper_rag/parse/dispatcher.py`：MinerU 成功→
  `mineru`；MinerU 失败且 PDF 有正文→PyMuPDF `degraded`；扫描件失败或降级空正文→
  `failed` 且批处理继续；禁用 fallback 时重抛 MinerU 错误。保留 `(Path, parser_name)`
  接口，`parse_status.json` 记录 `paper_id/status/parser/reason`，用正则剔除页标记后判定
  是否有实义正文。助手写 `tests/test_parse_dispatcher.py` 与
  `scripts/demo_parse_dispatcher.py`，用户写 `dispatcher.py`。
- `src/paper_rag/ingest/arxiv_source.py` 仍有未提交的 Task 6 元数据持久化迁移 diff；
  不要被后续提交顺带纳入，应作为独立提交，并配套单独规划
  `fix(ingest): 为 arXiv 真实请求增加超时`。
- （已完成）真实 MinerU 模型已下载、Doctor 严格通过、中英文真实 GPU OCR Demo 与
  无 mock 集成测试已通过（Task 13–14）。切块前的真实解析门禁已满足，剩余仅 Task 15。
- Semantic Scholar 真实验收仍未完成：缺少 API key；
  `scripts/demo_semantic_scholar_source.py` 已创建但未提交，且尚未创建无 mock 的
  `tests/test_semantic_scholar_source_real.py`。不得将该采集源标记为完整完成。
- 2026-07-31 Task 6 arXiv 真实验收出现外部网络卡顿：
  `uv run pytest -vv -s tests/test_arxiv_source_real.py` 卡在
  `[2/5] 查询真实 arXiv API 并下载 PDF` 后的 `ArxivSource().fetch()`。排查结论是
  通用网络可用，但 arXiv 真实端点不稳定：`export.arxiv.org/api/query` 的部分查询形式
  会超时，`arxiv` Python 包 4.0.0 内部 `requests.Session.get()` 没有显式 timeout，
  慢链路会表现为长时间卡住。探针证据：`curl -I --max-time 20
  'https://export.arxiv.org/api/query?search_query=id:1706.03762'` 超时；
  `curl -I --max-time 20 'https://export.arxiv.org/api/query?id_list=1706.03762'`
  返回 200；`arxiv.Client(...).results(Search(id_list=['1706.03762']))` 曾耗时约
  17 秒才返回；PDF 端点可达但下载偏慢。当前判断：这不是 Task 6 元数据持久化迁移引入的
  逻辑问题，应单独规划 `fix(ingest): 为 arXiv 真实请求增加超时`，不要混入 Task 6
  迁移提交。
- `AGENTS.md`、`CLAUDE.md` 为用户未跟踪文件，助手不得擅自纳入课程提交。
- `demo-local-data/`、`demo-url-data/`、`demo-arxiv-data/`、`demo-openalex-data/`、
  `demo-pymupdf-data/` 和新增 `demo-mineru-data/` 是用户保留的真实产物，不得纳入 Git。
- 全量测试须用 `uv run python -m pytest`（把工作目录纳入 sys.path），否则
  `scripts.init_store` 相关用例会因 `scripts/` 未安装为包而导入失败。真实测试
  （`tests/test_*_real.py`、`tests/test_mineru_bilingual_real.py`）在缺服务/密钥/环境变量
  时按约定明确失败、不 skip，需单独运行。全量 Ruff 仍有既有历史问题位于
  `scripts/demo_qdrant_store.py`、`src/paper_rag/ingest/arxiv_source.py`、
  `src/paper_rag/store/qdrant_store.py`，不在当前任务范围，勿混入后续提交。
- 仍未形成业务 checkpoint 的未跟踪文件：`src/paper_rag/parse/__init__.py`、
  `src/paper_rag/parse/fallback_pymupdf.py`、`tests/test_fallback_pymupdf.py`、
  `tests/test_mineru_doctor_script.py`、`tests/test_parse_package.py`、
  `scripts/mineru_doctor.py`、`scripts/demo_fallback_pymupdf.py`、
  `scripts/demo_semantic_scholar_source.py`、GPU 配置 `config/magic-pdf.json`。其中
  `tests/test_parse_package.py` 已由助手改为 hermetic（快照并弹出 sys.modules 再 fresh
  导入），可与 `parse/__init__.py` 作为独立的解析包入口 checkpoint 提交。课程状态提交
  不得顺带纳入这些文件。

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
- `e45e6ac feat(ingest): 实现 Semantic Scholar 采集边界`
- `e12284d docs(parse): 设计中英文 OCR 语言路由与降级策略`
- `ddff85b docs(parse): 规划中英文 OCR 语言路由实施步骤`
- `13dd3a6 feat(config): 支持 MinerU OCR 语言自动路由`
- `8afa910 feat(ingest): 为论文元数据添加语言字段`
- `381f748 feat(ingest): 保留论文人工语言标注`
- `3931b6a refactor(ingest): 统一本地论文元数据持久化`
- `e193b7f refactor(ingest): 统一 URL 论文元数据持久化`
- `d8b2673 refactor(ingest): 保留 OpenAlex 论文语言元数据`
- `7a88b37 refactor(ingest): 统一 Semantic Scholar 元数据持久化`
- `5abecae feat(parse): 实现中英文 OCR 语言自动判断`
- `8ce42b1 feat(parse): 诊断 Mineru 中英文OCR权重`
- `7207705 feat(parse): 诊断 MinerU 中英文 OCR 权重`（Task 10 测试可复现性修复）
- `768548d feat(parse): 按论文选择 MinerU OCR 语言`（Task 11）
- `ba4d0fb feat(parse): 增加 MinerU 双语模型下载入口`（Task 12）
- `2e147af test(parse): 验收 MinerU 中英文 GPU OCR`（Task 14；Task 13 为 runtime-only 无提交）
- 课程状态提交不得包含业务文件。

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
