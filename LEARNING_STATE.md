# 学习状态

## 当前定位

- 上下文恢复点：2026-07-30（双语 OCR 设计已确认，实施计划待提交，真实模型尚未下载）
- 当前阶段：P4（采集、解析和切块）
- 当前课次：P4.9（MinerU 本地 GPU OCR 解析器与中英文语言路由）
- 上一个确认完成文件：`scripts/mineru_doctor.py`
- 当前待处理事项：先提交双语 OCR 实施计划；随后执行计划 Task 1，把错误的固定 `en`
  测试契约改为 `auto`，扩展配置值域测试，再由用户修改 `src/paper_rag/config.py` 和
  `config/default.yaml`
- Semantic Scholar 恢复点：取得 API key 后运行
  `scripts/demo_semantic_scholar_source.py`，再创建并运行无 mock 真实集成测试
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
  目录、布局权重、按语言的 OCR 权重、CLI 版本和报告汇总。`tests/test_mineru_local.py`
  已通过（23 passed, 13 warnings）；警告来自第三方 `rapid_table` 的 SyntaxWarning。
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
  `docs/superpowers/plans/2026-07-30-bilingual-mineru-language-routing.md` 已生成并自审：共
  15 个逐文件任务，固定官方模型仓库修订
  `a4f6a8d29a4d96730f90ea174a9322e842b93552`，明确七个布局、LayoutReader 和中英文 OCR
  文件；占位符扫描无匹配，`git diff --check` 通过，但计划文件尚未提交。

## 待处理问题

- 当前首先提交
  `docs/superpowers/plans/2026-07-30-bilingual-mineru-language-routing.md`，规范提交信息为
  `docs(parse): 规划中英文 OCR 语言路由实施步骤`。
- 实施计划 Task 1 的首个 RED 是把 `tests/test_mineru_gpu_config.py` 中临时的固定 `en`
  期待改为 `auto`，并在 `tests/test_config.py` 增加 `auto/ch/en` 值域测试。当前未跟踪测试
  文件仍包含临时 `en` 期待，不得在修正前提交。
- `data/index/mineru_models/` 中尚未下载任何真实 MinerU 模型。必须在 Doctor 完成后，
  按固定官方修订下载 `Layout/YOLO`、`Layout/LayoutReader` 及 `ch/en` 四个 OCR 权重；
  模型下载完成前不得宣称 GPU OCR 可用。
- `config/default.yaml` 当前仍是 `mineru.lang: null`；必须按 Task 1 改成应用层
  `mineru.lang: auto`，且不得把字符串 `auto` 原样传给 `magic-pdf`。
- 真实 MinerU Demo 与无 mock 集成测试尚未运行；必须在模型准备、Doctor 通过后使用
  用户选择的真实 PDF 执行，并同时确认 Markdown/图片/layout 产物和真实 GPU 使用。
- Semantic Scholar 真实验收仍未完成：缺少 API key；
  `scripts/demo_semantic_scholar_source.py` 已创建但未提交，且尚未创建无 mock 的
  `tests/test_semantic_scholar_source_real.py`。不得将该采集源标记为完整完成。
- `AGENTS.md` 为用户未跟踪文件，助手不得擅自纳入课程提交。
- `demo-local-data/`、`demo-url-data/`、`demo-arxiv-data/` 和
  `demo-openalex-data/`、`demo-pymupdf-data/` 是用户选择保留的真实结果，不得纳入 Git。
- 当前解析实现、测试、Demo、GPU/OCR 配置和用户运行数据均尚未形成业务 Git
  checkpoint；课程状态提交不得顺带纳入这些文件。
- 双语实施计划尚未提交；本次课程状态提交不得顺带提交该计划文件。

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
- 课程状态提交不得包含业务文件。

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
