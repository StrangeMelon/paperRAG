# 学习历史归档

本文件是 `LEARNING_STATE.md` 的历史归档：逐课完成记录、逐提交注解与已关闭问题的
排查细节。**恢复课程不需要读取本文件**，仅在需要追溯某个历史决定或验收证据时查阅。

## 已完成的验证（P1–P4 解析层，截至 2026-07-31）

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
  持久化输出，并检查标准化 Markdown、figures、layout 与原始产物。
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
- 双语设计文档与实施计划已确认并提交（`e12284d`、`ddff85b`）。计划共 15 个逐文件任务，
  固定官方模型仓库修订 `a4f6a8d29a4d96730f90ea174a9322e842b93552`，明确七个布局、
  LayoutReader 和中英文 OCR 文件。
- Task 1 已提交为 `13dd3a6 feat(config): 支持 MinerU OCR 语言自动路由`：
  `mineru.lang` 配置值域为 `auto | ch | en`，默认值改为 `auto`。
- Task 2 已提交为 `8afa910 feat(ingest): 为论文元数据添加语言字段`：
  `PaperMeta.language` 支持领域语言 `zh/en/None` 并拒绝供应商值 `ch`。
- Task 3 已提交为 `381f748 feat(ingest): 保留论文人工语言标注`：
  新增 `src/paper_rag/ingest/metadata.py`，统一写入 `meta.json` 并保留已有人工语言。
- Task 4 已提交为 `3931b6a refactor(ingest): 统一本地论文元数据持久化`。
- Task 5 已提交为 `e193b7f refactor(ingest): 统一 URL 论文元数据持久化`。
- Task 7 已提交为 `d8b2673 refactor(ingest): 保留 OpenAlex 论文语言元数据`。
- Task 8 已提交为 `7a88b37 refactor(ingest): 统一 Semantic Scholar 元数据持久化`；
  真实 API 验收仍等待 API key。
- Task 9 已提交为 `5abecae feat(parse): 实现中英文 OCR 语言自动判断`：
  `resolve_ocr_language()` 支持强制 `ch/en`、人工元数据优先、PyMuPDF 文本采样和失败
  回退到 `ch`，对应测试覆盖中英文、扫描件、损坏元数据与损坏 PDF。
- Task 10 已提交为 `8ce42b1 feat(parse): 诊断 Mineru 中英文OCR权重`：Doctor 的
  `auto` 模式展开检查 `ch/en` 四个 OCR 权重，新增 OCR 权重路径/可用性函数和
  LayoutReader 两个文件检查，并接入 `diagnose()`。干净快照复核发现聚合测试依赖未跟踪的
  `config/magic-pdf.json`，测试可复现性修复后重提交为
  `7207705 feat(parse): 诊断 MinerU 中英文 OCR 权重`（`tmp_path` 自建临时配置 +
  monkeypatch `cfg.PROJECT_ROOT`）。
- Task 11 已提交为 `768548d feat(parse): 按论文选择 MinerU OCR 语言`：新增
  `_select_available_ocr_language()`，CLI 不再接收 `auto`；英文权重缺失且中文可用时降级
  `ch` 并写出含 `model_fallback` 的 `language.json`。`test_mineru_local.py` 与
  `test_parse_language.py` 共 35 passed。
- Task 12 已提交为 `ba4d0fb feat(parse): 增加 MinerU 双语模型下载入口`：
  `scripts/download_mineru_models.py` 固定官方修订、7 个权重的远端/本地映射、大小校验、
  `.part` 原子替换与复用；`tests/test_download_mineru_models.py` 3 passed。（脚本由助手
  编写，修复了用户初稿 `cached = Path` 的断裂调用。）
- Task 13 真实模型已下载到 `data/index/mineru_models/`：LayoutReader `model.safetensors`
  约 713MB、`ch/en` 四套 OCR 权重、YOLO 布局等共 7 个文件均非空；`mineru_doctor.py
  --strict` 全部 `[OK]`、退出码 0；`data/` 未进入 Git。Task 13 为 runtime-only，无代码提交。
- Task 14 已提交为 `2e147af test(parse): 验收 MinerU 中英文 GPU OCR`：
  `scripts/demo_mineru_local.py` 增加 `--lang auto|ch|en`、打印逐篇语言路由并校验
  `language.json`；`tests/test_mineru_bilingual_real.py` 为无 mock 双语真实测试，缺环境
  变量时明确失败不 skip。用户已在真实 GPU 上运行 Demo 与集成测试通过：英文
  `en/pdf_text`、中文 `ch/metadata`，`paper.md` 非空、`language.json` 记录完整。
- Task 15 已提交为 `c9c0c24 feat(parse): 实现解析后端降级与失败隔离`：
  `dispatcher.parse_pdf()` 返回 `(Path, parser_name)`；MinerU 成功→`succeeded/mineru`，
  MinerU 失败且 PDF 有正文→`degraded/pymupdf` 并保留 MinerU 原因，扫描件或降级空正文→
  `failed` 且抛 `ParseError`，禁用 fallback 时原样重抛 `MineruError`。
  `_has_meaningful_markdown()` 用正则剔除 `<!-- page N -->` 后判定实义正文，杜绝空结果
  伪装成功。`tests/test_parse_dispatcher.py` 9 passed。
- `scripts/demo_parse_dispatcher.py` 为零 mock 真实降级 Demo：临时配置把 `mineru.cli`
  指向不存在的可执行文件，让生产 `_resolve_cli()` 真实返回 `None` 并抛 `MineruError`；
  输出隔离到 `tempfile.mkdtemp()` 并清理。退出码语义：`0` 降级成功、`1` 不变量被破坏、
  `2` 所有后端均无正文。助手自测：文字 PDF `degraded/pymupdf` 9 项不变量全 OK、EXIT=0；
  空白扫描件 `failed/pymupdf`、EXIT=2。
- 2026-07-31 发现并修复解析层可复现性缺口：`src/paper_rag/parse/__init__.py` 与
  `fallback_pymupdf.py` 从未进入 Git，调度器降级路径在新克隆上是死代码。已按四个独立
  提交补齐（`75242f9`、`646cb94`、`fa4652f`、`12cd6bb`），补齐后 `git archive HEAD`
  干净快照实跑解析层七个测试文件 55 passed，缺口关闭。补提交边界逐个核对无夹带；
  `git check-ignore -v` 验证六个 `demo-*-data/` 命中 `.gitignore:131`、
  `data/index/mineru_models` 命中 `.gitignore:6`；已跟踪文件最大为 `uv.lock`（约 1MB），
  无 PDF、模型权重或解析产物进入 Git 历史。
- 切块切片 0+1 已提交为 `c22cf03 feat(chunk): 章节切分器切片 1 markdown 标题路径与
  Body 兜底`（2026-08-01）：包入口 `chunk/__init__.py` 与 `section_splitter.py` 的
  markdown 标题路径、Body 兜底、仅限关键字 `language` 参数；
  `tests/test_chunk_package.py` + `tests/test_section_splitter.py` 10 passed，Ruff 通过。

## 已关闭/已诊断问题的细节

- **arXiv 真实端点不稳定（2026-07-31 诊断，待修复）**：
  `uv run pytest -vv -s tests/test_arxiv_source_real.py` 卡在
  `ArxivSource().fetch()`。排查结论：通用网络可用，但
  `export.arxiv.org/api/query` 的部分查询形式会超时；`arxiv` Python 包 4.0.0 内部
  `requests.Session.get()` 没有显式 timeout，慢链路表现为长时间卡住。探针证据：
  `curl -I --max-time 20 '...search_query=id:1706.03762'` 超时；
  `curl -I --max-time 20 '...id_list=1706.03762'` 返回 200；
  `arxiv.Client(...).results(Search(id_list=['1706.03762']))` 曾耗时约 17 秒；PDF 端点
  可达但下载偏慢。结论：非 Task 6 迁移引入的逻辑问题，应单独规划
  `fix(ingest): 为 arXiv 真实请求增加超时`。

## 逐提交注解（截至 2026-08-01）

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
- `13dd3a6 feat(config): 支持 MinerU OCR 语言自动路由`（Task 1）
- `8afa910 feat(ingest): 为论文元数据添加语言字段`（Task 2）
- `381f748 feat(ingest): 保留论文人工语言标注`（Task 3）
- `3931b6a refactor(ingest): 统一本地论文元数据持久化`（Task 4）
- `e193b7f refactor(ingest): 统一 URL 论文元数据持久化`（Task 5）
- `d8b2673 refactor(ingest): 保留 OpenAlex 论文语言元数据`（Task 7）
- `7a88b37 refactor(ingest): 统一 Semantic Scholar 元数据持久化`（Task 8）
- `5abecae feat(parse): 实现中英文 OCR 语言自动判断`（Task 9）
- `8ce42b1 feat(parse): 诊断 Mineru 中英文OCR权重`（Task 10）
- `7207705 feat(parse): 诊断 MinerU 中英文 OCR 权重`（Task 10 测试可复现性修复）
- `768548d feat(parse): 按论文选择 MinerU OCR 语言`（Task 11）
- `ba4d0fb feat(parse): 增加 MinerU 双语模型下载入口`（Task 12）
- `2e147af test(parse): 验收 MinerU 中英文 GPU OCR`（Task 14；Task 13 runtime-only 无提交）
- `fcb4122 docs(course): 记录 Task 11-14 完成并规划 Task 15`
- `c9c0c24 feat(parse): 实现解析后端降级与失败隔离`（Task 15，双语 OCR 计划收尾）
- `75242f9 feat(parse): 实现解析包入口与 PyMuPDF 兜底解析`（补提交，修复克隆即失败）
- `646cb94 feat(parse): 增加 MinerU 环境诊断脚本`（补提交）
- `fa4652f chore(git): 忽略本地 Demo 真实产物目录`（补提交 `.gitignore`）
- `12cd6bb chore(parse): 提交 MinerU GPU 运行配置`（补提交 `config/magic-pdf.json`）
- `c74e8ff docs(course): 记录 Task 15 完成与解析层补提交`
- `1191284 docs(course): 记录切块切片方案与中文扩展约束`
- `c22cf03 feat(chunk): 章节切分器切片 1 markdown 标题路径与 Body 兜底`
