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
- 切块切片 2 已提交为 `08401bd`（2026-08-01）：英文纯文本标题四形态（行内 Abstract、
  孤立编号+标题、行内编号标题、裸规范标题）与段落边界/Table 上下文守卫、最小重叠
  去重。RED 8 failed → GREEN。
- 切块切片 3 已提交为 `4b626e9`（2026-08-01）：标题清洗（编号前缀/两端标点/空白
  压缩）、描述性合法性（词数/长度/大写比例/领域关键词）、first-abstract 守卫、
  层级只用编号 token 计算并封顶 4 级（修正基准把整行传入层级计算的缺陷）。
- 切块切片 4 已提交为 `8615c08`（2026-08-01）：markdown 优先级去重（markdown 标题
  顶替重叠的纯文本标题）、References 尾部过滤（仅放行 Appendix 并解除过滤）、
  `appendix ` 规范前缀、英文集成用例。基准的同起点同名去重分支在重建版不可达
  （纯文本扫描跳过 `#` 行），未保留并在 docstring 记录。已知误报面：段首
  "Appendix …" 叙述句会被判为标题，用户决定保留基准行为、后续再评估收紧（方案 B：
  标识符 token + 分隔符/标题式后文）。
- 切块切片 5 + 5b 已提交为 `ef7ed00`（2026-08-01）：中文规范白名单（比较前压掉
  内部空格兼容 "摘 要" 排版）、中文编号五形态（`一、`/`（一）`/`第X章`/`1、`/
  阿拉伯点分 `1. 1.1 2.3.1`；`（一）` 二级、点分按段数、其余一级）、`摘要：`
  行内切分、合法性用 2–30 字符数 + 必须含汉字 + 括号/句读否决、`图/表/算法`+编号
  黑名单（必须带编号，不误伤 "表示学习"）、单位字符黑名单 `倍/%/‰`（拦 "3.5 倍…"
  小数量词）、中文附录正则要求 "附录+短编号" 整行匹配（比英文前缀规则紧）、
  `参考文献→附录` 尾部过滤与英文共用、`zh/en/None` 语言路由（`_matchers_for`
  切换匹配器列表；双语全开的安全性来自两侧合法性都要求本语言文字）。
  RED 9 failed、2 failed 两轮 → GREEN。5b 为用户提出后追加的第五形态。
- `section_splitter.py` 真实验收已提交为 `5caefcb`（2026-08-01，新增
  `scripts/demo_section_splitter.py`）：4 份真实解析产物（MinerU 中文期刊
  《综合能源服务区块链》16 节 + MinerU 英文 Graph-Mamba 25 节 / LocAgent 26 节 +
  PyMuPDF 密排 Graph-Mamba 9 节）全部断言通过、exit 0。验收发现并修复两个缺陷：
  a) 中文期刊 markdown 标题纯数字直贴形态（`# 1综合…`）清洗残留，补
  `[0-9]{1,2}(?=汉字)` 分支（限 1–2 位保护年份开头的真标题）；点分清洗分支语言
  中立，顺带修好 `# 4.2Dataset Construction`。b) 真实 PyMuPDF 产物整篇无空行，
  切片 2 给英文编号形态加的段落边界守卫比基准更严，真实文件整篇只剩 3 节；删除
  该守卫与基准对齐（误报由描述性合法性兜住），中文编号形态保留守卫（中文合法性
  判别力弱、主路径是 MinerU），守卫不对称有正反测试。最终 57 passed；已知边界
  （字母编号附录被尾部过滤、图注 markdown 标题保留、单位黑名单精度、中文密排
  未验收）记录在 LEARNING_STATE 切块层一节。
- `text_chunker.py` 已完成（2026-08-01，助手直接编码，一课完成
  RED → 实现 → GREEN → Ruff → 真实 Demo；已提交为 `55b4e3d`，文件
  `src/paper_rag/chunk/text_chunker.py` + `tests/test_text_chunker.py` +
  `scripts/demo_text_chunker.py`）。保留基准"段落贪心打包 + 尾段 overlap"骨架，
  四处经确认的偏离：a) 偏移改 `body.find` 真实定位，不变量强化为
  `body[char_start:char_end] == text`（基准 cursor 算术在 4+ 连续换行时漂移、
  末块 char_end 多 2）；b) 回退 token 估算 CJK 逐字计 1 + 其余 len//4（基准
  len//4 对中文低估 4–7 倍，实测 cl100k_base 中文约 0.87 字符/token、英文约
  6.07）；c) 超长段落句子切分：zh `[。！？；…]+` 加后随收尾引号、en `[.!?]` 加
  后随空白（小数 3.5 不切）、None 取并集，无句读段按 token 等分硬切保证上界
  （基准从不切超长段；真实数据里 PyMuPDF 密排"段落"最大 1469 tok、中文期刊
  单段 3545 tok）；d) overlap 防重守卫：尾段 token*2 > target 放弃携带（基准
  会把接近 target 的尾段重复输出成独立 chunk）。RED 16 failed（ModuleNotFound）
  → GREEN 16 passed、全量 212 passed。RED→GREEN 之间修的两处均为**测试构造
  问题**而非实现缺陷：语言路由用例三句各恰 10 字符，en 路由的等分硬切碰巧落在
  句边界上无法区分，改 6/10/10 不等长；中文引号用例句子本身 11 tok 超过
  target=10 落入硬切，target 调为 12。真实验收 `scripts/demo_text_chunker.py`
  （复用 splitter 课 4 份真实产物，逐 chunk 断言偏移回切原文、token ≤
  1.2×target、无重复 chunk、语言路由按 language.json 生效）一次通过 exit 0：
  中文期刊 16 节 60 块（最大段 3545 tok → 最大 chunk 546 tok）、Graph-Mamba
  MinerU 25 节 39 块（2571→537）、LocAgent 26 节 44 块（712→503）、PyMuPDF
  密排 9 节 40 块（1469→499）。546 略超 target 源于硬切按字符等分而 BPE 密度
  不均（中文期刊无句读表格区），在文档化余量内。已记账边界与接口决定同步记入
  LEARNING_STATE 切块层一节。
- `contextual.py` 已完成（2026-08-01，与 text_chunker 同日连课；已提交为
  `39e296f`，用户将当时的课程文档更新一并纳入该提交；
  `src/paper_rag/chunk/contextual.py` + `tests/test_contextual.py` +
  `src/paper_rag/config.py`（+1 行）+ `config/default.yaml`（+1 行））。基准仅
  11 行：`chunk.context_prefix` 模板 format 后拼在 chunk 文本前；沿调用链确认的
  关键位置：builder 写入 `context_text`，`ingest_pipeline.py:188` 用 BGE-M3 嵌入
  的正是 `context_text`（裸 `text` 走 BM25），即前缀直接塑造稠密向量。英文隐式
  假设三条：英文标签使 zh 论文嵌入输入三语混排；空值渲染出 `[Title: ]` 死架子
  嵌进无标题论文的每个 chunk；模板单一全局无语言机制。经确认扩展（两问均选
  推荐项）：zh 路由到新配置键 `chunk.context_prefix_zh`
  （`[标题: {title}] [章节: {section}]\n`），en/None 用基准模板；空 title/section
  按渲染后 `[...: ]` 形态整段移除（半/全角冒号均识别），都空直接返回原文，
  自定义模板不用该形态时退化为基准空串填入；值含花括号安全（format 只解释模板
  占位符）已测试。RED 9 failed（ModuleNotFound）→ GREEN 9 passed，全量
  221 passed，Ruff 全绿（config.py 仅 +1 行，其既有格式漂移未触碰）。纯函数无
  独立 Demo，真实链路覆盖并入 builder 课。
- `parse/page_markers.py` 小课次已完成（2026-08-01，已提交为 `395309d`）：
  `inject_page_markers(md, layout)` 按 content_list 块序推进游标、块文本前
  20 字符 `md.find` 顺序对齐，每页第一个可定位文本块处在**行首**插入
  `<!-- page N -->`（N = page_idx+1，1 基，与 PyMuPDF 同款；markdown 标题的
  `# ` 不被拆开）。降级规则不抛错：非 list 布局（middle.json 形态）原样返回；
  定位失败跳过该块、同页后续块兜底；锚定块最短长度双档（含 CJK ≥2 字符保
  "引言/摘要"，纯 ASCII ≥4 字符拦页脚 "1"/"12" 假匹配）。接入
  `_normalize_into`：布局选择提前到写 paper.md 之前，content_list 先注标再落盘，
  layout.json 写盘不变。RED 12 failed（collection error）→ GREEN 12 passed、
  全量 233 passed、解析层回归 47 passed。真实验收 `scripts/demo_page_markers.py`
  双通道 exit 0：纯函数注标（中文期刊 15 页注 14、Graph-Mamba 15 页注 14、
  LocAgent 17 页全注，页码严格递增、剥标逐字节还原）+ 集成强断言（真实
  `_mineru_raw` 重跑 `_normalize_into`，剥标后与存量已验收 paper.md 逐字节
  一致，证明"只多标记不动内容"）。缺页逐块诊断：两个缺页在 content_list 里
  只有空文本块（纯图表页），属预期降级；连带影响（该页图表 chunk 继承前页
  页码 ±1）记入 LEARNING_STATE 已记账边界。应用户要求，Demo 集成段产物持久化
  到 `demo-page-markers-data/parsed/<id>/`（.gitignore 的 `demo-*-data/` 覆盖，
  每轮只清理自己的产物；存量 demo-mineru-data 只读），供人工查看与 builder 课
  复用。同日真实块检查（用户要求查看 chunk 形态）发现三个后续课次决策点
  （参考文献节去留 / 全角点 `．` 入 zh 边界集 / 页标记混入 chunk 文本），
  记入 LEARNING_STATE"块检查观察"。
- `builder.py` 课次已完成（2026-08-01，已提交为 `eb897ed`）：切块层组装枢纽
  `build_chunks(paper_id, parsed_dir, *, title) -> (sections, chunks)`，签名与
  基准一致。课前方案经用户三项确认：多模态循环后补（multimodal_chunker 课接回，
  vision enrich 钩子一并推迟——文本主路径挂钩子只会空打 warning）、偏移精确化
  纳入本课修复、页码标记保留在 chunk 文本（块检查观察决策点 c 就此关闭）。
  三处相对基准的差异：a) 语言贯通——新增 `_read_language` 从
  `parsed_dir/language.json` 读 `document_language`（缺失/损坏/域外值如 `fr`
  一律降级 None 不终止），传给 `split_sections`/`chunk_text`/`with_context`，
  builder 是全链唯一语言枢纽；b) 偏移精确化——`md.index(body, sec.start)` 求
  body 真实起点做绝对偏移基准（基准 sec.start + 相对偏移在节头多空行时整体
  漂移），全局不变量升级为 `md[char_start:char_end] == chunk["text"]`；
  c) 页码归属 `_page_for_offset` 与基准逐字节同款，但 MinerU 产物因上课注标
  从全员 `page=None` 变为全员有页码（本链路核心修复）。chunk 字典 schema 与
  基准完全一致。RED 8 failed（collection error）→ GREEN 8 passed（过程中修正
  一个测试构造问题：页码标记被空行包围自成段落，极小 target=8 下独立成块，
  断言改为 pages == [1, 2, 2] 并注释真实语义）→ 全量 241 passed → Ruff 全绿。
  真实验收 `scripts/demo_builder.py` exit 0：输入组装自上课产物（MinerU 注标
  md 来自 demo-page-markers-data + language.json 来自 demo-mineru-data；
  PyMuPDF 原样），4 案例——中文期刊(zh) 16 节/61 块、Graph-Mamba MinerU(en)
  25/39、LocAgent(en) 26/45、Graph-Mamba PyMuPDF(None) 9/40；逐 chunk 断言
  md 切片逐字节回切、page 非空且单调不减、zh 论文 context_text 全部
  `[标题: …]` 中文模板。首轮发现同一论文双解析器 sha1 目录互相覆盖，输出目录
  加 `--mineru/--pymupdf` 来源后缀修复。产物持久化
  `demo-builder-data/parsed/<id>--<flavor>/chunks.json`。中文期刊 61 块比
  text_chunker 课的 60 多 1，是注入页码标记占 token 的自然结果。
- `sanity.py` 课次已完成（2026-08-01，含两个决策点落地共三个提交
  `800a531`/`b055e8f`/`e641019`）：基准 `sanity.py` 是**章节完整性打分器**而非
  块过滤器——`grade_sections(section_names) -> complete|partial|minimal|broken`，
  节名小写子串匹配四大区域（intro/method/experiment/conclusion），唯一调用方
  `ingest_pipeline` 把结果拼进 `parsed_with="{parser}+{quality}"` 供日后过滤
  坏解析，打分失败仅 warning。英文隐式假设：四张关键词表全英文，完美解析的
  中文论文会被降级误判。重建版扩展（用户确认语言路由方案）：新增 `_AREAS_ZH`
  四区中文关键词表，`grade_sections(names, *, language=None)`——zh 查中文表、
  en 查英文表（=基准行为）、None 查双表并集；输出四值标签与基准逐字一致。
  RED 15 failed（collection error）→ GREEN 15 passed。同课落地两个块检查观察
  决策点（均经用户确认）：a) 参考文献块**保留入库并打标**——builder 给
  References/Bibliography/参考文献 节的块加 `metadata["is_references"]=True`，
  普通块不带该键（schema 与基准逐键一致），检索课拿真实评测数据再决定降权/
  过滤（+2 测试）；b) 全角点 `．`（U+FF0E）加入 zh 句读边界集（一行修复
  +1 回归测试）。全量 266 passed（`tests/test_mineru_bilingual_real.py` 2 例
  按约定缺 `PAPER_RAG_REAL_ENGLISH_PDF` 环境变量明确失败不 skip，需单独带环境
  跑，与本课无关）、7 个触碰文件 Ruff check+format 全绿。真实验收
  `scripts/demo_sanity.py` exit 0（读 demo-builder-data 的 4 份 chunks.json）：
  中文期刊 zh 路由判 complete，同批节名 en 路由（=基准真实行为）仅 minimal
  （英文副标题的 architecture/model 碰巧命中 method 区）——基准对真实中文论文
  的降级误判实锤；三篇英文/None 路由全 complete；4 份产物参考文献打标块
  12/62、14/39、15/45、15/40（占比 20%–37%）。全角点修复真实效果：中文期刊
  参考文献从等分硬切变为按条目边界切（块尾均为 `…[J]．`/`…et al．` 完整条目），
  重跑 demo_text_chunker/demo_builder 均 exit 0（块数 60→61 / 61→62，上界
  看护不变，最大 546 块是无句读表格区与参考文献无关）。
- `multimodal_chunker.py` 课次已完成（2026-08-01，已提交为 `a972bd8`，切块层
  7 文件收官）：课前真实数据侦查改变了方案重心——三篇 MinerU 论文的 md 里
  0 管道表、0 `$$` 公式、只有图片（19/10/16 个 `![]` 且 alt 全空）；
  layout.json 里 image 块带 img_caption/page_idx，table 块（5/7/11 个）带
  table_caption 和自己的 img_path，md 图片与 layout 块 basename 同哈希可精确
  配对（配对率 3 篇均 100%）。方案三项经用户确认：前缀语言路由、
  layout 增强取"页码+图注+表重定型"最全档、vision 钩子本课接回。实现：
  抽取器识别正则与基准逐字一致，`compose_figure/table/formula_text` 模板助手
  公开导出（zh: 图:/表:/公式:/上下文:/路径:）；表格块 span 与 strip 后 raw
  对齐（基准 span 含尾随换行不可回切）；builder `_load_layout_assets` 配对
  增强 + 重定型，chunk_id 命名空间保持抽取器 kind 防撞，layout 异常优雅降级；
  vision try/except 钩子按基准同款接回。RED 11 failed（collection error）→
  GREEN 11、builder 新增切片 5 五个测试、聚焦 26 passed、全量 282 passed、
  Ruff 全绿。**本课首次执行新验收流程**：助手只自查开发测试，验收命令与预期
  效果交用户实跑对比——用户实跑 26 passed + `scripts/demo_multimodal_chunker.py`
  exit 0，逐项与预期一致：mm 分布 19(图14/表5)/10(3/7)/16(5/11)、图注覆盖
  19/19、10/10、9/16（LocAgent 7 个 layout 块自身无图注，如实记账）、PyMuPDF
  零召回、每篇一行 vision warning（预期诚实信号）。真实收获：OCR 模式下
  MinerU 把表格全渲染成图片，23 张表在基准链路会以"无 alt 的图"进索引，
  重定型让它们带 `表: {真实表题}` 语义入库。课后用户问询 vision 模块作用与
  顺序，确认其为可选增强（基准默认 enabled:false，无它系统完整），推迟到
  核心 RAG 链跑通后再建（需视觉 API key/本地模型，且 QA 跑通后才能度量价值）；
  同时发现依赖顺序修正：ingest_pipeline import embed.bge_m3，故 P5 先建
  `embed/bge_m3.py` 再收 `store/ingest_pipeline.py`。
- `embed/bge_m3.py` 课次已完成（2026-08-01，已提交为 `a91a30b`，P5 第一课）：
  BGE-M3 稠密嵌入封装，入库(ingest_pipeline)与查询(retrieve/dense)共用的
  唯一向量出口。与基准 1:1 保真：惰性单例 `_model()`（首次 encode 才 import
  FlagEmbedding）、设备策略（auto: macOS 强制 CPU / CUDA 优先）、fp16 仅
  非 CPU、`resolve_cached_snapshot` 离线缓存、只取 dense 1024 维（sparse 走
  BM25/FTS5）、空输入返回 []；唯一差异 Iterable 改 collections.abc。零中文
  代码扩展的一课：BGE-M3 原生多语种中英同空间，中文约束落在验收断言。环境
  准备：`uv sync --extra dev --extra embed --extra ingest --extra mineru`
  装入 FlagEmbedding 1.4.0（教训：只 sync 部分 extra 会卸掉其余 extra 的包，
  首次漏 dev 卸了 pre-commit/fastapi，补回后全量 287 passed 确认无损；四个
  extra 必须齐）；BAAI/bge-m3 预下载到项目本地 data/index/models（4.3G HF
  布局，双权重 blob 硬链接，全局 ~/.cache/huggingface 无污染，二跑日志显示
  本地快照路径证明离线加载）。边界测试 RED 5 failed → GREEN 5（假模型验
  批参数传递/只取 dense/tolist 转换/空输入不碰模型/生成器输入）；真实集成
  测试 tests/test_bge_m3_real.py 3 passed（无 mock 真加载）。用户实跑验收
  `scripts/demo_bge_m3.py` exit 0：真实中英混批 16 条全 1024 维数值健康、
  批次一致性余弦 1.0000、真实中文查询"区块链节点的信用值如何评价和更新"在
  中文期刊 62 个真实块上 top-3 全部命中"基于综合能源服务的信用评价体系"节
  （0.683/0.666/0.659 vs 全体均值 0.586）、中英同义句对 0.772 vs 不相关
  0.376（中英同空间实证，zh/en 同索引的模型侧前提成立）。课后用户两问均
  已核实答复：真实验收确实真加载模型（CUDA fp16 日志 + 本地快照路径）；
  应用户要求 Demo 增加向量落盘 demo-bge-m3-data/embeddings.json（查询 +
  top-3 chunk 完整 1024 维向量，L2 归一范数≈1、分量 ±0.23、全非零，
  余弦=点积与 Qdrant cosine 距离配合）。
- `store/ingest_pipeline.py` 课次已完成（2026-08-02 用户实跑验收通过，P5 收口；
  功能文件提交待用户执行，hash 下次进度更新时补记）：整链编排唯一入口
  `ingest(result, *, force=False)`，与基准同构（状态机 fetched→…→done、`_step`
  逐步记 ingest_runs、DOI>arxiv>标题三级去重 merged_into、done 跳过/force
  重建、元数据卡片插 chunks[0]、grade_sections 拼 parsed_with、Qdrant 先删后
  插替换语义）。方案三项经用户确认后的偏离：a) 语言贯通——builder
  `_read_language` 提升公开 `read_language`，pipeline 读 language.json 同一
  语言值喂 grade_sections（基准不传语言会误判中文论文）与元数据卡片模板；
  b) 卡片按语言路由——`_CARD_LABELS` 双语文案表（zh: 论文元数据记录。/标题:/
  作者:/年份:/摘要:），`_title_aliases` 缩写逻辑保持基准（中文标题优雅空集）；
  c) wiki 入队钩子接回（try/except 非致命 warning，与 vision 同策）。另发现并
  修复**基准死代码缺陷**：真空 chunks 守卫位于插卡之后，卡片必然存在使其永不
  触发；重建版前移到插卡之前（测试锁定）。边界测试 11 个（RED collection
  error → GREEN；全下游打桩：状态机顺序/卡片位次/zh 贯通到打分与卡片/去重/
  幂等/force/失败隔离/死代码修复），全量 296 passed、Ruff 全绿。真实验收
  `scripts/demo_ingest_pipeline.py` **全链首次端到端**（用户实跑 exit 0，与
  预期逐项一致）：真实 Graph-Mamba PDF → 真实 GPU MinerU（本篇仅约 11s）→
  49+1 chunks（卡片别名 GM）→ BGE-M3 → embedded Qdrant 50 点；
  parsed_with=mineru+complete、ingest_runs 四步全 ok、真实问题
  "How does Graph-Mamba capture long-range dependencies..." top-1 命中本论文
  Introduction 块(0.741)、重复 ingest → skipped/done、force 重建后点数不变。
  隔离设计：运行数据全落 demo-ingest-pipeline-data/（配置 monkeypatch 重定向
  paths+qdrant.local_path，models_dir 保持真实缓存只读复用）。课后答疑三则：
  向量双库分工（SQLite 无向量列存内容/溯源/FTS 底座，Qdrant 存向量+payload
  拷贝免回表）；数据库位置与打开方式（sqlite3 CLI + qdrant_client 片段）；
  远程 dashboard 访问（服务器有长跑 paper-rag-qdrant 容器 6333，VS Code/SSH
  端口转发 + embedded→服务模式一次性拷贝脚本 demo_pipeline_chunks 隔离
  collection）。

## P6 检索层逐课细节（2026-08-04 — 08-05，七文件全部完成）

漏斗全貌：百万 chunks → dense 20 + sparse 20 → RRF 融合 top_k*2 → cross-encoder
精排 → 论文多样化 top_k → format_evidence 渲染成带引用令牌的证据块。

- **`dense.py`** ✅ `45c86c0`：零偏离薄封装，存在理由是分层边界（查询与文档
  必须同一编码器同一空间）。文档侧嵌 `context_text`（带标题/章节前缀）、查询
  侧嵌裸查询的不对称是刻意设计（BGE-M3 对称双塔不需要 query 指令前缀）。
  4 打桩测试；Demo 只读复用入库课产物，中文问题跨语言命中英文论文，实证
  dense 层中文适配点为零。
- **`fts5.py`** ✅ `aeec63d`（ADR-0001，accepted）：三处结构性偏离——CJK
  bigram 分词镜像表 + JOIN 回原文、补 porter 词干器、去 SQL 触发器改 Python
  同步 + search 行数自愈。顺手修复基准查询清洗把 `Graph-Mamba` 黏成
  `GraphMamba`（改为替空格）。**真实数据修正了 ADR 结论**：基准中文召回不是
  "0 命中"而是 1/26（4%），命中者靠标点两侧巧合切出独立 run；断言改为
  "召回 <20%"，ADR 按实测改写。20 测试（真实临时 SQLite）。
- **`sparse_bm25.py`** ✅ `e5ce271`：中文 unigram 统一为 bigram（复用
  `segment_cjk`，跨模块一致性测试钉住，关闭 ADR-0001 粒度记账）；删只写不读的
  pickle 死代码；payload 填真实 section/title；**丢弃 0 分块**（基准把无词项
  交集的块凑满 top_k 返回，纯噪声）；规模护栏 `bm25_max_chunks: 200000`
  超限拒建 + 告警（20000 篇约束的直接产物，定位改写为小规模后备 + 评测对照）。
  Demo 双后端对照，porter 行为差用真实语料预核对过的词对
  （`prioritizes`/`prioritization`；`dependency` 词对因语料真含单数词形被弃）。
- **`hybrid.py`** ✅ `3ecc6c8`：RRF 只看名次不看分数（cosine 与 BM25 量纲不可
  加），k=60 钝化让"两条腿都靠前"者胜出；提升 `score_dense` 保留绝对相似度
  给 abstain 课。偏离一处：拷贝条目不改写调用方 dict。`fts5.sync_paper` 接入
  ingest index 步（非致命，测试钉住 delete→upsert→fts_sync 顺序，关闭 ADR-0001
  规模修订待办）。Demo 跨语言改述查询实证两腿失败模式互补（sparse=0 /
  dense 命中 / 融合不空），注入故障验证 bm25 接管。
- **`rerank.py`** ✅ `4140338`：bi-encoder 向量在入库时冻结，cross-encoder 让
  查询与文档逐词交互，精度换算力，只能用在漏斗末端少量候选上。懒加载单例 +
  `_LOAD_FAILED` 闩锁、四层降级全部回退 RRF 原序、单对裸 float 兼容。偏离：
  拷贝候选不改写调用方。Demo 真实加载 bge-reranker-v2-m3：英文精排纠偏 7/8、
  中文 gap 0.917、跨语言英文相关块 0.907 胜中文无关块 0.000（不被语言相同
  迷惑；元数据卡片因摘要相关被弃作无关对照）。
- **`format.py` + `pipeline.py`** ✅ `11d80d9`：format 是 `[chunk:<id>]` 硬
  不变量的物理源头（测试逐字钉死令牌行），信封保持英文（LLM 协议文本，非用户
  可见，中文正文原样在 body）。pipeline 四个组装心思：多查询池化取高分、中英
  模态线索追加定向轮、`top_k*3` 精排窗口、单篇限额 2 + 溢出补位。过渡偏离：
  `rag.query_rewrite`（P7）缺席时恒等改写 + warning，P7 落地自动恢复。
  **同课修复中文长查询串长分流**（hybrid 课记账边界）：≤6 字按 bigram 短语保
  精度，>6 字拆 bigram OR 词袋靠 IDF 排序，配置 `fts5_phrase_max_run: 6`。
  Demo 端到端联试；写作中自纠"单篇 ≤2"断言——补位场景下合法超限。

## P7 RAG/QA 层逐课细节（2026-08-05 起）

- **开题依赖核对（2026-08-05）**：逐文件扫基准 `rag/` 的 import，确认
  `query_rewrite.py` 在**模块顶层** `from .llm import chat`，`llm.py` 是整个 rag
  层依赖图的根（`intent_classifier`/`reflect`/`history`/`research_memory`/三条 QA
  路径全依赖它）。据此把 P7 第一课由状态文件原定的 `query_rewrite` **改为
  `llm.py`**，后续顺序修正为 llm → query_rewrite → intent_classifier →
  evidence_select → abstain → reflect → citation_check → qa_simple → qa_agentic
  → qa_stream。
- **流式与异步边界（用户提问后核实）**：基准全链路只有一处流式调用——
  `qa_stream.py::_stream_chat` 绕过 `chat()` 直接 `get_client()` 传 `stream=True`，
  唯一目的是给 DeerFlow 前端 SSE 供打字机效果（引用校验仍等 token 攒完再做）。
  其余全部非流式。异步同理：基准是"全同步引擎 + 网关边界线程池包装"
  （`async_api.py` 用 `anyio.to_thread.run_sync`，理由写在其 docstring：全面异步
  化要动约 30 文件，而客户端单例已复用连接、Qdrant/SQLite 亚毫秒、LLM 往返等待
  不占 GIL）。**结论**：`llm.py` 保持同步非流式；流式推迟到 `qa_stream` 课再决定
  是否重建，异步推迟到网关阶段作独立小课。
- **`llm.py`** ✅ `92662d2`（2026-08-05 真实验收通过）：模块级单例
  三件套 `_CLIENT`/`_CLIENT_KEY`/`_LOCK`，惰性构建 + 双重检查加锁，
  `(base_url, api_key)` 变化自动重建（测试 monkeypatch 与热改配置都不失效），
  openai 包在函数内懒导入（符合重依赖进函数约定）。`chat()` 签名与基准逐参
  一致（`model` 形参覆盖配置、默认 `temperature=0.2/max_tokens=1024`、
  `content or ""` 兜空）。**一处确认偏离——`llm.extra_body` 透传**：`_Llm` 新增
  `extra_body: dict[str, Any] = {}` + `default.yaml` 对应加键，非空时透传给
  `chat.completions.create`；空表（缺省）时调用形参与基准逐键一致，有专测钉死
  （`test_empty_extra_body_keeps_baseline_call_shape`）。动因是 Qwen(DashScope
  兼容模式)思考型模型**非流式调用必须 `enable_thinking: false`，否则 400**
  （联网核实：`parameter.enable_thinking must be set to false for non-streaming
  calls`），从此只是一行本地配置，换回 OpenAI 零成本。同课新建包入口
  `rag/__init__.py`（解析层"克隆即失败"教训）与 `.env.example`（DashScope 兼容
  模式模板；`.env` 已被 gitignore）。边界测试 13 个四切片全打桩不发网络；全量
  纯逻辑 380 passed 无回归；Ruff 干净。
- **`llm.py` 真实验收证据（2026-08-05，助手实跑，用户 `.env` 已就位）**：
  `scripts/demo_llm_chat.py` exit 0——真实 `CHAT_MODEL=qwen3.8-max` 非流式中文
  回复非空、`get_client()` 两次同一对象、`extra_body={enable_thinking: false}`
  被 DashScope 接受（英文回复非空）、`model` 形参覆盖 `qwen3.7-flash` 回"收到"；
  `tests/test_llm_real.py` 2 passed（无 mock，缺配置 `pytest.fail` 不 skip）。
  **偏离对 Qwen 全系有效**（用户实际用的是比预设 `qwen-plus` 更新的一代）。
  真实观察记账：`qwen3.8-max` 在短问题上会自行漂移话题（问 RAG 答成
  GraphRAG）——不违反 `llm.py` 的"非空字符串"契约，但预示 `query_rewrite` 课
  要求严格 JSON 输出时必须保留基准的 `re.search(r"\{.*\}", raw, re.DOTALL)`
  抠 JSON + 异常回退启发式。
- **`query_rewrite.py`** ✅（2026-08-05 真实验收通过，功能提交待用户执行）：按下条
  已确认方案全部落地。四处中文扩展逐一有测试钉死；`_original_aliases` 抽为独立
  纯函数收敛英文 + 两种中文别名正则；`_heuristic_variants` 的别名回查加
  try/except（sqlite 不可用时降级不拖垮主出口）；`rewrite` 对 LLM 返回的
  `variants` 做 `isinstance(list)` 与 `str()` 强化（真实模型会返回非列表/非字符串
  元素）。**CJK 邻接缺陷已实证**：基准正则 `\b[A-Z][A-Z0-9-]{1,20}\b` 在
  "基于GNN的图神经网络建模" 上 `findall` 返回 `[]`，改用显式 lookaround 后返回
  `['GNN']`，且英文侧边界不放宽（`xxGNNyy` 两版都不匹配，有反向专测
  `test_aliases_for_acronym_between_ascii_still_bounded`）。开题时误举的
  `Mamba` 是混合大小写，基准英文侧同样取不到，故改用全大写 `GNN` 举证。
  **P6 过渡契约翻转**：`tests/test_retrieve_pipeline.py::test_identity_fallback_when_query_rewrite_missing`
  是 P6 期间为"模块缺席"写的临时契约，query_rewrite 落地后必然失败（pipeline
  真的导入成功、不再 warning）。改为两个测试——`test_uses_real_query_rewrite_by_default`
  （断言拿到含 `bm25_query` 的完整载荷且无 identity warning，用
  `PAPER_RAG_FORCE_LOCAL_REWRITE=1` 保证不发网络）+
  `test_identity_fallback_when_query_rewrite_import_fails`（打补丁
  `builtins.__import__` 模拟导入失败，保住降级路径覆盖）。边界测试 35 个七切片
  全打桩；全量纯逻辑 426 passed 无回归；Ruff 干净（`src/paper_rag/rag` 与本课
  测试全绿；`arxiv_source.py` I001 与 `qdrant_store.py` B905 是本课未触碰文件的
  预存问题，归属 `7d86734` 与用户未提交 diff，未顺带修）。
- **`query_rewrite.py` 真实验收证据（2026-08-05，助手实跑）**：
  `scripts/demo_query_rewrite.py` exit 0 —— 英文问走英文模板产出 3 变体 + HyDE
  段落 + `bm25_query`；中文问 `检索增强生成怎么缓解大模型的幻觉问题?` 走中文模板，
  keywords 真实混出 `检索增强生成 retrieval augmented generation RAG 幻觉
  hallucination 事实性 faithful grounding 外部知识 external knowledge`（中英双语
  断层跨越已实证），变体含 1 条英文改写，HyDE 为中文论文腔段落；逃生门置真时
  LLM 调用计数 0；中文别名两形态均识别为 RAG。`tests/test_query_rewrite_real.py`
  3 passed（无 mock，缺配置 `pytest.fail` 不 skip）——含真实模型下的双语 keywords
  断言与"逃生门在真实环境下也一次不发"。落地后 pipeline 的 `identity rewrite`
  warning 已实测消失。
- **`.env` 加载缺陷与 `tests/conftest.py`（2026-08-05，用户报告后修复）**：用户按验收
  命令直跑 `uv run pytest -s tests/test_query_rewrite_real.py` 报"配置缺失"，用户自行
  判断"是不是没有 load_env"——诊断正确。根因：`test_llm_real.py` 自带 `_load_dotenv`
  并在导入时调用，写 `test_query_rewrite_real.py` 时**漏抄**这一步，该文件只读
  `os.environ` 而无人写入；助手自查时先执行过 `set -a; . ./.env`，恰好掩盖缺陷。
  **教训：真实验收命令交付前必须在剥离环境变量的干净 shell 里复跑一遍**（用
  `env -u VAR ...` 而非依赖当前 shell 状态）。修法不是在第二个文件再抄一份加载器，
  而是新建 `tests/conftest.py`——pytest 自动发现，收集测试前统一
  `load_dotenv(override=False)`（`python-dotenv>=1.0` 已在 pyproject 声明，另留极简
  兜底解析防依赖缺失），并删掉 `test_llm_real.py` 里的重复实现，只留一处真相。
  `override=False` 让已导出变量优先，支持 `CHAT_MODEL=qwen-turbo uv run pytest ...`
  临时覆盖。验证：剥离四个变量后两个真实文件 5 passed（证明确由 conftest 加载）；
  纯逻辑 426 passed 不变（conftest 对无配置环境无副作用）；`.env` 临时改名后三个
  真实用例仍 `Failed: 真实 LLM 配置缺失: ...` 明确失败、0.12s 退出不发网络——
  验收协议"缺配置不 skip"未被削弱。
- **`query_rewrite.py` 已确认方案（2026-08-05 用户确认，已实现）**：基准英文隐式
  假设与中文扩展——a) prompt 单一英文模板且要求 lowercase keywords（对中文无
  意义）→ 新增 `_query_language(q)` 按 CJK 码位占比判 `zh/en`（查询侧无
  `meta.json` 可依赖，只能启发式），zh 走中文模板；b) **跨语言 BM25 断层**：
  FTS5/BM25 是词面匹配，纯中文 keywords 永远打不中英文论文块 → zh 模板要求
  keywords **中英双语混出**并含 1 条英文改写变体（稠密侧 BGE-M3 本身跨语言，
  无需处理）；c) `_ORIGINAL_ALIAS_RE` 只识英文 "the original RAG paper" → 增
  `最初/原始/最早的 X`、`X 的原(始)论文` 两类中文形态；d) **`_aliases_for_title`
  的 CJK 邻接缺陷**：Python `re` 把汉字算 `\w`，`\b[A-Z][A-Z0-9-]{1,20}\b` 在
  "基于Mamba的图建模" 里 `于M` 之间无词边界，中文标题内嵌拉丁缩写词（重建语料
  真实存在，如别名 GM）提取不到 → 换显式 lookaround（前后非 `[A-Za-z0-9]`）；
  e) `_heuristic_variants` 的话题触发表是基准演示语料专属启发式，小写子串匹配
  对中文天然不触发、无害，照抄保留。`PAPER_RAG_FORCE_LOCAL_REWRITE` 逃生门、
  wiki 钩子（try/except + warning，与 vision 同款）、`_dedupe`、sqlite 别名回查
  全部按基准同构。
- **`intent_classifier.py` 已确认方案（2026-08-05 用户确认，已实现）**：基准 62 行，
  职责是"一次 LLM 调用决定后续检索力度"——factual 取少量块一轮（查一个数字）、
  reasoning 取中等两轮（跨论文比较）、explore 取大量三轮（综述）。基准同构部分：
  三分类 prompt、`re.search(r"\{.*\}", raw, re.DOTALL)` 抠 JSON、`temperature=0`
  （分类要确定性不要多样性）、`max_tokens=120`、**永不抛异常**（任何失败落
  reasoning 中间档——猜错也不会太离谱）。三处确认偏离：a) 基准把
  `top_k 5/10/15`、`max_iter 1/2/3`、`rrf_k 60` 硬编码在模块级 `_DEFAULTS`，违反
  CLAUDE.md "永不硬编码可调项" → 新增 `rag.intent` 配置段（三档各自 top_k/
  max_iter/rrf_k + `enabled` 开关），`enabled: false` 时直接走本地启发式且**零 LLM
  调用**（省一次往返，也是无 key 环境的逃生门，与上一课
  `PAPER_RAG_FORCE_LOCAL_REWRITE` 同款思路）；b) 基准 prompt 的三档说明与例子全
  英文，对中文提问引导力弱 → 复用上一课已验证的 `_query_language()`（直接 import，
  不重复实现），zh 走中文模板，把"对比/区别/综述/有哪些进展"这些中文信号词写进
  模板；c) 新增本地信号词启发式取代基准"一律落 reasoning"——中文含"区别/对比/
  相比/差异"→reasoning，含"有哪些/进展/综述/现状/概览"→explore，含"是多少/定义/
  是什么"且短→factual，英文侧同理。这让无 key 环境行为合理，并给边界测试一个
  不依赖 LLM 的确定性断言面。
- **`intent_classifier.py` 真实验收证据（2026-08-05，助手实跑）**：
  `scripts/demo_intent_classifier.py` exit 0——真实 Qwen 对中英各三问
  **6/6 全判对**：英文 `What is the FactScore metric?`→factual、
  `How do Self-RAG and CRAG differ...`→reasoning、`What are recent advances...`
  →explore；中文 `FactScore 指标是什么?`→factual、`Self-RAG 和 CRAG 在检索决策上
  有什么区别?`→reasoning、`检索增强生成近年有哪些研究进展?`→explore（证明中文模板
  的信号词引导有效）。配置驱动已实证：临时把 explore 档 `top_k` 改为 24，真实调用
  带出 24 而非硬编码 15；逃生门 `enabled: false` 时 LLM 调用计数 0、本地启发式仍
  判为 explore。`tests/test_intent_classifier_real.py` 8 passed，边界测试 37 passed，
  全量纯逻辑 463 passed，Ruff 干净。**本课起验收命令交付前已按上一课教训在
  `env -u OPENAI_BASE_URL -u OPENAI_API_KEY -u CHAT_MODEL -u SMALL_MODEL` 剥离
  变量的干净 shell 里复跑 Demo 与真实测试**（均通过，证明 conftest 加载链有效）。
- **本课一处自查捉到的测试自相矛盾（2026-08-05）**：启发式切片里原本用
  `检索增强生成近年有哪些研究进展?` 去断言"无信号词时落 reasoning 默认档"，但该问
  句本身含 explore 信号词"有哪些/进展"，被启发式正确判为 explore 而 RED 失败。是
  测试写错而非实现错——改用无信号词的中性问句隔离被测行为。
- **`evidence_select.py` 已确认方案与实现（2026-08-05 用户确认，已实现）**：基准
  127 行纯函数，职责是把检索宽窗(intent 三档 5/10/15 块)确定性收敛为 ≤4 块可
  引用证据(单篇 ≤2)——上下文纪律 + 引用面控制。打分四层：模型分键位优先链
  (score_rerank > score_rrf > score_dense > score) 占大头、词面重叠 ×0.2 裁平局、
  章节提示 ×0.03 微加分、1/rank ×0.001 兜底锚；trace 逐候选记账、落选也记。
  两处中文扩展：a) **基准 `_TOKEN_RE = [a-z0-9]+` 对中文问题一个 token 都抽不出，
  overlap 恒 0，平局裁决层对中文整体失明**（混排 "RAG 的召回率" 只抽出 rag，
  重叠失真）→ token 化改为拉丁词 + CJK bigram 并集（口径与 FTS5 ADR-0001 一致，
  问题侧与文本侧同一 `_tokens()` 函数；单字 CJK 组退化为单字 token）；
  b) `_SECTION_HINTS` 全英文 → 增 摘要/引言/绪论/方法/实验/评估/结果/结论/总结。
  保持基准：打分权重不进配置（算法常数非运维项）；`max_chunks=4, max_per_paper=2`
  保留签名默认，配置化推迟到 qa_agentic 课由调用方决定；trace 的
  `chunk in selected` 字典相等比较（同内容块会同标 selected 的理论边界）照抄记账。
  **新记账边界**：单字 CJK 问题({图}) 与文本 bigram 集合({图神,神经,...})无交集、
  overlap=0——与 P6 "bigram 索引无 unigram" 同口径，单字查询本就在词面盲区，由
  稠密分兜底（专测钉死该行为并写明缘由）。RED 期自纠一处测试设计：原断言单字
  问题 overlap>0，与 bigram 口径矛盾，改为断言 0 并记账。
- **`evidence_select.py` 真实链路验收证据（2026-08-05，助手实跑，无 LLM 消耗）**：
  `scripts/demo_evidence_select.py` exit 0——数据与 P6 retrieve_pipeline Demo 同源
  （英文 Graph-Mamba 库副本 + 中文期刊 62 块），真实 BGE-M3 + embedded Qdrant +
  FTS5 + 真实 reranker。英文问宽窗 8 → 选 2 块（窗口单篇故限额 2 生效），top-1
  model=0.992/overlap=0.750/hint=1；中文问 top-1 **overlap=0.650 > 0**（CJK bigram
  修复在真实数据上实证，选中块 section=摘要 亦吃到中文提示加分）；混合问单篇 ≤2；
  同一窗口两遍选择选集与全部候选得分逐项一致（确定性）。边界测试 18 passed，
  全量纯逻辑 481 passed，Ruff 干净。数据隔离在 `demo-evidence-select-data/`。
- **`rag/abstain.py`（P7 第五课，2026-08-05）**：三路证据充分性裁决——硬不变量
  "无证据宁拒答"的执行者，纯函数无 LLM。检索永远返回"最相似"top-k（哪怕域外
  问题），abstain 在 LLM 调用前按 top-`min_chunks` 归一化证据分裁四态：
  no_chunks / no_evidence（跳过 LLM 返回拒答文案）/ weak_evidence（LLM 照常但
  prompt 注入不足提示）/ confident。核心设计：分数字段优先级
  rerank > dense > score > bm25 > rrf，**首个可用字段服务全列表不混用**；分字段
  归一化进 [0,1] 带（rerank 天然 0-1、dense 裁剪、RRF ×15、BM25 sigmoid
  center=8）使阈值语义在 reranker 开/关时稳定；**信号质量分级 + fail open**——
  低质字段（BM25/RRF 排名型信号区分不了"无关块排前"与"相关块排前"）与字段
  全缺一律放行 confident，降级态经 `signal_quality` 透出。配置 `rag.abstain`
  （0.21/0.48/3 + 中文拒答文案）早前配置课已就位，本课零配置改动；YAML 校准值
  与签名缺省 0.20/0.40 不同，运行时 YAML 赢。两处确认偏离：a) 新增
  `weak_evidence_hint(language)` zh/en 路由 + `WEAK_EVIDENCE_HINT_ZH` 中文文案
  （基准硬编码英文提示，中文 prompt 风格割裂；en/None/未知语言仍走基准英文
  常量）；b) `no_chunks` 早退分支补 `signal_quality="no_chunks"`（基准漏键，
  四态 schema 拉平免去 trace 消费方缺键防御）。照抄并记账：BM25 sigmoid
  center=8 按英文语料校准、中文 FTS5 bigram 分布不同——但低质字段 fail open
  使该常数从不影响裁决只影响展示数字；阈值 0.21/0.48 系基准英文 33 题集校准，
  中英混合语料未重校（评测阶段可复刻 calibrate_abstain 重校）；空表早退先于
  enabled 检查（disabled + [] 仍返回 no_chunks，以基准代码行为为准）；
  `no_evidence_message` 基准本就中文，英文侧路由推迟到 qa_agentic 课定夺。
- **`abstain.py` 真实链路验收证据（2026-08-05，助手实跑，无 LLM 消耗）**：
  `scripts/demo_abstain.py` exit 0——数据与 evidence_select Demo 同源（英文
  Graph-Mamba 库副本 + 中文期刊 62 块），真实 BGE-M3 + embedded Qdrant + FTS5 +
  真实 reranker，阈值取 YAML 校准值非签名缺省。域内英文问证据分 **0.9903**、
  中文问 **0.9899** → 双双 confident；域外问"上海明天的天气怎么样"检索照样
  返回 8 块噪声但 rerank 证据分塌陷到 **0.0000** → **no_evidence（LLM 将被
  跳过）**，教科书式对照；同批域外块剥掉高质字段后落 score_bm25（0.1265）→
  confident + low_degraded、全剥 → confident + missing（fail open 实证）；
  空列表 no_chunks 带 signal_quality、enabled=False 逃生门、hint 路由全过。
  边界测试 31 passed；全量 `uv run python -m pytest -q` 525 passed + 2 failed
  （仅 `test_mineru_bilingual_real` 缺 `PAPER_RAG_REAL_ENGLISH_PDF` 环境变量的
  既有按约定"响亮失败"，与本课无关；该轮顺带跑过了 llm/rewrite/intent 真实
  测试均通过，消耗少量 LLM 调用）；排除 `*_real*` 后 502 passed；Ruff 干净。
  测试与 Demo 均在 `env -u` 剥离变量的干净 shell 复跑通过。数据隔离在
  `demo-abstain-data/`。
- **`rag/reflect.py`（P7 第六课，2026-08-05）**：反思式多轮检索的循环控制器
  ——把 RAG 变成 Agentic-RAG 的文件。每轮检索后 LLM 回答"证据够吗/缺什么/
  下轮搜什么"，输出 `{sufficiency, missing, follow_up, score}` 四键；循环语义
  （`qa_agentic::_retrieve_loop`）：sufficient 停轮、不充分拿 follow_up 作下轮
  检索查询（reflect 每轮评估**原始问题**，只有检索查询在换）、follow_up 空也停、
  最后一轮不调用（判定无法再触发下轮，调了白花钱）、各轮块按 chunk_id 汇池。
  与 abstain 分工：reflect 是循环中段方向盘，abstain 是作答前最终闸门；失败
  姿态同向（reflect 异常返回 sufficient 宁停不空转）。两处确认偏离：a) zh/en
  双 prompt 模板路由（复用 `_query_language`；follow_up 不强制双语——回喂
  retrieve_round 后 pipeline 内部再过 query_rewrite 的中英混出，不重复操心）；
  b) **修基准真缺陷**：基准 `return {..., "score": float(data.get("score", 0.5))}`
  在 try/except **之外**，LLM 回合法 JSON 但 score 非数值（如 "high"）时
  ValueError 炸穿整个 QA 请求——辅助判定打死主链路，违反自身兜底承诺；重建版
  输出净化全走安全路径：`_safe_score`（强转失败 0.5 + [0,1] 裁剪）、
  `_norm_sufficiency`（大小写归一 + 三值域校验，域外/非串落 sufficient 与缺键
  缺省一致）、`_as_str`（missing/follow_up 非串含显式 null 落空串，循环
  `if r["follow_up"]` 语义不变、trace 无 None）。照抄并记账：6000 字符截断对
  中文成本不对称（CJK ~1 字 1 token，反思调用略贵但内容量更大，非正确性）；
  `temperature=0/max_tokens=300` 字面量与 intent 课同口径不配置化；chat() 走
  默认 chat_model（基准 rag 层从不用 small_model）。
- **`reflect.py` 真实验收证据（2026-08-05，助手实跑，Demo 每轮 3 次 LLM 调用，
  调试期共约 3 轮）**：`scripts/demo_reflect.py` exit 0——数据与 abstain Demo
  同源，真实 BGE-M3 + embedded Qdrant + FTS5 + reranker + 真实 `qwen3.8-max`。
  通过轮：强证据英文问 sufficient/0.90 收敛停轮；ImageNet 缺口问
  insufficient/0.05、missing 指名"无 ImageNet 分类数据"、follow_up 真实喂回
  `retrieve_round` 第二轮，证据池并集 8→9（agentic 循环真实转动）；中文问
  sufficient/0.88。**过程发现两点**：a) demo 首跑忘加 `.env` 加载（demo 不走
  conftest），补 `_load_dotenv`（与 intent Demo 同款极简实现）；b) **DashScope
  `qwen3.8-max` temperature=0 跨调用不可复现**——同一输入英文强证据一轮判
  sufficient/0.90、另一轮判 partial/0.65，中文亦在 sufficient/partial 摆动
  （partial 时 missing/follow_up 均为流利中文，反证中文模板与"follow_up 同
  语言"指令生效）；Demo 断言相应改钉稳定不变量（题内强证据 != insufficient、
  缺口问题 != sufficient、非充分必带 follow_up 且中文问的 follow_up 含 CJK），
  受控 `== sufficient` 断言由 `tests/test_reflect_real.py` 用手工完备证据承担
  （3 passed）——**该经验对后续 qa_agentic/qa_stream/评测课通用：真实 LLM 判定
  类断言一律钉稳定不变量，精确断言配受控输入**。边界测试 25 passed；排除
  `*_real*` 全量 527 passed；Ruff 干净；边界/真实测试与 Demo 均在 `env -u`
  干净 shell 复跑通过。数据隔离在 `demo-reflect-data/`。
- **`rag/citation_check.py`（P7 第七课，2026-08-05，纯函数课零 LLM 消耗）**：
  第二条硬不变量"答案只以 `[chunk:<id>]` 引用已检索块"的执行层。LLM 的三种
  背叛方式各由一个纯函数看守：编造 chunk id（形式合规的内容幻觉）由
  `validate_citations` 白名单校验删除并收合法 id（保序去重）；预训练学术惯性
  `[1]` 与 `(Vaswani et al., 2017)` 由 `detect_suspicious_citations` 报告
  （`{"numeric", "author_year", "count"}`）；`strip_suspicious_citation_forms`
  剥形态并收拾标点前空格。消费方 qa_simple/qa_agentic/qa_stream 三条路径同款
  固定顺序：validate → detect → count>0 才 strip。四处确认偏离（中文答案三
  盲区 + 死代码）：a) 数字引用增全角 `【1】`（Qwen 中文答案高频形态）；
  b) 新增 CJK 作者-年份 `(张三等, 2020)`——CJK 姓名 1-4 字 + 可选"等"(=et
  al.) + 年份，括号/逗号半全角均认，归入既有 author_year 键消费方 schema
  不变，无年份的 `(见图 2)`/`（详见附录）` 不误报；c) strip 标点收拾字符类
  扩入全角 `，。；：、`；d) 不抄两处死代码——严格 hex `_CITE_RE`（基准定义后
  全仓库无人使用，validate 用宽松 `[^\]\s]+` 口径）与 `_NUM_CITE_RE` 上永不
  生效的 `(?<!chunk:)` lookbehind（纯数字要求已天然排除 `[chunk:...]`，基准
  注释还把 lookbehind 误写为 lookahead）。照抄并记账的既有边界：区间/合并型
  `[1-3]`、`[1,2]` 不识别；作者在括号外的 "Vaswani et al. (2017)" 与全角括号
  包拉丁名不识别；validate 删无效引用留双空格（仅 strip 触发时顺手收拾）。
  Ruff 处置：全角字符是本模块的业务数据（正则/测试样本），pyproject
  per-file-ignores 对本模块与其测试豁免 RUF001/2/3（比逐行 34 个 noqa 干净）。
  证据：边界测试 22 passed（六切片，含 validate→detect→strip 三段管道在混合
  污染答案上的端到端切片），全量纯逻辑 549 passed，Ruff 干净，`env -u` 干净
  shell 复跑通过。真实链路覆盖按确认并入 qa_simple 课 Demo（先例：contextual
  并入 builder 课），本课不造人为拼装答案的伪真实 Demo。
- **`rag/qa_simple.py`（P7 第八课，2026-08-05）**：单轮 QA——P7 前七课组件的
  第一次合龙。基准 phase-1 产物但现役：`scripts/ask.py` CLI 入口、评测
  ablation 的 dense-only 最小基线、citation_check 三段管道的最简单完整消费方。
  结构：dense-only `retrieve` → 空检索短路 → `format_evidence` → system+user
  prompt → `chat()` 全默认参数 → validate → detect → count>0 才 strip → 四键
  输出（answer/citations/chunks/suspicious_citations，无 trace）。`_SYSTEM` 是
  硬不变量的 **prompt 端**（让模型少犯），citation_check 是执行端（犯了也漏
  不出去），两端合起来才是完整不变量。两处确认偏离：a) 系统与用户模板 zh/en
  双语路由（复用 `_query_language`；中文版明确禁止全角引用形态 【1】/全角
  括号作者-年份——上一课刚教会执行端认这些形态，prompt 端同步收紧）；b) 无
  证据短路文案按语言路由（"(未检索到证据)"/基准英文原文）。照抄并记账：
  dense-only 检索**不升级**为 retrieve_round（设计决策——qa_simple 的存在
  意义就是最小对照组，ablation 需要该基线，完整管道是 qa_agentic 的事）。
- **`qa_simple.py` 真实验收证据（2026-08-05，助手实跑，Demo 3 次 + 真实测试
  2 次 LLM 调用）**：`scripts/demo_qa_simple.py` exit 0——数据与 abstain/
  reflect Demo 同源，真实 BGE-M3 + embedded Qdrant + 真实 `qwen3.8-max`。
  英文问 Graph-Mamba：citations=4、全部 ⊆ 检索集、strip 后 detect 复检
  count=0、答案带 [chunk:] 令牌；中文问区块链：citations=6、同纪律、中文
  prompt 路由下答案为中文；**域外行为观察（教学对照）**：问"上海明天的天气"
  检索仍返回 6 块噪声，模型诚实声明"证据不足以回答"**却仍引用了 3 个噪声块**
  ——prompt 端的"不足要明说"守住了内容，但守不住引用行为，这正是 qa_agentic
  需要 abstain 在 LLM 前砍掉调用的实证；citation_check 课确认的"真实链路覆盖
  并入 qa_simple Demo"随本 Demo 兑现。`tests/test_qa_simple_real.py` 2 passed
  （按开题确认的数据注入打 LLM 边界：手工真实形态 chunks + 真实 Qwen，断言
  只引给定 id、无可疑形态残留、中文答案中文）。边界测试 12 passed；全量纯
  逻辑 561 passed；Ruff 干净；测试与 Demo 均在 `env -u` 干净 shell 复跑通过。
  数据隔离在 `demo-qa-simple-data/`。
- **`rag/qa_agentic.py` + `observability/`（P7 第九课，2026-08-05）**：P7 总
  合龙。七 Stage 编排：history 改写(钩子)→wiki 背景(钩子, 只作 prompt 背景
  永不作证据——硬不变量"证据边界")→qa_cache 短路(钩子)→intent 分类 +
  `max_iter=min(intent档, max_inner_iters)` + `_retrieve_loop`(retrieve_round+
  reflect)→`final_chunks=池[:top_k*2]` 空则 no_chunks 短路→abstain(no_evidence
  跳过 LLM / weak 注入 hint)→evidence_select→组 prompt(Allowed citation
  tokens 白名单 + 最多 2 引用)→chat(temperature=answer, max_tokens=1024, 失败
  降级 evidence-only)→citation 三段管道→输出 + trace(intent/iters/stopped_by/
  abstain/evidence_selection/trace_id) + loop trace(latency_ms) + memory 键。
  开题前依赖核查发现 **observability 是无 try/except 保护的硬依赖**(qa_agentic
  函数内 `from ..observability import counter` 直接炸)→前置一并重建：基准自研
  165 行小包, 零 prometheus 依赖(加锁 dict 计数器/直方图 + 自渲染 Prometheus
  文本 + snapshot/reset), ADR-0020 同决策。三处确认偏离：a) observability 前置
  重建；b) prompt/文案 zh/en 路由(系统模板、用户模板含"最多使用 2 个引用"、
  no_chunks 与 chat 失败文案), weak hint 改调 abstain 课 `weak_evidence_hint
  (language)` 兑现伏笔；c) 配置新增 `rag.abstain.no_evidence_message_en`,
  拒答文案按问题语言路由(关闭 abstain 课"英文问题收中文文案"记账项)。钩子
  照抄：qa_cache/history/research_memory/wiki 全部 try/except 非致命降级,
  模块未建时 warning 诚实信号(wiki 每次 QA 一行)；conversation_id 未传时
  history/research_memory 静默跳过。
- **`qa_agentic.py` 真实验收证据（2026-08-05，助手实跑，Demo 约 7 次 + 真实
  测试约 5 次 LLM 调用）**：`scripts/demo_qa_agentic.py` exit 0——真实全链
  (真实 BGE-M3/Qdrant/FTS5/reranker + 真实 Qwen 的 intent/reflect/作答)。
  英文问: intent=factual、1 轮、abstain confident(0.990)、citations=2(prompt
  的"最多 2 引用"实证生效)、suspicious=0；中文问: intent=reasoning、
  confident(0.994)、citations=2、中文答案(主从多链结构…)；**域外"上海明天的
  天气": abstain no_evidence(0.0000)、作答 chat 调用计数为 0、citations=0、
  返回中文拒答文案——与 qa_simple 同题的 citations=3 正面对照, "无证据宁
  拒答"与"[chunk:] 引用纪律"两条硬不变量在同一链路闭环**；Prometheus 指标
  节选(qa_total 按 intent/stop 分桶、abstain_total 按 decision 分桶、latency
  直方图)现场打印。`tests/test_qa_agentic_real.py` 2 passed(数据注入
  `_retrieve_round` + 真实 intent/reflect/作答: 强证据 answered+confident+
  引用纪律；低分噪声 no_evidence_abstain + 作答零调用)。边界测试 19 passed +
  observability 8 passed；全量纯逻辑 588 passed；Ruff 干净；测试与 Demo 均在
  `env -u` 干净 shell 复跑通过。数据隔离在 `demo-qa-agentic-data/`。
- **`rag/qa_stream.py`（P7 第十课·收官，2026-08-05）**：qa_agentic 的流式
  变体——同一条管道改造成事件生成器，每阶段完成即刻 yield，前端(DeerFlow
  SSE/chat UI)增量渲染。八类事件：intent → rewrite(仅首轮，透出改写载荷,
  检索因此改走 `retrieve_round_with_rewrite`) → retrieved(每轮) → reflect →
  abstain → answer_chunk(逐 token) → done(完整结果) | error(终止)。与
  qa_agentic 的结构差异是分工不是偷工：无 trace_id/metrics/cache/wiki/
  history(前端要过程可见而非审计账本)；错误路径 yield error 事件而非降级
  响应；abstain 拒答文案也作为 answer_chunk 发出(前端渲染路径统一)。
  **流式边界定夺(llm 课记账兑现)**：`_stream_chat` 是全仓唯一流式调用，直接
  `get_client().create(stream=True)` 逐 `delta.content` yield，不下沉进
  llm.py 公共 API——主链路保持全同步，与基准同构。两处确认偏离：a) 系统/
  用户模板、no-chunks 文案、拒答文案 zh/en 路由(qa_agentic 同款先例，weak
  hint 调 `weak_evidence_hint(language)`)；b) `_stream_chat` 补
  `llm.extra_body` 透传——基准漏配：llm 课的透传只盖非流式 chat()，思考型
  模型流式默认 enable_thinking=true 会先吐 reasoning_content delta，基准只认
  delta.content，思考 token 被静默丢弃但照常计费；透传后配置可显式关闭，
  空表缺省时调用形参与基准逐键一致。照抄记账：`max_tokens=600`(与
  qa_agentic 的 1024 不一致系基准原样，流式面向前端短答案)；
  `temperature=temperatures.stream`。
- **`qa_stream.py` 真实验收证据（2026-08-05，助手实跑，Demo 约 7 次 + 真实
  测试约 4 次 LLM 调用）**：`scripts/demo_qa_stream.py` exit 0——**按用户
  要求实现 CLI 事件流渲染器**：intent/rewrite/retrieved/reflect/abstain 逐
  事件一行(rewrite 现场展示 HyDE 多查询与中英混出关键词)，answer_chunk 逐
  token 打字机效果实时打出且**完整答案不截断**，done 后打印 citations/
  suspicious/paper_ids 明细。英文问：factual、confident(0.9904)、**26 个流式
  片段**、citations=2；中文问：reasoning、reflect sufficient(0.88) 停轮、
  confident(0.9936)、**43 个流式片段**、中文全答案、citations=2；域外"上海
  明天的天气"：no_evidence(0.0000)、**流式 LLM 调用计数为 0**、中文拒答文案
  经单个 answer_chunk 流出——流式链路上双硬不变量同样闭环。
  `tests/test_qa_stream_real.py` 2 passed(数据注入+真实流式 Qwen：强证据
  answer_chunk>1 真流式+引用纪律；噪声 abstain 短路零调用)。边界测试
  17 passed；全量纯逻辑 605 passed；Ruff 干净；测试与 Demo 均在 `env -u`
  干净 shell 复跑通过。数据隔离在 `demo-qa-stream-data/`。**P7 RAG/QA 层十课
  全部完成收口**(qa_cache/history/research_memory 视需要排后，async_api 归
  网关阶段)。
- **`scripts/ask.py`（第十一课，2026-08-05）**：CLI 问答入口，闭合 CLAUDE.md
  核心三步 init_store → ingest_one → ask 的最后一环。基准考古：基准 ask.py
  停在 phase-1 只接 qa_simple——qa_agentic/qa_stream 建成后没人回来升级 CLI
  (agentic 的消费方是 DeerFlow 网关)，基准里无法从命令行体验完整 agentic
  链路。确认偏离：补 `--agentic`(qa_agentic + TRACE 摘要: intent/iters/
  stopped_by/abstain 决策与证据分/latency)与 `--stream`(qa_stream 打字机
  事件渲染, error 事件返回非零)两个互斥模式；默认模式仍 qa_simple 与基准
  同构；`--no-llm` 照抄；输出标头保持基准英文(工具层文案, 答案本身已由引擎
  按问题语言路由, 记账可后续加 --lang)。测试性小偏离: parse_args/main 接受
  argv 形参(缺省 sys.argv 行为不变)。
- **`ask.py` 真实验收证据（2026-08-05，助手实跑，首次进程级验收形态）**：
  `scripts/demo_ask.py` exit 0——组装隔离双语库落 `demo-ask-data/`，基于
  default.yaml 生成指向隔离数据的完整 config.yaml，经 `PAPER_RAG_CONFIG`
  注入后 **subprocess 运行真实 CLI 进程五次**(此前 Demo 均为进程内调用,
  本课以用户实际使用形态验收)：--no-llm 输出 4 个证据块零 LLM；默认模式
  ANSWER/CITATIONS 结构成立(citations=4)；--agentic 中文问得中文答案 +
  TRACE 摘要(reasoning/answered/confident(0.9936)/latency 59987ms)；
  --stream 事件行+流式 ANSWER+引用清单；--stream 域外"上海天气" abstain
  短路、拒答文案流出、citations=0。**新边界教训**：首跑失败——embedded
  Qdrant 对存储目录持锁, 父进程(数据准备)未释放时 CLI 子进程的 search
  **静默返回空结果**(rc 仍 0, 软降级掩盖问题)；修复为数据准备完毕后调用
  `qdrant_store.close_client()` 释放目录锁再起子进程。该边界对所有
  "embedded 模式 + 多进程"场景通用(生产走远程服务模式无此问题, embedded
  仅 demo——与既有约束一致)。边界测试 8 passed(`scripts.ask` 命名空间导入
  需 `uv run python -m pytest`, 与 init_store 脚本测试同款约束)；全量纯逻辑
  613 passed；Ruff 干净；边界测试与 Demo 均在 `env -u` 干净 shell 复跑通过。
  数据隔离在 `demo-ask-data/`, 用户可直接
  `PAPER_RAG_CONFIG=demo-ask-data/config.yaml uv run python scripts/ask.py
  "问题" --stream` 亲手体验四种模式。
- **`ask.py` 用户实跑捉缺口与修复（2026-08-05）**：用户按交付命令直跑
  `--stream`, 输出 `[rewrite] 1 queries`(恒等改写降级)、`reflect failed`、
  作答 `chat stream failed: CHAT_MODEL not set`。根因：**CLI 是独立进程入口,
  没有 conftest 帮忙加载 `.env`**——demo_ask.py 父进程加载 `.env` 后经 env
  传给子进程, 把这个缺口在验收里掩盖了(与 query_rewrite 课"助手 shell 先
  export 掩盖 conftest 缺失"同一类教训: **验收环境与用户环境的差异本身就是
  被测面**)。修复：`ask.py` 启动时 `_load_dotenv`(不覆盖已导出变量, 与
  conftest/demo 同约定)+2 个专测(加载/引号剥除/不覆盖/缺文件 no-op; 共
  10 passed, 全量 615)。实证：在 `env -u` 剥离全部 LLM 变量的干净 shell 里
  复跑用户原命令, 真实流式中文答案(主从多链/六层逻辑架构)+2 引用+exit 0。
  顺带一个 Ruff 细节：E402 允许 `sys.path.insert(...)` 表达式语句先于
  import, 但赋值语句(`_REPO_ROOT = ...`)会破坏豁免——常量移到 import 之后。
- **`scripts/ingest_one.py` + `scripts/ingest_batch.py`（第十二课，
  2026-08-05）**：由用户真实需求驱动("一个文件夹几百篇中英混合 PDF 全部
  入库")。方案讨论结论：a) **元数据补全推迟为独立后续课**——用户本地 PDF 无
  元数据, 确认方案为三级递进(PDF 首页正则抽 arXiv/DOI → 喂现有权威源;
  抽不到用疑似标题查 S2 `/paper/search`(现有代码无此方法, 需新增)/OpenAlex
  并做相似度校验, 错的元数据比空的更有害; 兜底文件名+unresolved 标记;
  中文期刊 API 覆盖率低, 更宜首页结构化解析含 Qwen 兜底), 先入库后补
  (meta.json 可重写 + --force 可重建); b) 分两趟(元数据纯网络/入库占 GPU)
  的执行策略同样归档待后续。基础设施核验: 用户 Qdrant 1.18.3 服务运行中,
  paper_chunks(1024/Cosine)/wiki_entries 已建且空, SQLite 四表已建全空,
  正式库干净; GPU RTX PRO 6000 Blackwell 97G 空闲; data/ 8.1G 全是模型缓存。
  实现: ingest_one 照抄基准(--arxiv/--pdf/--title/--force) + .env 自加载 +
  argv 形参; ingest_batch 按需求设计——平铺扫 *.pdf(大小写不敏感排序)、
  逐篇 try/except 隔离(单篇失败不中断)、断点续跑(引擎幂等 done→skipped)、
  --dry-run/--limit(试水)/--force/--report、逐篇 JSON 报告(status/chunks/
  reason/error/seconds + summary)、退出码 0 无失败/1 有失败/2 参数目录错、
  标题取文件名去扩展名、language 由解析层 auto 判定。
- **`ingest_batch` 真实验收证据（2026-08-05，助手实跑，零 LLM 消耗）**：
  `scripts/demo_ingest_batch.py` exit 0——**首次完整三步闭环的进程级实证**:
  组装真实双语试验集(英文 Graph-Mamba PDF + 中文期刊 origin PDF)与隔离配置
  (embedded Qdrant + 独立 SQLite, models_dir 指真实缓存离线命中), subprocess
  依次跑 init_store → ingest_batch --dry-run(2 篇清单零副作用) →
  ingest_batch 全量(**真实 GPU MinerU**: 英文 done 50 chunks/38.0s, 中文
  done 82 chunks/32.8s, 合计 70.7s) → 原命令重跑(skipped=2, 断点续跑实证)
  → ask --no-llm 中文问题新库检索命中 → 报告 JSON 逐篇字段齐全。**过程
  发现**: 首轮两篇全 failed "Collection paper_chunks not found"——全新库必须
  先跑 init_store 建表建 collections(逐篇隔离与失败报告恰好在此得到真实
  锻炼: rc=1、错误落报告、批不中断); demo 补 init_store 步骤后通过, 也让
  demo 恰好覆盖 CLAUDE.md 核心三步全链。单篇耗时 33-85s(重跑可复用已解析
  产物), 几百篇预估 3-10 小时, 断点续跑可分多晚。边界测试 13 passed
  (两脚本引擎全打桩); 全量纯逻辑 628 passed; Ruff 干净; 测试与 Demo 均在
  `env -u` 干净 shell 复跑通过。数据隔离在 `demo-ingest-batch-data/`。
  **用户正式批量入库待执行**(默认配置即指向其运行中的 Qdrant 服务与正式
  data/, 无需改配置; 流程 --dry-run → --limit 3 → 全量)。

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
- `6781807 docs(course): 精简学习状态并归档历史记录`
- `08401bd feat(chunk): 章节切分器切片 2 英文纯文本标题四形态与守卫`
- `4b626e9 feat(chunk): 章节切分器切片 3 标题清洗与描述性合法性`
- `8615c08 feat(chunk): 章节切分器切片 4 markdown 优先级去重与 References 尾部过滤`
- `ef7ed00 feat(chunk): 章节切分器切片 5 中文标题规则、阿拉伯点分编号与语言路由`
- `5caefcb fix(chunk): 章节切分器真实验收修复数字直贴清洗与密排编号标题识别`
- `55b4e3d feat(chunk): 文本切块器双语句子切分、CJK 回退计数与真实定位偏移`
- `39e296f feat(chunk): 上下文前缀语言路由与空值省略`（含当时的课程文档更新）
- `395309d feat(parse): MinerU 产物注入页码标记并接入标准化`
- `eb897ed feat(chunk): 切块组装器语言贯通、页码归属与偏移精确化`
- `800a531 feat(chunk): 章节完整性打分器中文关键词表与语言路由`
- `b055e8f feat(chunk): 参考文献块保留入库并打元数据标记`
- `e641019 fix(chunk): 全角点加入中文句读边界集`
- `d988ebc docs(course): 记录切块组装器完成与真实验收结论`
- `4459315 docs(course): 记录打分器课次完成并更新协作规则`
- `a972bd8 feat(chunk): 多模态切块语言路由、layout 图注页码增强与表重定型`
- `b07c363 docs(course): 记录切块层收官与嵌入课依赖顺序修正`
- `a91a30b feat(embed): BGE-M3 稠密嵌入封装与中英同空间真实验收`
- `92662d2 feat(rag): OpenAI 兼容 LLM 客户端与 Qwen extra_body 透传`（P7 第一课）
- `46be7c3 feat(rag): 查询改写与 HyDE 的中文语言路由与双语关键词`（P7 第二课，含 conftest）
- `762d353 feat(rag): 意图三档分类与配置化检索档位`（P7 第三课）
- `83b9d30 feat(rag): 确定性证据选择与 CJK bigram 词面重叠`（P7 第四课；ADR-0002 漏纳入，待补提交）
