# 学习状态

> 本文件只保留恢复课程所需的最小状态。逐课完成记录、逐提交注解与已诊断问题的细节
> 已归档到 `docs/learning/LEARNING_HISTORY.md`，恢复课程**不需要**读取归档，仅在
> 追溯历史时查阅。

## 当前定位

- 当前阶段：P4（采集、解析和切块）；当前课次：P4.10（切块模块）。
- 解析阶段已于 2026-07-31 全部完成（真实 GPU 双语 OCR 验收 + 可复现性缺口补提交，
  细节见归档）。
- 切块进度：`section_splitter.py` 已全部完成并通过真实验收（切片 0–5b，
  提交至 `5caefcb`；真实验收用 4 份真实解析产物，含 1 篇中文期刊，细节见归档）。
- **唯一下一步**：`text_chunker.py` 课次。动手前先检查基准实现的英文隐式假设
  （token/字符计数对中文的影响是重点），提出中文扩展方案并经用户确认，再按
  RED → 实现 → GREEN → Ruff 的节奏进行。
- 切块文件顺序：`section_splitter.py` ✅ → `text_chunker.py` → `contextual.py` →
  （插入 `parse/page_markers.py` 小课次）→ `builder.py` → `sanity.py` →
  `multimodal_chunker.py`。每个文件动手前先检查基准的英文隐式假设（中文扩展约束）。
- 源码基准：`/home/user_kyh/paper-rag-agent-main`（只读，不得写入）；重建目录：
  `/home/user_kyh/paper-rag-agent-rebuild`。

## 新会话恢复协议

用户可以把下面这句话作为新会话的第一条消息：

```text
请继续 /home/user_kyh/paper-rag-agent-rebuild 的后端重建课程。先完整读取
LEARNING_STATE.md 和 AGENTS.md，再检查 Git 状态；不要回退任何现有改动，
从状态文件记录的唯一下一步继续。
```

恢复顺序：

1. 工作目录为重建目录；基准仓库只作源码参考，不能写入新代码。
2. 完整读取本文件；若重建目录中存在 `AGENTS.md`，同时读取并遵守。
3. 只读执行 `git status --short` 和 `git log -5 --oneline`，以实际文件系统为准；不得
   reset、checkout、清理或覆盖任何未提交文件；`demo-*-data/` 和 `data/` 已被
   `.gitignore` 忽略，是用户保留的真实产物，不得删除。
4. 从"当前定位"的唯一下一步继续；"待处理问题"中的已知阻塞项除非用户明确要求，
   不要在课次中间插队处理。

## 切块层已确认方案（2026-08-01）

- **全局新约束（中文论文扩展）**：重建版已支持中文论文（语言元数据链路 `zh/en`），
  从切块层起**所有后续模块**都必须显式考虑中文论文，不得照抄基准的纯英文逻辑。每个
  新文件动手前先检查基准实现中的英文隐式假设（英文标题白名单、空格分词、大写比例
  启发式、English-only 正则等），提出中文扩展方案并经用户确认。
- **splitter 接口**：`split_sections(md: str, *, language: str | None = None)`。
  `language` 取领域值 `zh | en | None`（`None` = 双语规则同时启用），由上游调用方从
  `meta.json` / 解析层 `language.json` 传入；供应商值 `ch/en` 只存在于 MinerU 边界。
- **TDD 切片（全部完成）**：
  0) ✅ 包入口 `chunk/__init__.py`；
  1) ✅ markdown 标题主路径 + 无标题兜底 `Body`，签名即含 `language` 参数；
  2) ✅ 英文纯文本标题四形态 + 守卫 + 最小重叠去重；
  3) ✅ 标题清洗、合法性判定（含描述性规则）、层级计算与 first-abstract 守卫；
  4) ✅ markdown 优先级去重 + references 尾部过滤 + 英文集成用例；
  5) ✅ 中文扩展（规范白名单、编号形态、`摘要：`行内切分、2–30 字符合法性、
  `图/表/算法` 前缀黑名单、`参考文献→附录` 尾部过滤、`zh/en/None` 路由）；
  5b) ✅ 中文阿拉伯点分编号（`1.`/`1.1`/`2.3.1`，层级按段数；单位字符黑名单
  `倍/%/‰` 拦小数量词、句读一票否决拦编号列表句；用户提出后追加）。
- **真实验收（2026-08-01，`scripts/demo_section_splitter.py`）**：4 份真实解析
  产物（MinerU 中文期刊 + MinerU 英文 2 篇 + PyMuPDF 密排英文）全部断言通过。
  验收发现并修复两个缺陷：a) 中文期刊 markdown 标题的纯数字直贴形态
  （`# 1综合…`），清洗正则补 `[0-9]{1,2}(?=汉字)` 分支（限 1–2 位保护年份标题）；
  b) 真实 PyMuPDF 产物整篇无空行，切片 2 给英文编号形态加的段落边界守卫比基准
  更严导致整篇只剩 3 节，已删除该守卫与基准对齐（误报由描述性合法性兜住）；
  **中文编号形态保留边界守卫**（中文合法性判别力弱，且中文主路径是 MinerU）。
- **splitter 已记账的已知边界（暂不处理，处理时机由用户决定）**：英文
  `Appendix ` 前缀白名单会把段首 "Appendix A contains proofs." 误判为标题
  （收紧方案 B：标识符 token + 分隔符/标题式后文）；References 之后的字母编号
  附录（`A. Dataset Description.`）被尾部过滤丢弃（基准同款）；MinerU 把图注
  标成 markdown 标题（`# Fig.5Integrated…`）时按 markdown 信任保留（基准同款）；
  单位黑名单仅收 `倍/%/‰`，"10.5 万条…" 类段首行仍可能误报（万/亿 不能入黑名单，
  会误伤 "万维网技术" 类真标题）；中文密排 PyMuPDF 版面未验收（无真实样本）。
- **已确认的基准缺陷修正**：基准把整行传给 `_level_from_number`，
  `"2. Related Work"` 被误判为二级；重建版只用编号本身算层级（切片 2 测试已按修正
  行为断言）。
- **页码归属（方案 A）**：基准 `builder.py` 仅靠 `<!-- page N -->` 标记归属页码，
  MinerU 路径的 `paper.md` 无标记，导致 MinerU 论文所有 chunk `page=None`。重建版
  新增纯函数 `src/paper_rag/parse/page_markers.py::inject_page_markers(md, blocks)`：
  按 `layout.json`（MinerU content_list，块含 `type/text/page_idx`，0 基）在
  `page_idx` 跳变处用块文本前缀顺序对齐定位 md 偏移，插入 `<!-- page N -->`
  （`N = page_idx + 1`，与 PyMuPDF 一致的 1 基）；定位失败跳过、优雅降级。已用
  `demo-mineru-data/` 真实中文论文（15 页、159 块）验证 116 个非空文本块零失配。
  排期：`builder.py` 之前插入小课次（助手写 RED：正常对齐、定位失败、空 layout、
  0 基转 1 基；用户实现并接入 MinerU 标准化；真实验收用 `demo-mineru-data/`、无
  mock）。splitter 为纯函数以单元测试验收；真实链路 Demo 推迟到 builder 完成后。

## 已确认的约束

- 使用与基准仓库相同的 Python 依赖、嵌入/重排模型、Qdrant、Docker 和
  OpenAI-compatible LLM 配置；`uv` 管理环境（不用 Conda）；Python `>=3.10,<3.14`，
  `[tool.uv]` 允许 MinerU 预发布依赖。
- 按可运行的依赖顺序重建；核心 RAG 优先，随后扩展 Discovery、Wiki、反馈、主动
  Agent、交付物和 DeerFlow；不讲 DeerFlow 前端及无关上游功能。
- 所有教学文本使用中文；以运行行为、公开接口、测试和评测为等价标准，不要求源文件
  字节级一致。
- MinerU 生产默认 `.venv/bin/magic-pdf`、强制 `ocr` 模式和 CUDA GPU；模型下载到
  项目自己的 `data/index/mineru_models/`。
- 语料中英混合：纯扫描 PDF 由人工在 `meta.json` 顶层标注 `language: "zh"|"en"`；
  应用配置 `mineru.lang: auto|ch|en`，`auto` 优先人工元数据、其次 PyMuPDF 采样、
  失败回退 `ch`；领域元数据用 `zh/en`，仅在 MinerU 边界映射 `ch/en`。
- 语言判断失败不终止流程；MinerU 失败可降级 PyMuPDF；扫描件无正文记录单篇失败、
  批处理继续；空结果不得伪装成功。

## 协作规则

- （2026-08-01 更新分工）助手直接创建/修改**所有代码文件**（含 `src/paper_rag/`
  正式功能文件、测试与 `scripts/` 脚本），并自行运行测试、Ruff 与 Demo 验证；
  对话中只给出改动摘要与设计讲解，不再要求用户手抄代码。
- **所有 Git 命令由用户执行**（checkpoint 提交、分支等）；助手在每个课次收尾时
  给出建议的 `git add` 文件清单与 commit message。需要真实 GPU/外部服务的命令
  也可由用户执行，助手提供命令与预期输出。
- 用户明确要求"更新进度"时，助手维护课程文档（本文件与 `docs/learning/` 归档），
  由用户单独提交，提交不得包含业务文件。
- 每次只讲解和实现一个项目文件；测试可以作为该文件的前置验收契约。
- Git message 使用 Conventional Commits：`<type>(<scope>): <中文摘要>`。
- 提交新模块必须同时提交包入口 `__init__.py`（解析层教训：克隆即失败）；验证手段是
  `git archive HEAD | tar -x -C /tmp/<snapshot>` 后在快照里实跑聚焦测试。

## 强制验收协议

对存在依赖边界或外部副作用的功能，按顺序：**边界测试**（可 mock，只证接口设计）→
**生产实现** → **真实 Demo**（`scripts/demo_*.py`，真实服务/数据/公共 API，隔离临时
数据并清理，断言 + 非零退出码）→ **真实集成测试**（无 mock 的 `tests/test_*_real.py`，
`uv run pytest -vv -s` 单独运行；缺服务/密钥时明确失败，不能 skip 后宣称验收）→
**checkpoint**（全部通过 + Ruff 才提交）。纯函数可仅用单元测试，但应在调用它的真实
链路 Demo 中得到覆盖。SQLite 用真实临时库；Qdrant 用真实 Docker/embedded 与隔离
collection；LLM、嵌入、MinerU 用真实配置/模型/API。

## P4 固化顺序

1–6 采集（抽象契约、本地 PDF、URL、arXiv、OpenAlex、Semantic Scholar 边界）与
7 解析（PyMuPDF 兜底、MinerU 双语 GPU OCR、调度器降级）均已完成，提交清单见归档。
当前为 8 切块：section splitter ✅ → text chunker → contextual →
（page_markers 小课次）→ builder → sanity → multimodal chunker。

阶段门禁：解析门禁已于 2026-07-31 满足。临时例外（2026-07-29 用户确认）：Semantic
Scholar 缺 API key 先跳过，不算完整 checkpoint，最终后端验收前必须回补真实 Demo、
无 mock 集成测试与独立提交。

## 待处理问题

- `src/paper_rag/ingest/arxiv_source.py` 有未提交的 Task 6 元数据持久化迁移 diff
  （6 insertions / 11 deletions）；必须作为独立提交，不得被顺带纳入。另需单独规划
  `fix(ingest): 为 arXiv 真实请求增加超时`（arXiv 端点不稳定 + `arxiv` 包无显式
  timeout，诊断细节见归档）。用户已知悉，暂缓处理。
- Semantic Scholar 真实验收未完成：缺 API key；`scripts/demo_semantic_scholar_source.py`
  已创建未提交，`tests/test_semantic_scholar_source_real.py` 未创建。恢复点：取得
  API key 后先跑 Demo，再补无 mock 集成测试。
- `AGENTS.md`、`CLAUDE.md`、`scripts/demo_semantic_scholar_source.py` 为未跟踪文件，
  助手不得擅自纳入提交。
- 全量测试须用 `uv run python -m pytest`（把工作目录纳入 sys.path），否则
  `scripts.init_store` 用例导入失败。真实测试缺服务/密钥时按约定明确失败不 skip，
  需单独运行。全量 Ruff 的既有历史问题位于 `scripts/demo_qdrant_store.py`、
  `src/paper_rag/ingest/arxiv_source.py`、`src/paper_rag/store/qdrant_store.py`，
  不在当前任务范围，勿混入后续提交。

## 每次课结束必须更新

- 当前阶段和课次。
- 本次完成的目标文件及其接口/行为验证证据（详细记录写入归档，本文件只留当前态）。
- 执行的命令、通过/失败结果和原因。
- 新出现的设计疑问、待复习概念和下一个目标文件。
