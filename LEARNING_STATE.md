# 学习状态

> 本文件只保留恢复课程所需的最小状态。逐课完成记录、逐提交注解与已诊断问题的细节
> 已归档到 `docs/learning/LEARNING_HISTORY.md`，恢复课程**不需要**读取归档，仅在
> 追溯历史时查阅。

## 当前定位

- 当前阶段：P1–P6 **已全部完成并收口**；当前阶段 **P7（RAG/QA 层）进行中**，
  前三课 `rag/llm.py` ✅（`92662d2`）、`rag/query_rewrite.py` ✅（`46be7c3`，含
  `tests/conftest.py` 统一 `.env` 加载）与 `rag/intent_classifier.py` ✅
  （`762d353`）均已完成并提交；第四课 `rag/evidence_select.py` ✅（`83b9d30`）。
  **注意：`83b9d30` 漏掉了 ADR-0002**，`docs/adrs/0002-evidence-select-cjk-overlap.md`
  仍未跟踪，待用户补独立提交（建议
  `git commit -m "docs(adr): 查询侧 CJK bigram 词面匹配口径"`）。
  第五课 `rag/abstain.py` ✅（2026-08-05 真实链路验收通过，已提交 `395cf7f`，
  ADR-0002 亦已补提交 `c8d109a`）。第六课 `rag/reflect.py` ✅（2026-08-05 真实
  验收通过，已提交 `2a32494`）。第七课 `rag/citation_check.py` ✅（2026-08-05
  纯函数验收通过，已提交 `170c12d`；真实链路覆盖已随 qa_simple 课 Demo 兑现）。
  第八课 `rag/qa_simple.py` ✅（2026-08-05 真实链路验收通过，已提交 `d52f466`）。
  第九课 `rag/qa_agentic.py` ✅（2026-08-05 真实链路验收通过，含前置
  `observability/` 包，已提交 `cfe67fd` + `9175749`）。第十课
  `rag/qa_stream.py` ✅（2026-08-05 真实流式验收通过，已提交 `e976dc2`）。
  **P7 十课全部完成并全部入库，RAG/QA 层收口**。第十一课 `scripts/ask.py` ✅
  （2026-08-05 进程级真实验收通过，**功能提交待用户执行**，建议清单：
  `scripts/ask.py`、`tests/test_ask_script.py`、`scripts/demo_ask.py`，message
  `feat(cli): ask 问答入口与 agentic/流式模式`）：CLAUDE.md 核心三步
  init_store → ingest_one → ask 闭环收尾。基准停在 phase-1 只接 qa_simple，
  确认偏离补 `--agentic`（trace 摘要）与 `--stream`（打字机渲染）互斥模式，
  默认模式仍 qa_simple 与基准同构。验收首次采用**进程级形态**：demo_ask.py
  生成指向隔离数据的完整配置，经 `PAPER_RAG_CONFIG` 注入后 subprocess 运行
  真实 CLI 五次（--no-llm 4 块零 LLM / 默认 ANSWER+CITATIONS / --agentic 中文
  答案+TRACE confident(0.9936) / --stream 流式结构 / --stream 域外 abstain
  短路拒答文案流出）。**新边界教训（已归档）：embedded Qdrant 持目录锁，父
  进程准备数据后必须 `qdrant_store.close_client()` 再起 CLI 子进程，否则子
  进程 search 静默返回空**。证据：边界测试 8 passed（`scripts.ask` 导入需
  `python -m pytest`，与 init_store 同款约束）、全量纯逻辑 613 passed、Ruff
  干净、`env -u` 干净 shell 复跑通过。**用户实跑捉到一个真实缺口并已修复**：
  CLI 直跑时无人加载 `.env`（demo 父进程加载后传子进程把缺口掩盖了，与
  query_rewrite 课 conftest 教训同类），rewrite/reflect 全降级、作答直接
  error；修复为 `ask.py` 启动自加载 `.env`（不覆盖已导出变量）+2 个专测
  （共 10 passed，全量 615），并在剥离全部 LLM 变量的干净 shell 里复跑用户
  原命令实证通过（真实流式中文答案 + 2 引用 + exit 0）。用户可用
  `PAPER_RAG_CONFIG=demo-ask-data/config.yaml uv run python scripts/ask.py
  "问题" --stream` 亲手体验。第十二课 `scripts/ingest_one.py` +
  `scripts/ingest_batch.py` ✅（2026-08-05 进程级真实验收通过，**功能提交待
  用户执行**，建议清单：`scripts/ingest_one.py`、`scripts/ingest_batch.py`、
  `tests/test_ingest_one_script.py`、`tests/test_ingest_batch_script.py`、
  `scripts/demo_ingest_batch.py`，message
  `feat(cli): 单篇与批量入库入口`）：为用户"几百篇本地 PDF 文件夹入库"的
  真实需求而建。ingest_one 照抄基准 + .env 自加载；ingest_batch 新增
  (基准有对应物但未细读, 按需求独立设计)：平铺扫 *.pdf、逐篇 try/except
  隔离、断点续跑(靠引擎幂等 skipped)、--dry-run/--limit/--force、逐篇
  JSON 报告(status/chunks/耗时/错误, 缺省 <data_root>/ingest_batch_report.json)、
  退出码 0/1/2。标题暂取文件名——**元数据补全方案已确认推迟为独立后续课**
  (三级递进: PDF 首页抽 arXiv/DOI 查权威源 → 标题模糊搜索加相似度校验 →
  中文首页结构化解析含 LLM 兜底; S2 标题搜索需新增 /paper/search 方法)。
  真实验收 `demo_ingest_batch.py`：**首次完整三步闭环进程级实证**——隔离库
  init_store → ingest_batch 真实双语 PDF(GPU MinerU: 英文 50 chunks/38s、
  中文期刊 82 chunks/33s) → 重跑 skipped=2 幂等续传 → ask --no-llm 新库
  检索命中。**过程发现**: 全新库必须先 init_store, 否则 Qdrant collection
  不存在整批失败(demo 已补该步; 用户正式库已核验就绪: Qdrant 1.18.3 服务
  运行中、paper_chunks/wiki_entries 已建、SQLite 四表已建全空)。证据: 边界
  测试 13 passed、全量纯逻辑 628 passed、Ruff 干净、`env -u` 干净 shell
  复跑通过。**功能已提交**：`cb4f414`(ask CLI)与 `dd40519`(入库 CLI)均已
  入库。**用户已完成 --limit 3 真实试水**(2026-08-05, 正式库): 3 篇中文 PDF
  全部 done/99.8s——5G车联网(79 chunks, mineru+complete)、6G移动网络
  (73 chunks, mineru+minimal)、一篇征文通知(7 chunks, **mineru+broken**——
  非论文文档被质量打分如实标记, 行为正确; 提示用户文件夹可能混有非论文
  PDF, 全量后可按 parsed_with 清查)。**下一步: 用户放全量批跑**(断点续跑
  可分多晚); 全量完成后下一课候选: 元数据补全 / 评测层 / 入库质量清单。
- 解析阶段已于 2026-07-31 全部完成（真实 GPU 双语 OCR 验收 + 可复现性缺口补提交，
  细节见归档）。
- 切块进度（**7 个文件全部完成**）：`section_splitter.py` ✅（`5caefcb`）；
  `text_chunker.py` ✅（`55b4e3d`）；`contextual.py` ✅（`39e296f`）；
  `parse/page_markers.py` ✅（`395309d`）；`builder.py` ✅（`eb897ed`）；
  `sanity.py` ✅（含决策点落地 `800a531`/`b055e8f`/`e641019`）；
  `multimodal_chunker.py` ✅（2026-08-01，含 builder 多模态循环/layout 增强/
  vision 钩子接回，`a972bd8`，本课首次按新验收流程由用户实跑测试与 Demo 并
  与预期对比通过，细节见归档）。
- **P6 检索层：七文件全部完成，已收口**（2026-08-04 — 08-05，逐课细节见归档
  `docs/learning/LEARNING_HISTORY.md` 的"P6 检索层逐课细节"）：
  `dense.py` ✅ `45c86c0` → `fts5.py` ✅ `aeec63d`（ADR-0001 accepted）→
  `sparse_bm25.py` ✅ `e5ce271` → `hybrid.py` ✅ `3ecc6c8` →
  `rerank.py` ✅ `4140338` → `format.py` + `pipeline.py` ✅ `11d80d9`。
  全量 375 passed，Ruff 干净，每课均由用户实跑测试 + 真实 Demo 验收通过。
  当前可用的检索链路：`retrieve_round(query, paper_ids, top_k)` 走完
  rewrite（P7 前恒等回退）→ 多查询池化 hybrid → cross-encoder 精排 →
  论文多样化，出口块带 `score_rerank`；`format_evidence` 渲染带
  `[chunk:<id>]` 令牌的证据文本。
- **P6 遗留记账边界**（均已确认暂不处理）：bigram 索引无 unigram、fts5 原地
  UPDATE 不触发行数自愈；`_diversify_by_paper` 补位场景下单篇可合法超限额。
- **P7 RAG/QA 层（进行中，2026-08-05 开题）**：依赖核对结论——`llm.py` 是 rag 层
  依赖图的根（`query_rewrite` 顶层 `from .llm import chat`），故第一课由原定
  `query_rewrite` 改为 `llm.py`；修正后顺序：**llm ✅ → query_rewrite ✅ →
  intent_classifier ✅ → evidence_select ✅ → abstain ✅ → reflect ✅ →
  citation_check ✅ → qa_simple ✅ → qa_agentic ✅ → qa_stream ✅——P7 十课
  全部完成收口**（qa_cache/history/research_memory 视需要排后；async_api 归入
  网关阶段）。
  - `rag/llm.py` ✅ `92662d2`（2026-08-05 真实验收通过）：
    模块级单例 + 同步非流式 `chat()`，与基准逐参一致；**一处确认偏离
    `llm.extra_body` 透传**（Qwen 思考型模型非流式必须 `enable_thinking: false`
    否则 400；空表缺省时调用形参与基准逐键一致，有专测钉死）。同课新建
    `rag/__init__.py` 与 `.env.example`。证据：边界测试 13 passed、全量纯逻辑
    380 passed、Ruff 干净、`scripts/demo_llm_chat.py` exit 0（真实
    `qwen3.8-max`/`qwen3.7-flash`）、`tests/test_llm_real.py` 2 passed。
  - **流式与异步边界已定**（用户提问后核实基准）：基准只有 `qa_stream::_stream_chat`
    一处流式（为 DeerFlow 前端 SSE），其余全非流式；异步是"全同步引擎 + 网关边界
    `anyio.to_thread.run_sync` 包装"。**结论**：非流式为主链路，流式推迟到
    `qa_stream` 课再定，异步推迟到网关阶段独立小课，不提前引入。
  - `rag/query_rewrite.py` ✅（2026-08-05 真实验收通过，已提交 `46be7c3`）：
    zh/en 双 prompt 模板路由（`_query_language` 按 CJK 码位占比判定）、zh 的
    keywords **中英双语混出**以跨越 BM25 词面断层（真实模型已实证混出中英术语）、
    中文"原始/最早的 X 论文"别名正则（`_original_aliases` 收敛三条正则）、修
    `_aliases_for_title` 的 CJK 邻接词边界缺陷（基准正则在"基于GNN的图神经网络
    建模"返回 `[]`，改显式 lookaround 后返回 `['GNN']`，英文侧边界不放宽）。
    pipeline 的 `identity rewrite` warning 已实测消失；P6 的过渡测试
    `test_identity_fallback_when_query_rewrite_missing` 按预期翻转为
    `test_uses_real_query_rewrite_by_default` + 导入失败降级专测。证据：边界测试
    35 passed、全量纯逻辑 426 passed、Ruff 干净、`scripts/demo_query_rewrite.py`
    exit 0、`tests/test_query_rewrite_real.py` 3 passed。
    **已提交 `46be7c3`**（7 files / +1006：query_rewrite、三个测试、conftest、
    demo，含 `test_llm_real.py` 删重复加载器）。
  - **`tests/conftest.py`（本课中途补的横切修复）**：用户直跑真实测试报"配置缺失"，
    根因是 `test_query_rewrite_real.py` 漏抄 `.env` 加载（助手自查时先 `set -a;
    . ./.env` 掩盖了缺陷）。改为 conftest 统一 `load_dotenv(override=False)`，所有
    真实测试自动受益、不必各自复制。**教训已归档：真实验收命令交付前必须在
    `env -u VAR ...` 剥离变量的干净 shell 里复跑一遍。**
  - `rag/intent_classifier.py` ✅（2026-08-05 真实验收通过，已提交 `762d353`）：
    一次 LLM 调用把问题判为 factual / reasoning / explore 三档，输出
    `{intent, top_k, max_iter, rrf_k}` 驱动后续检索力度（基准 `qa_agentic`/
    `qa_stream` 均消费它）。**永不抛异常**——LLM 挂、JSON 坏、未知 intent 名统统
    落 reasoning 中间档。三处确认偏离：a) 基准硬编码的 `_DEFAULTS` 三档参数挪进
    配置 `rag.intent`（CLAUDE.md 要求可调项进 YAML），新增 `enabled` 逃生门；
    b) zh/en 双模板路由，复用 `query_rewrite._query_language`（不重复实现）；
    c) 新增本地信号词启发式，LLM 不可用时按"区别/对比"→reasoning、
    "有哪些/进展/综述"→explore、"是什么/定义"→factual 判定，取代基准的一律落
    中间档，同时给边界测试一个不依赖 LLM 的确定性断言面。证据：边界测试
    37 passed、全量纯逻辑 463 passed、Ruff 干净、`scripts/demo_intent_classifier.py`
    exit 0（真实 Qwen 中英 **6/6 全判对**）、`tests/test_intent_classifier_real.py`
    8 passed（均在 `env -u` 干净 shell 复跑）。
  - `rag/evidence_select.py` ✅（2026-08-05 真实链路验收通过，已提交 `83b9d30`；
    **ADR-0002 未随该提交纳入，待用户补提交**）：从检索宽窗(5/10/15 块)确定性挑出 ≤4 块可引用证据(单篇 ≤2)，
    纯打分排序不调 LLM——模型分占大头、词面重叠裁平局、章节提示微加分、原始
    排名兜底；trace 逐候选记账四项得分。两处中文扩展：a) **词面重叠 token 化
    从 `[a-z0-9]+` 改为拉丁词 + CJK bigram 并集**（基准对中文问题 overlap 恒 0，
    平局裁决层整体失明；bigram 口径与 ADR-0001 一致）；b) `_SECTION_HINTS` 增
    中文条目（摘要/引言/方法/实验/结果/结论等）。已记账边界：单字 CJK 问题与
    文本 bigram 无交集、overlap=0（与 P6 "bigram 无 unigram" 同口径，由稠密分
    兜底）。权重与签名默认照抄基准，配置化推迟到 qa_agentic 课。证据：边界
    测试 18 passed、全量纯逻辑 481 passed、Ruff 干净、
    `scripts/demo_evidence_select.py` exit 0（真实检索链路：英文 overlap 0.750、
    中文 overlap 0.650 > 0 修复实证、单篇限额、两遍逐项一致）。
    决策已成文 `docs/adrs/0002-evidence-select-cjk-overlap.md`（**待补提交**，
    建议独立 `docs(adr)` 提交，勿夹带 `arxiv_source.py` 遗留 diff）。
  - `rag/abstain.py` ✅（2026-08-05 真实链路验收通过，**功能提交待用户执行**，
    建议清单：`src/paper_rag/rag/abstain.py`、`tests/test_abstain.py`、
    `scripts/demo_abstain.py`，message
    `feat(rag): 三路证据充分性裁决与 fail open 信号分级`）：纯函数无 LLM，
    在 LLM 调用前按 top-`min_chunks` 归一化证据分裁四态（no_chunks /
    no_evidence 跳过 LLM / weak_evidence 注入提示 / confident）。字段优先级
    rerank > dense > score > bm25 > rrf、首个可用字段服务全列表；低质字段
    （BM25/RRF）与字段全缺一律 **fail open** 放行并经 `signal_quality` 透出
    降级态。配置 `rag.abstain` 早已就位，本课零配置改动。两处确认偏离：
    a) `weak_evidence_hint(language)` zh/en 路由 + 中文 hint 文案；
    b) `no_chunks` 分支补 `signal_quality="no_chunks"` 拉平四态 schema
    （基准漏键）。已记账：BM25 sigmoid center=8 系英文校准但不影响裁决；
    阈值 0.21/0.48 未按中英混合语料重校（评测阶段再定）；
    `no_evidence_message` 英文侧路由推迟到 qa_agentic 课。证据：边界测试
    31 passed、排除 `*_real*` 全量 502 passed、Ruff 干净、
    `scripts/demo_abstain.py` exit 0（真实检索链：域内中英 0.99 → confident，
    域外"上海天气"8 块噪声证据分 0.0000 → no_evidence，剥高质字段 fail open
    实证），测试与 Demo 均在 `env -u` 干净 shell 复跑通过。
  - `rag/reflect.py` ✅（2026-08-05 真实验收通过，**功能提交待用户执行**，
    建议清单：`src/paper_rag/rag/reflect.py`、`tests/test_reflect.py`、
    `tests/test_reflect_real.py`、`scripts/demo_reflect.py`，message
    `feat(rag): 反思式循环控制器与输出净化`）：每轮检索后 LLM 判
    sufficiency 三态 + follow_up 驱动下一轮（消费方 `qa_agentic::_retrieve_loop`
    /`qa_stream`；最后一轮不调用；异常兜底 sufficient 宁停不空转）。两处确认
    偏离：a) zh/en 双 prompt 模板路由（复用 `_query_language`；follow_up 不强制
    双语——回喂 retrieve_round 后 pipeline 内部再过 query_rewrite）；b) **修基准
    缺陷**——基准 `float(data["score"])` 在 try 外，LLM 回非数值 score 会炸穿
    QA 请求；重建版输出净化全走安全路径（score 强转失败落 0.5 + [0,1] 裁剪、
    sufficiency 大小写归一 + 三值域校验域外落 sufficient、missing/follow_up 非
    串强转空串）。**新经验（写给后续 qa 课）**：DashScope `qwen3.8-max`
    temperature=0 **跨调用不可复现**，临界证据判定在 sufficient/partial 相邻档
    摆动——真实 Demo 断言只钉稳定不变量（题内强证据 != insufficient、缺口问题
    != sufficient、非充分必带 follow_up），受控 `== sufficient` 断言放
    `test_reflect_real.py` 用手工完备证据承担。证据：边界测试 25 passed、排除
    `*_real*` 全量 527 passed、Ruff 干净、`scripts/demo_reflect.py` exit 0
    （真实检索链 + 真实 Qwen 3 次调用：强证据 sufficient 0.90 收敛、ImageNet
    缺口 insufficient 0.05 且 follow_up 真实驱动第二轮证据池 8→9、中文模板
    sufficient 0.88；另一轮实测中文 partial 时 missing/follow_up 均流利中文），
    `tests/test_reflect_real.py` 3 passed；测试与 Demo 均在 `env -u` 干净 shell
    复跑通过。
  - `rag/citation_check.py` ✅（2026-08-05 纯函数验收通过，**功能提交待用户
    执行**，建议清单：`src/paper_rag/rag/citation_check.py`、
    `tests/test_citation_check.py`、`pyproject.toml`（per-file-ignores 两行：
    全角引用形态是本模块的业务数据，RUF001/2/3 按文件豁免），message
    `feat(rag): 引用校验与中文引用形态检测`）：硬不变量 `[chunk:<id>]` 的
    执行层，三纯函数按 validate（删编造 id、valid 保序去重）→ detect（数字/
    作者-年份可疑形态报告）→ 有可疑才 strip 的固定顺序被 qa 三条路径消费。
    四处确认偏离：a) 数字引用增全角 `【1】`；b) 新增 CJK 作者-年份
    `(张三等, 2020)`（姓名 1-4 字 + 可选"等" + 年份，括号/逗号半全角均认，
    归入既有 author_year 键）；c) strip 标点收拾扩入全角 `，。；：、`；
    d) 不抄基准死常量 `_CITE_RE`（定义后全仓库未用）与永不生效的
    `(?<!chunk:)` lookbehind。照抄记账：`[1-3]`/`[1,2]` 区间型、作者在括号外
    的 "Vaswani et al. (2017)"、全角括号包拉丁名均不识别（基准同款口径）；
    validate 删无效引用留双空格仅在 strip 触发时被收拾。证据：边界测试
    22 passed（含三段管道端到端切片）、全量纯逻辑 549 passed、Ruff 干净、
    `env -u` 干净 shell 复跑通过。真实链路覆盖并入 qa_simple 课 Demo
    （先例：contextual 并入 builder）。
  - `rag/qa_simple.py` ✅（2026-08-05 真实链路验收通过，**功能提交待用户
    执行**，建议清单：`src/paper_rag/rag/qa_simple.py`、
    `tests/test_qa_simple.py`、`tests/test_qa_simple_real.py`、
    `scripts/demo_qa_simple.py`，message
    `feat(rag): 单轮 QA 与双语引用纪律合龙`）：dense-only 检索 → zh/en 双语
    prompt → chat 全默认参 → citation_check 三段管道 → 四键输出（无 trace）。
    现役价值：`scripts/ask.py` CLI 入口、评测 ablation 最小基线、citation_check
    的第一个完整消费方。两处确认偏离：a) `_SYSTEM`/user 模板 zh/en 路由（中文
    版明确禁止全角引用形态，与 citation_check 中文扩展同步收紧）；b) 无证据
    短路文案按语言路由（"(未检索到证据)"）。照抄记账：dense-only 不升级
    retrieve_round（最小对照组的存在意义）。证据：边界测试 12 passed、全量
    纯逻辑 561 passed、Ruff 干净、`scripts/demo_qa_simple.py` exit 0（真实
    链路硬不变量端到端：英文 citations=4/suspicious=0、中文 citations=6/
    suspicious=0 且中文答案、域外行为观察——模型声明证据不足**但仍引用了
    3 个噪声块**，正是 qa_agentic 需要 abstain 闸门的实证对照）、
    `tests/test_qa_simple_real.py` 2 passed（数据注入打 LLM 边界）；测试与
    Demo 均在 `env -u` 干净 shell 复跑通过。citation_check 真实链路覆盖随本
    Demo 兑现。
  - `rag/qa_agentic.py` + `observability/` ✅（2026-08-05 真实链路验收通过，
    **功能提交待用户执行，建议拆两笔**：
    ① `git add src/paper_rag/observability tests/test_observability.py` +
    `feat(observability): 进程内指标与 trace id`；
    ② `git add src/paper_rag/rag/qa_agentic.py tests/test_qa_agentic.py
    tests/test_qa_agentic_real.py scripts/demo_qa_agentic.py
    src/paper_rag/config.py config/default.yaml` +
    `feat(rag): agentic QA 总合龙与双语拒答路由`）：七 Stage 编排(history 改写→
    wiki 背景→qa_cache 短路→intent+检索循环→no_chunks 短路→abstain→
    evidence_select+LLM+引用管道)，trace 全程账本 + loop trace + latency。
    三处确认偏离：a) 前置重建 `observability/`(counter/histogram/new_trace_id，
    零 prometheus 依赖自渲染文本格式——qa_agentic 对它的函数内 import **无**
    try/except 保护，属硬依赖非钩子)；b) 系统/用户 prompt 与 no_chunks/chat
    失败文案 zh/en 路由，weak hint 改调 abstain 课的
    `weak_evidence_hint(language)`(伏笔兑现)；c) 配置新增
    `rag.abstain.no_evidence_message_en` 按语言路由拒答文案(关闭 abstain 课
    记账项)。钩子照抄：qa_cache/history/research_memory/wiki 均 try/except
    降级(warning 诚实信号)。证据：边界测试 19+8 passed、全量纯逻辑
    588 passed、Ruff 干净、`scripts/demo_qa_agentic.py` exit 0(真实全链：英文
    factual/abstain 0.990/citations=2；中文 reasoning/0.994/citations=2 中文
    答案；**域外"上海天气" abstain 0.0000 短路、作答 LLM 调用计数为 0、
    citations 从 qa_simple 的 3 归零**——两条硬不变量在同一链路上闭环的实证；
    Prometheus 指标节选打印)、`tests/test_qa_agentic_real.py` 2 passed(数据
    注入+真实 intent/reflect/作答)；测试与 Demo 均在 `env -u` 干净 shell
    复跑通过。
  - `rag/qa_stream.py` ✅（2026-08-05 真实流式验收通过，**已提交 `e976dc2`**，
    4 files：qa_stream、两个测试、demo）：八类事件生成器
    (intent/rewrite/retrieved/reflect/abstain/answer_chunk/done/error)，
    检索走 `retrieve_round_with_rewrite` 透出改写载荷，错误路径 yield error，
    abstain 拒答文案也走 answer_chunk(前端渲染统一)。**流式边界定夺(llm 课
    记账兑现)**：流式只在 `_stream_chat` 一处，不下沉 llm.py 公共 API。两处
    确认偏离：a) prompt/文案 zh/en 路由(qa_agentic 同款 + weak_evidence_hint
    伏笔)；b) `_stream_chat` 补 `llm.extra_body` 透传(基准漏配——思考型模型
    流式默认吐 reasoning_content，基准只认 delta.content 静默烧钱；空表缺省
    逐参一致)。照抄记账：max_tokens=600、无 trace/metrics/钩子(轻装分工)。
    证据：边界测试 17 passed、全量纯逻辑 605 passed、Ruff 干净、
    `scripts/demo_qa_stream.py` exit 0(**CLI 事件流渲染器按用户要求逐 token
    打字机效果 + 完整答案不截断**：英文 26 个流式片段/citations=2、中文 43 个
    片段/中文全答案、域外 abstain 短路流式 LLM 零调用且拒答文案经
    answer_chunk 流出；rewrite 事件现场展示 HyDE 多查询)、
    `tests/test_qa_stream_real.py` 2 passed(真实流式)；测试与 Demo 均在
    `env -u` 干净 shell 复跑通过。
  - **功能提交已由用户执行**：`92662d2`（8 files / +550，含 `rag/__init__.py`、
    `rag/llm.py`、`config.py`、`config/default.yaml`、两个测试、demo、
    `.env.example`）。`arxiv_source.py` 遗留 diff 与三个未跟踪文件仍留在工作区，
    未被顺带纳入。
- **P5 进度（两课全部完成，P5 收口）**：`embed/bge_m3.py` ✅（`a91a30b`；环境
  同步命令 `uv sync --extra dev --extra embed --extra ingest --extra mineru`，
  **四个 extra 必须齐**，漏 dev 会卸掉 pre-commit/fastapi；BGE-M3 已缓存在
  项目本地 `data/index/models`，4.3G 离线命中）；
  `store/ingest_pipeline.py` ✅（2026-08-02 验收通过，提交待用户执行。与基准
  同构：状态机/_step 记录/三级去重/force/元数据卡片/打分拼 parsed_with/Qdrant
  替换语义。四处确认偏离：a) 语言贯通——builder `_read_language` 提升公开
  `read_language`，同一语言值喂 grade_sections 与元数据卡片；b) 卡片模板按
  语言路由（`_CARD_LABELS` 双语文案表，zh 用 论文元数据记录。/标题:/作者:…，
  缩写词逻辑保持基准中文优雅空集）；c) **发现并修复基准死代码缺陷**——
  "chunks 为空则 failed" 守卫在插元数据卡片之后永不触发，重建版前移到插卡前；
  d) wiki 钩子接回（非致命 warning，与 vision 同策）。边界测试 11 个全下游
  打桩；全量 296 passed；用户实跑 `scripts/demo_ingest_pipeline.py` **端到端**
  exit 0：真实 Graph-Mamba PDF → MinerU GPU → 50 chunks（含卡片，别名 GM）→
  BGE-M3 → embedded Qdrant，parsed_with=mineru+complete、四步 ingest_runs 全
  ok、真实问题检索 top-1 命中本论文 Introduction(0.741)、重复 ingest skipped、
  force 重建点数 50 不变。运行数据隔离在 `demo-ingest-pipeline-data/`
  （SQLite WAL + embedded Qdrant，双库落地已现场核验：SQLite 无向量列存内容
  溯源，Qdrant 存 1024 维向量+payload 拷贝）。服务器另有 `paper-rag-qdrant`
  服务容器(6333)长跑，dashboard 经 SSH/VS Code 端口转发可视化，embedded 与
  服务模式数据互不相通，已给用户拷贝浏览脚本）。
- 源码基准：`/home/user_kyh/paper-rag-agent-main`（只读，不得写入）；重建目录：
  `/home/user_kyh/paper-rag-agent-rebuild`。

## 新会话恢复协议

用户可以把下面这句话作为新会话的第一条消息：

```text
请继续 /home/user_kyh/paper-rag-agent-rebuild 的后端重建课程。先完整读取
LEARNING_STATE.md 和 AGENTS.md，再只读检查 Git 状态；不要回退任何现有改动，
从状态文件记录的唯一下一步继续。分工按状态文件"协作规则"：你写代码并自跑
开发自查，课程文档由你提交，功能文件由我提交；每课先讲为什么再讲怎么实现，
最终验收命令和预期效果交给我实跑对比。
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
  （`N = page_idx + 1`，与 PyMuPDF 一致的 1 基）；定位失败跳过、优雅降级。已于
  2026-08-01 实现并提交（`395309d`，接入 `_normalize_into`：布局选择提前、
  content_list 形态先注标再写 `paper.md`）。实现补充：锚定块最短长度双档（含
  CJK ≥2 字符、纯 ASCII ≥4 字符，页脚页码不做锚点）；标记插在所在行行首。真实
  验收：3 份 MinerU 产物注标 14/15、14/15、17/17 页，剥标后与存量产物逐字节
  一致。**已记账边界**：纯图表页（content_list 只有空文本块）无锚可用不注标，
  该页图表多模态 chunk 会继承前一页页码（±1 页误差），builder 课可评估用图块
  自身 `page_idx` 兜底。
- **块检查观察（2026-08-01，三条决策点已全部关闭）**：a) 参考文献节块——
  sanity 课确认**保留入库并打标** `metadata["is_references"]=True`（普通块
  不带该键，schema 与基准逐键一致；四篇真实论文的参考文献块占比 20%–37%），
  检索课拿真实评测数据再决定降权/过滤（提交 `b055e8f`）；b) 全角点 `．`
  （U+FF0E）已加入 zh 句读边界集（一行修复 + 回归测试，提交 `e641019`；
  真实效果：中文期刊参考文献从等分硬切变为按条目边界切）；c) 页码标记保留在
  chunk 文本里——builder 课已按"保留标记"实现（基准同款）。
- **chunker 接口与已确认偏离（2026-08-01，`text_chunker.py` 已实现）**：
  `chunk_text(body: str, *, language: str | None = None) -> list[TextChunk]`，配置仍走
  `chunk.text`（target/overlap/encoding，无新增项）。不变量强化为
  `body[char_start:char_end] == text`（builder 溯源可直接依赖）。相对基准四处偏离：
  a) 偏移用 `body.find` 真实定位（基准 cursor 算术在 4+ 连续换行时漂移、末块
  char_end 多 2）；b) 无 tiktoken 回退按 CJK 码位逐字计 1 + 其余 len//4（基准
  len//4 低估中文约 4–7 倍）；c) 超长段落先句子切分再贪心打包，zh 用
  `。！？；…`+收尾引号、en 用 `[.!?]`+后随空白（小数不切）、None 取并集，
  无句读段按 token 等分硬切保证上界（基准从不切超长段）；d) overlap 携带尾段加
  防重守卫（尾段 token*2 > target 时放弃携带；overlap_tokens 仍只作布尔开关，
  与基准一致）。
- **chunker 已记账的已知边界（暂不处理，处理时机由用户决定）**：zh 路由下内嵌的
  超长纯英文段无中文句读可用、落到硬切（改 zh∪en 并集即一行修复）；硬切按字符
  等分，BPE 密度不均可小幅超 target（真实验收用 1.2×target 上界看护，实测最大
  546/500，为中文期刊无句读表格区）。
- **contextual 已确认方案与实现（2026-08-01）**：
  `with_context(text: str, *, title: str, section: str, language: str | None = None)`。
  `context_text` 是 BGE-M3 的稠密嵌入输入（裸 `text` 走 BM25），前缀直接塑造
  向量。zh 路由到新配置键 `chunk.context_prefix_zh`
  （`[标题: {title}] [章节: {section}]\n`，default.yaml + `_Chunk` 模型各 +1 行），
  en/None 用基准英文模板（未知语言不猜）；空 title/section 按渲染后 `[...: ]`
  形态整段移除（半/全角冒号均识别，基准会给每个 chunk 嵌入 `[Title: ]` 死架子），
  都空时直接返回原文；自定义模板不用括号形态时退化为基准的空串填入。值含花括号
  安全（format 语义）。纯函数仅单元测试验收，真实链路覆盖并入 builder 课 Demo。
- **multimodal 已确认方案与实现（2026-08-01，已提交为 `a972bd8`）**：三个抽取器
  `extract_figures/tables/formulas(body, *, language=None)`，识别正则与守卫和
  基准逐字一致。三处偏离：a) 嵌入前缀语言路由（zh 用 图:/表:/公式:/上下文:/
  路径:，en/None 用基准英文；`compose_*_text` 模板助手公开导出）；b) 表格块
  span 与 strip 后的 raw 对齐，不变量 `body[char_start:char_end] == raw` 三类
  块都成立；c) `MMChunk` 增带 alt/context 原料字段。builder 侧 layout 增强
  （基准无）：`_load_layout_assets` 按图片 basename 配对 layout.json 的
  image/table 块——图块页码优先用自身 `page_idx+1`（纯图表页从此有页码，
  关闭记账优化点）、`img_caption` 注入嵌入文本（真实产物 alt 全空，图注是图的
  语义本体）、配对到 table 块的图片**重定型为 `modality="table"`** 用
  `table_caption`；`chunk_id` 命名空间保持抽取器 kind 防撞 id；layout 缺失/
  损坏/异形优雅降级回基准。vision enrich 钩子按基准同款接回（try/except，
  vision 模块未建前每次 build 打一行 warning，属预期诚实信号）。真实验收
  发现：OCR 模式下 MinerU 把表格全部渲染成图片，三篇论文合计 23 张表
  （5/7/11）靠重定型找回正确身份；PyMuPDF 密排版面三类元素实测零召回
  （基准 docstring 承认的 reduced recall，如实记账）。
- **builder 已确认方案与实现（2026-08-01，已提交为 `eb897ed`）**：
  `build_chunks(paper_id: str, parsed_dir: Path, *, title: str)` 签名与基准一致，
  `language` 不进公开签名——builder 从 `parsed_dir/language.json` 读
  `document_language`（缺失/损坏/域外值降级 `None` 不终止），是把语言贯通到
  `split_sections`/`chunk_text`/`with_context` 的**全链唯一枢纽**。偏移精确化：
  `md.index(body, sec.start)` 求 body 真实起点，全局不变量
  `md[char_start:char_end] == chunk["text"]`（基准在节头多空行时整体漂移）。
  页码归属 `_page_for_offset` 与基准同款（回扫最近 `<!-- page N -->`），标记
  保留在 chunk 文本里。chunk 字典 schema 与基准完全一致（含
  `metadata.section_level/chunk_ordinal`、`neighbors=[]`）。**已确认分片**：
  多模态循环（figure/table/formula）与 vision enrich 钩子推迟到
  `multimodal_chunker` 课一并接回（当前文本主路径不需要空 warning 的钩子）。
  真实验收 4 份产物全部通过：中文期刊 16 节/61 块、Graph-Mamba MinerU 25/39、
  LocAgent 26/45、Graph-Mamba PyMuPDF 9/40；MinerU 论文 chunk 从基准的
  `page=None` 变为全员有页码且单调不减，偏移逐字节回切，zh 论文 context_text
  全部中文模板；产物在 `demo-builder-data/parsed/<id>--{mineru|pymupdf}/
  chunks.json`（同论文双解析器用来源后缀防覆盖）。**已记账边界**：纯图表页的
  图块页码待 multimodal 课用图块自身 `page_idx` 兜底；页码标记被空行包围时
  自成段落，极小 target 下会独立成块（真实 500 目标并入邻块，无影响）。

## 已确认的约束

- **知识库目标规模 20000 篇（≈10^6 chunks，2026-08-04 用户确认）**：后续每课按
  此规模审视——rank_bm25 仅作小规模后备 + 评测对照（超限护栏
  `retrieve.bm25_max_chunks`）；FTS5 增量 `sync_paper` 须接入 ingest（ADR-0001
  规模修订）；生产 Qdrant 走远程服务模式，embedded 仅 demo。

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
  正式功能文件、测试与 `scripts/` 脚本），并在开发过程中自行运行测试与 Ruff；
  对话中给出改动摘要与设计讲解，不要求用户手抄代码。
- **Git 分工（2026-08-01 二次更新）**：`LEARNING_STATE.md` 与
  `docs/learning/LEARNING_HISTORY.md` 两个课程文档由**助手自己提交**
  （`docs(course)` 独立提交，不得夹带业务文件）；其余功能文件仍由**用户提交**，
  助手在课次收尾给出建议的 `git add` 清单与 commit message。需要真实 GPU/外部
  服务的命令也可由用户执行，助手提供命令与预期输出。
- **教学顺序（2026-08-01 新增）**：每个功能模块开始前，先讲**为什么要写这个
  文件、有什么作用**，再讲**功能怎么实现**（含基准英文隐式假设与中文扩展方案，
  经用户确认），然后才动手编写代码。
- **验收方式（2026-08-01 新增）**：代码写完后，**测试文件与真实验收脚本由用户
  亲自执行**；助手先给出验收命令与预期效果（输出的样子），用户实跑后与预期
  对比。助手开发过程中的 RED/GREEN 自查仍自行运行。
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

1–6 采集（抽象契约、本地 PDF、URL、arXiv、OpenAlex、Semantic Scholar 边界）、
7 解析（PyMuPDF 兜底、MinerU 双语 GPU OCR、调度器降级）与 8 切块（section
splitter → text chunker → contextual → page_markers → builder → sanity →
multimodal chunker，7 文件全部 ✅）均已完成，提交清单见归档。P5（嵌入与入库）
已完成：`embed/bge_m3.py` ✅ → `store/ingest_pipeline.py` ✅（`5201259`）。
P6 检索层已收口：dense ✅ → fts5 ✅ → sparse_bm25 ✅ → hybrid(RRF) ✅ →
rerank ✅ → format/pipeline ✅。P7（RAG/QA 层）已开题，**顺序已按基准依赖核对
修正**（原预排的 query_rewrite → llm 与实际依赖相反）：llm ✅ → query_rewrite ✅ →
intent_classifier ✅ → evidence_select ✅ → abstain ✅ → reflect ✅ →
citation_check ✅ → qa_simple ✅ → qa_agentic → qa_stream（qa_cache/history/async_api/research_memory
视需要排后；async_api 归入网关阶段）。

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
- P7 起真实验收需 `.env` 提供 `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`CHAT_MODEL`
  （用户用通义千问 DashScope 兼容模式，2026-08-05 实测 `qwen3.8-max` +
  `qwen3.7-flash`）。`.env` 被 gitignore；模板见已跟踪的 `.env.example`。
  Qwen 思考型模型非流式调用需在配置加 `llm.extra_body: {enable_thinking: false}`。
- `vision/` 模块推迟到核心 RAG 链（嵌入→存储→检索→QA）跑通之后（2026-08-01
  与用户确认）：基准默认 `vision.enabled: false`，builder 钩子已接回、行为与
  基准默认态等价，仅多一行 warning 诚实信号。届时真实验收需要
  `VISION_BASE_URL`/`VISION_API_KEY` 或本地视觉模型（GPU），且 QA 跑通后才能
  真实对比"带视觉摘要 vs 只有图注"的检索差异。
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
