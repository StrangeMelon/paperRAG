# ADR-0003 · Wiki 概念身份、跨语言解析与持久化任务队列

- **日期**: 2026-08-06
- **状态**: accepted（wiki 全模块 14 课 TDD 落地，纯逻辑套件 100+ 测试通过；
  真实 Demo `scripts/demo_wiki.py` 中英闭环验收）
- **关联**: 重建课程 wiki 模块（`src/paper_rag/wiki/*` 12 个文件 +
  `scripts/wiki_worker.py`）；[ADR-0001](0001-fts5-cjk-bigram.md) 与
  [ADR-0002](0002-evidence-select-cjk-overlap.md)（CJK bigram 口径，本 ADR 的
  `context.py` 改写提示沿用同一口径）；全局约束"重建版必须支持中文论文"

## 背景：wiki 要解决什么，基准的实现拦在哪

wiki 是自进化的概念笔记层：论文入库后抽取概念，建立"定义 + 双语别名 + 关键
论文 + 证据 chunk"的词条，供 QA 侧做查询改写增强与术语背景。**证据边界不变：
wiki 只作背景，答案仍只能引用 `[chunk:<id>]`。**

基准实现（1228 行 / 12 文件）有五个在 20,000 篇目标规模下站不住的设计：

1. **进程内 daemon 队列**：批量 CLI 退出时未消费任务静默丢失。
2. **`find_match` 走 `list_all()`**：每篇论文每个概念一次全表扫描 + Python 侧
   逐条比别名。词条数上千即不可接受。
3. **`key_papers_json` 数组**：无界增长，且无法反查"哪些概念由这篇论文支撑"。
4. **24h 锁锁整条词条**：批量入库时同概念的后续论文关联被整条丢弃。
5. **`_definition_phrases` 用 `[A-Za-z]` 正则**：中文定义一个短语都抽不出来，
   改写提示对中文词条完全失效（与 ADR-0001/0002 同源的英文假设）。

## 决策

### 1. 概念身份：创建时一次生成的稳定句柄，查找一律走 labels 表

`entry_id = "concept:" + normalize_label(创建时的名字)`，**创建后永不重算**。
`normalize_label` 采用 NFKC + casefold + 剔除空白/连字符/标点，CJK 原样保留。

考虑过 `wiki:<uuid>`（提案的原始方案）。否决理由：UUID 使真实 Demo 与集成
测试无法断言稳定 ID，Qdrant point id 由 entry_id 派生后无法反查词条，克隆态
复跑对不上基线——与本项目"Demo 必须带断言、失败非零退出"的验收协议冲突。

关键认识：**"强化学习"与 Reinforcement Learning 产生两个 ID 的根因不是 ID
格式，而是创建前没做跨语言解析**。有了 `wiki_labels` 索引表与三级解析，中文
论文进来时先解析到已有词条，根本走不到"生成第二个 ID"那一步。名字演化由
labels 表承载，ID 只是句柄，两者解耦。

代价：主名若从中文换成英文，ID 会显得过时。接受——它不参与任何查找。

### 2. 三级解析：向量只负责召回，判定权在 LLM

不设"跨语言相似度阈值"。zh↔en 词对的 BGE-M3 相似度分布与同语言不同，单一
阈值必然在漏合并与误合并之间摇摆。改为：

```
规范化名/别名精确命中（labels 表索引查询）      -> match
相似度 >= auto_merge_same_lang(0.90) 且同语言   -> match
相似度 >= recall_floor(0.60)                    -> LLM 带定义/类别/证据验证
                                                   -> match | review | novel
相似度 <  recall_floor                          -> novel
短标签（ASCII <= 4 字符 / CJK 单字）            -> 永不单独触发 match
```

阈值偏差只导致多走几次 LLM 验证，不会直接产生错误合并。`review` 既不建也不
并，进人工复核队列。中文短标签单独分档：`is_short_label` 按 CJK 字数判定，
"强化学习"（4 字）与"蒸馏"（2 字）都不是短标签，只有单字才算。

### 3. 关系型真相源：8 张表，SQLite 权威，Qdrant 可重建

`wiki_entries`（快照 + `merged_into` 重定向 + `qdrant_dirty` 脏标）、
`wiki_labels`（**唯一查找入口**，`text_norm` 索引）、`wiki_entry_papers`、
`wiki_entry_evidence`、`wiki_versions`（追加式）、`wiki_jobs`、
`wiki_review_queue`、`wiki_usage`。

Qdrant 同步失败**不回滚 SQLite**：置 `qdrant_dirty`，worker 补偿轮重试。

### 4. 合并与重定向闭环（两份方案都缺的一环）

`merged_into` 非空即 tombstone。读路径默认跟随重定向（带环保护，最多 10 跳）。
`review_queue.resolve_merge` 把源词条的 labels / papers / evidence 全部吸收进
目标词条，源保留为 tombstone 不删除。缺这条路径，复核队列只能看不能动。

### 5. 持久化 `wiki_jobs` + 独立 worker 进程

ingest 侧只做一次 INSERT（`paper_id + content_fingerprint` 幂等，force 重建
自然产生新指纹→新任务），LLM 调用全在 `scripts/wiki_worker.py`（`--once` /
`--drain` / 退避重试 / `requeue_stale` 断点续跑）。

**副作用**：批跑与 wiki 建设天然解耦，全量入库期间无需关闭 wiki——这撤回了
"批跑期间关 wiki"的原始建议。

### 6. 限频只锁定义重写，白名单操作只追加

24h 锁只限制昂贵的 `propose_definition`；`add_label` / `add_key_paper` /
`add_evidence` / `add_variant` / `add_open_problem` 不受限。LLM 只能返回白名单
操作，未知操作忽略——**自动流程不可能删除旧事实**。`triggers` 在 match 分支
下即使 `patch_entry` 被 self_eval 拦下，论文/证据/双语标签的机械关联仍直接
落库。

### 7. 解析质量门槛（正式库实测暴露的需求）

`--limit 3` 试入的三篇里，一篇征文通知被标 `mineru+broken` 且只有 7 chunk。
这类文件喂进概念抽取会产出垃圾词条并污染 QA 消费端。故按 `parsed_with` 黑名单
与 `min_chunks` 下限过滤，被跳过者在 `wiki_jobs` 记 `skipped` 加原因，不静默
丢弃。

### 8. 中英文差异的其余落点

- 三个 LLM prompt 全部中英双模板，按论文语言路由；语言随任务显式传递，
  worker 不靠标题猜。
- 新词条定义语言跟随创建论文；patch 保守不改写定义语言。
- 抽取字符预算分档（zh 4000 / en 6000，中文信息密度更高）。
- `consistency` 定义长度门槛分档（en 20 字符 / zh 12 汉字）。
- `context._definition_phrases` 增 CJK 分支（抽 CJK 连续片段、剥离中文停用词
  后截前 8 字），并配一份中文停用词表；BM25 侧交给既有 bigram（ADR-0001/0002）。

### 9. 词条定义进 QA 背景前剥离 chunk 引用（真实 Demo 实证）

建条 prompt 要求 LLM 在定义里引用 `[chunk:xx]` 以保证有据可依，这对词条自身的
可溯源是必要的。但真实全链路 Demo 暴露出：这些 id 属于词条的历史证据，**不是
本轮检索结果**，原样进入 QA 背景块会诱导模型照抄成伪引用，直接冲击"答案只能
引用检索到的 chunk"这条不变量。

故 `context._entry_to_context` 在组装上下文时剥离该字面；背景块表头也刻意不写
引用的字面格式（写出来同样会被当成可照抄的样例）。结果是背景块内 `[chunk:`
零出现，由纯逻辑测试与 Demo 双向钉死。剥离只发生在消费端，词条自身存储的定义
保留引用不变。

## 后果

**正面**：查找从全表扫描变为索引查询；任务不随进程丢失；跨语言概念可合并且
误合并有复核闭环；wiki 全链路支持中文论文；所有可调项入 `config/default.yaml`
的 `wiki.*`。

**负面 / 待观察**：8 张表是可观的 schema 增量（由 `scripts/init_store.py` 显式
建出，不靠懒建）；三级解析的 LLM 验证增加 token 成本（换取合并正确性）；
`context.resolve_wiki_context` 仍是每问一次的内存遍历（千级词条可接受，万级
需改为标签表反向索引查询，届时再迭代）。

**未做**：词条的自动 GC / 归并建议、跨概念关系图谱、`review_queue` 的 Web UI。
