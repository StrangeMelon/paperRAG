# 参考文献 Chunk 检索降权开发方案

- **日期**: 2026-08-14
- **状态**: implementation checkpoint（TDD、回归与真实验收已通过）
- **范围**: 参考文献 chunk 的识别、混合检索排序、证据选择、abstain 和 QA 旁路
- **关联代码**: `src/paper_rag/chunk/builder.py`、`src/paper_rag/retrieve/`、
  `src/paper_rag/rag/evidence_retrieval.py`、`src/paper_rag/rag/evidence_select.py`、
  `src/paper_rag/rag/abstain.py`、`src/paper_rag/rag/qa_simple.py`

## 1. 背景与问题

论文的 References/Bibliography 节目前保留入库，并在切块时写入
`metadata.is_references=true`。这是正确的数据保留策略，因为“本文引用了哪些工作”
本身是合法问题。

问题在于检索链路没有消费这个标记：

1. dense、BM25 和 FTS5 把参考文献当作普通文本召回；
2. reranker 只看到 query 和 chunk 文本，作者名、题名、年份与问题词面重合时会给出高分；
3. `evidence_select` 再次按模型分数和词面重合选择证据；
4. `abstain` 在最终证据过滤之前计算可信度，参考文献高分可能触发 `confident`；
5. `qa_simple` 直接调用 dense 检索，绕过主检索管道。

此外，FTS5 和内存 BM25 当前没有完整返回 `metadata.is_references`，导致同一个 chunk
在不同检索后端的行为不一致。

## 2. 目标与非目标

### 2.1 目标

1. 普通事实、方法、实验和比较问题中，参考文献只能作为低权重发现线索。
2. 普通问题的最终 LLM 证据和 citation 不包含参考文献 chunk。
3. 用户明确询问参考文献、被引工作或书目信息时，参考文献 chunk 可以正常召回和引用。
4. Qdrant、FTS5、BM25 和 dense-only 旁路使用同一个识别和排序策略。
5. 参考文献降权不会破坏已有的 chunk ID、引用协议和增量入库逻辑。
6. 旧数据缺少 `metadata.is_references` 时，可以通过 section 名称兼容识别。

### 2.2 非目标

- 不删除参考文献 chunk，不改变论文原文和 chunk ID。
- 不使用 LLM 判断参考文献意图；意图判断必须是低延迟、可测试的确定性规则。
- 不在本阶段修改 reranker 模型或重新训练 embedding/reranker。
- 不把所有包含“引用/citation”的问题都归为参考文献问题。

## 3. 目标行为

| 查询类型 | 召回 | 排序 | 最终证据 |
| --- | --- | --- | --- |
| 普通事实/方法/实验问题 | 保留参考文献候选 | 使用惩罚系数降权 | 排除参考文献 |
| 明确参考文献问题 | 正常召回 | 不降权 | 允许参考文献 |
| 普通问题只命中参考文献 | 保留诊断结果 | 低有效分 | `no_evidence` |

降权只影响普通问题。参考文献意图一旦由原始 query 确定，必须贯穿 query rewrite、reflect
和多轮检索，不能因为后续 query 改写而丢失。

## 4. 设计原则

### 4.1 召回保留，答案隔离

参考文献仍然入库和参与召回，避免丢失文献发现能力；但普通问题的最终证据集必须隔离
参考文献。不能只依赖 prompt，因为高分参考文献仍可能影响 abstain 和 evidence selection。

### 4.2 单一策略函数

新增 `src/paper_rag/retrieve/reference_policy.py`，集中提供以下纯函数：

```python
def is_reference_chunk(chunk: dict) -> bool: ...
def detect_reference_intent(query: str) -> bool: ...
def apply_reference_ranking(
    chunks: list[dict], *, reference_intent: bool, penalty: float
) -> list[dict]: ...
def filter_answer_evidence(
    chunks: list[dict], *, reference_intent: bool
) -> list[dict]: ...
```

所有检索入口和最终证据路径调用这些函数，禁止在 pipeline、qa_simple、evidence_select
分别实现一套近似逻辑。

### 4.3 保留原始分数

降权后不能覆盖原始模型分数。建议保留：

```text
score_rerank_raw       cross-encoder 原始分数
score_effective        参考文献策略处理后的排序分数
reference_penalized    是否应用过惩罚
```

诊断页面和 trace 展示 `score_rerank_raw` 与 `score_effective`，方便校准 penalty。

## 5. Chunk 识别与后端字段贯通

### 5.1 识别规则

`is_reference_chunk()` 按以下优先级判断：

1. `chunk["metadata"]["is_references"] is True`；
2. 若结果来自 SQLite 且只有 `metadata_json`，先解析 JSON；
3. 对旧数据，规范化 `section` 后匹配：`references`、`bibliography`、`参考文献`。

规范化应处理大小写、首尾空白和连续空格；不应使用宽泛的“包含 reference”子串，避免
误判 `reference architecture` 等正文章节。

### 5.2 FTS5

修改 `src/paper_rag/retrieve/fts5.py` 的 JOIN 查询，返回 `section`、`title` 和
`metadata_json` 解析后的 `metadata`。FTS5 的索引镜像仍只保存可检索文本，不把
`metadata_json` 放入 FTS 字段。

### 5.3 rank_bm25

修改 `src/paper_rag/retrieve/sparse_bm25.py` 的 payload 构造，读取 SQLite 的
`metadata_json` 并返回 `metadata`。BM25 内存缓存失效和重建时必须保留该字段。

### 5.4 Qdrant

Qdrant 当前 upsert 会写入完整 chunk payload，继续保持该行为。搜索结果应直接透传
`metadata`，不需要新增过滤条件。第一阶段不做硬过滤，以保留参考文献问题的召回能力。

## 6. 参考文献 query 意图

新增确定性词表，初始包括：

```text
参考文献、文献列表、引用了哪些论文、被引工作
references、bibliography、citation list、which papers are cited
```

单独的“引用”“cite”“citation”不应触发，因为它们也可能表示“回答时使用证据引用”。

`detect_reference_intent()` 应支持中英文大小写不敏感匹配，并在测试中覆盖正例和误判例。
如果未来需要扩展词表，应优先增加评测样本，而不是直接引入 LLM 分类。

## 7. 排序和证据流程

### 7.1 主检索 pipeline

在 `retrieve_round_with_rewrite()` 中：

1. 根据原始 query 计算 `reference_intent`；
2. 多查询池化时保留 metadata；
3. rerank 后调用 `apply_reference_ranking()`；
4. 使用 `score_effective` 排序；
5. 再执行论文多样化和 `top_k` 截断。

普通问题的有效分数：

```python
score_effective = score_rerank_raw * reference_penalty
```

非参考文献 chunk 和明确的参考文献问题不乘 penalty。建议初始值为 `0.15`，通过离线
评测调整，不在代码中写死。

### 7.2 Abstain 顺序

当前 `evidence_retrieval.py` 在证据选择之前执行 abstain。应调整为：

```text
多轮候选合并
→ 应用参考文献排序策略
→ 普通问题过滤参考文献资格
→ abstain
→ evidence_select
→ LLM
```

普通问题过滤后没有候选时，直接返回 `no_evidence`。不能让参考文献的原始高分触发
`confident`。

`abstain` 的高质量分数字段增加 `score_effective` 优先级；普通 chunk 的分数保持原有
语义，参考文献只能以降权后的有效分参与裁决。

### 7.3 最终 evidence_select

在 `select_evidence()` 入口增加最终保护：

```python
if not reference_intent:
    chunks = filter_answer_evidence(chunks, reference_intent=False)
```

过滤后的普通问题候选为空时返回空证据，让上层按现有 `no_evidence` 协议处理。明确参考
文献问题继续使用现有的分数、词面重合和单篇限额逻辑。

### 7.4 qa_simple 旁路

`qa_simple.answer()` 继续保持 dense-only 的 ablation 语义，但在 `retrieve()` 返回后必须
调用相同的 `apply_reference_ranking()` 和 `filter_answer_evidence()`。否则主 pipeline
修复后，simple QA 仍会泄漏参考文献。

### 7.5 Prompt 标注

`format_evidence()` 对明确参考文献问题可以保留普通证据头；对内部诊断或未来允许的
参考文献候选，增加：

```text
evidence_role=bibliographic_lead
```

系统 prompt 明确：参考文献块只能支持作者、题名、年份、引用关系等书目信息，不能单独
证明方法、实验结果、数值或结论。该规则是最后一道保护，不替代代码过滤。

## 8. 配置变更

在 `config/default.yaml` 增加：

```yaml
retrieve:
  references:
    enabled: true
    penalty: 0.15
    exclude_from_evidence: true
    legacy_section_fallback: true
```

在 `src/paper_rag/config.py` 增加对应的 Pydantic 配置模型：

- `enabled: bool`
- `penalty: float`，约束为 `0 <= penalty <= 1`
- `exclude_from_evidence: bool`
- `legacy_section_fallback: bool`

`penalty=1.0` 应表示关闭降权，便于 A/B 测试；`exclude_from_evidence=false` 仅用于调试和
评测，不建议生产环境使用。

## 9. 数据迁移和缓存

### 9.1 新数据

现有 builder 标记逻辑继续生效。入库、Qdrant upsert、SQLite round-trip 和 FTS5 sync
必须保证 `metadata.is_references` 不丢失。

### 9.2 旧 SQLite/Qdrant 数据

旧数据分两类处理：

1. 有 `section` 且名称是标准参考文献节：运行时通过 section fallback 识别，不要求立即
   重嵌入。
2. 既没有标记也没有可识别 section：重新解析并入库，补齐 metadata。

重新入库后必须执行对应的 FTS5 单篇同步；否则 SQLite 已有标记但 FTS5 结果仍可能缺失。
Qdrant payload 更新可以只覆盖 metadata，不需要因为该字段变化重新生成向量。

### 9.3 内存索引

BM25 缓存重建必须包含 metadata；新增测试确保缓存失效前后 `is_references` 结果一致。

## 10. TDD 开发门禁

本功能必须按 TDD 顺序开发，任何阶段不得跳过门禁或先写功能代码再补测试。

### 10.1 单个功能切片的执行顺序

每个功能切片严格执行：

```text
编写单元测试明确行为边界
→ 运行测试并确认因缺少目标功能而失败（RED）
→ 编写满足当前边界的最小功能代码（GREEN）
→ 运行聚焦测试和相关回归测试
→ 重构，但不得改变已锁定行为
→ 再次运行测试
```

RED 阶段必须记录失败用例和失败原因。测试因导入错误、环境缺失或测试自身错误而失败，
不算有效 RED；只有断言清楚地证明目标行为尚未实现，才能进入功能编码。

GREEN 阶段只实现当前测试定义的功能边界，不提前混入下一个切片。已有测试失败时必须先
修复，不能通过删除断言、降低测试强度、扩大 mock 范围或跳过测试来推进。

### 10.2 功能切片顺序

按以下顺序推进；前一切片的聚焦测试和相关回归测试全部通过后，才能开始下一切片：

1. 参考文献 chunk 识别和 query 意图识别纯函数。
2. FTS5、BM25、Qdrant 的 metadata 字段贯通。
3. pipeline 的 reference intent、effective score、排序和 trace。
4. evidence filtering、abstain 顺序和 `no_evidence` 行为。
5. `qa_simple`、agentic QA 和 stream QA 旁路一致性。
6. 配置开关、兼容数据和回滚行为。

每个切片先补测试，再改对应功能代码。不得一次性完成多个切片后统一补测。

### 10.3 真实验收门禁

全部单元测试、流程测试、完整测试和 Ruff 检查通过后，最后新增并运行：

```text
scripts/accept_reference_chunk_policy.py
```

真实验收脚本不得 mock 以下核心边界：

- SQLite chunk 和 `metadata_json` 持久化；
- FTS5 实际建表、同步和查询；
- Qdrant 实际 payload 写入和查询；
- 主 retrieve pipeline、abstain 和 evidence selection；
- 最终返回的 evidence chunk IDs。

验收至少准备一篇包含正文和 References/参考文献节的真实解析产物，并执行两类 query：

1. 普通方法或实验问题：允许诊断候选中出现参考文献，但最终 evidence 和 citation 中不得
   出现参考文献 chunk。
2. 明确参考文献问题：参考文献 chunk 不受 penalty，且至少一个参考文献 chunk 进入最终
   evidence。

还必须覆盖普通问题只命中参考文献时返回 `no_evidence`。脚本输出每次查询的 intent、
raw/effective score、最终 evidence IDs、abstain decision 和断言结果；任一断言失败必须以
非零退出码结束。

真实验收脚本必须在项目实际配置和真实存储服务上运行，不能用单元测试 fixture 代替。
验收通过后记录命令、数据范围和结果摘要。只有真实验收全部通过，才能进入后续功能开发
或将本文档状态更新为 implementation checkpoint。

## 11. 测试计划

### 11.1 单元测试

- `is_reference_chunk()`：metadata、metadata_json、section fallback 和普通章节。
- `detect_reference_intent()`：中英文正例、单独“引用/citation”误判例。
- penalty 为 `0`、`0.15`、`1` 时的排序和原始分数保留。
- 普通问题过滤后为空时返回空证据。

### 11.2 检索后端测试

- FTS5 返回 `metadata.is_references`。
- BM25 返回 `metadata.is_references`。
- Qdrant payload 透传该字段。
- hybrid RRF 合并不会丢 metadata。

### 11.3 流程测试

- 普通问题中参考文献 raw rerank 第一，最终 evidence 不包含该 chunk。
- “本文引用了哪些工作”不应用 penalty，且可以返回参考文献。
- 普通问题只命中参考文献时，abstain 结果为 `no_evidence`。
- query rewrite/reflect 多轮后仍保留原始参考文献意图。
- `qa_simple` 和 agentic/stream QA 行为一致。

### 11.4 回归测试

按项目约定执行：

```bash
uv run pytest tests/test_builder.py tests/test_fts5.py tests/test_sparse_bm25.py -q
uv run pytest tests/test_retrieve_pipeline.py tests/test_evidence_retrieval.py \
  tests/test_evidence_select.py tests/test_abstain.py tests/test_qa_simple.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## 12. 评测和上线门槛

新增以下指标：

```text
reference_leak_rate =
非参考文献意图答案中引用参考文献 chunk 的数量 /
非参考文献意图答案总数
```

建议验收门槛：

1. `reference_leak_rate = 0`；
2. 明确参考文献问题的 Recall@8 不下降；
3. 普通检索 Golden Set 的 Recall@8 降幅不超过 1 个百分点；
4. 参考文献误判率低于 1%；
5. 普通问题只命中参考文献时，`no_evidence` 判定稳定；
6. 诊断 trace 能同时看到 raw score、effective score、penalty 和 intent。

上线顺序：

1. 先部署 metadata 贯通和 trace 字段，penalty 设为 `1.0` 做无行为变化观测；
2. 离线评测校准 `penalty`，建议从 `0.15` 开始；
3. 开启普通问题的最终 evidence 排除；
4. 观察 reference leak、Recall、abstain 和 citation 指标；
5. 稳定后再考虑 Qdrant/FTS5 的查询级预过滤优化。

## 13. 实施任务拆分

1. 为 reference policy 纯函数编写失败单测，再实现 `reference_policy.py`。
2. 为 FTS5、BM25、Qdrant metadata round-trip 编写失败单测，再修改后端字段。
3. 为配置模型和默认值编写失败单测，再增加配置实现。
4. 为 intent、effective score、排序和 trace 编写失败单测，再修改 retrieve pipeline。
5. 为 abstain 顺序和空证据行为编写失败单测，再修改 evidence retrieval。
6. 为 evidence select 和 qa_simple 旁路编写失败单测，再修改对应实现。
7. 完成流程测试、完整回归、Ruff 和离线评测。
8. 最后编写并运行 `scripts/accept_reference_chunk_policy.py` 真实验收脚本。
9. 真实验收通过后，对现有数据执行兼容检查和必要的单篇重同步。

## 14. 实施与验收记录

### 14.1 TDD 结果

六个开发门禁均按 RED → GREEN 顺序完成：

1. reference chunk 与 query intent 纯函数；
2. FTS5、BM25、Qdrant metadata 贯通；
3. 配置、pipeline 有效分与 diagnostics；
4. reflect、abstain 和 evidence selection 隔离；
5. simple、agentic 和 stream QA 入口；
6. 真实存储与模型验收。

完整回归命令：

```bash
uv run pytest -q --ignore=tests/test_mineru_bilingual_real.py
```

结果为 `1013 passed`。排除的 MinerU 双语真实测试要求
`PAPER_RAG_REAL_ENGLISH_PDF` 和 `PAPER_RAG_REAL_CHINESE_PDF`，与本功能无关；缺少这两个
外部 PDF 时它们按仓库约定明确失败而不是 skip。

本次涉及的 24 个 Python 文件执行 `ruff check` 和 `ruff format --check` 均通过。全仓
Ruff 仍包含工作区既存的无关脚本和文档格式问题，本功能未批量改写这些用户文件。

### 14.2 真实验收

验收命令：

```bash
uv run python scripts/accept_reference_chunk_policy.py
```

脚本实际运行 SQLite、FTS5、BGE-M3、embedded Qdrant、BGE reranker、hybrid/RRF、
reference policy、abstain 和 evidence selection，不 mock 核心边界。最终输出：

```text
REFERENCE POLICY ACCEPTANCE PASSED
report: demo-reference-policy-data/acceptance-report.json
```

关键结果：

| 场景 | raw rerank | effective score | 最终结果 |
| --- | ---: | ---: | --- |
| 普通问题中的 Method | 0.9998 | 0.9998 | 进入 evidence |
| 普通问题中的 References | 0.9903 | 0.1485 | 从 evidence 排除 |
| 明确被引工作问题中的 References | 0.9667 | 0.9667 | 进入 evidence |
| reference-only 普通问题 | 0.9979 | 0.1497 | `no_evidence` |

真实验收通过后，本方案才从 `design proposal` 更新为 `implementation checkpoint`。

## 15. 风险与回滚

### 风险

- section 命名不规范导致旧数据漏标；通过 metadata 优先、section fallback 和重入库解决。
- penalty 过低导致合法的“发现相关论文”问题召回下降；通过显式 intent 分支和 Recall@8
  评测控制。
- 只修改主 pipeline 而遗漏 qa_simple 或 stream 路径；通过共享策略模块和旁路测试避免。
- 在 abstain 之后才过滤，导致参考文献仍然触发高置信度；流程测试必须覆盖该顺序。

### 回滚

- 将 `retrieve.references.enabled` 设为 `false`，恢复不降权行为。
- 将 `penalty` 设为 `1.0`，保留 metadata 和 trace，不需要回滚数据。
- `exclude_from_evidence=false` 可用于临时诊断，但不应作为生产默认值。
