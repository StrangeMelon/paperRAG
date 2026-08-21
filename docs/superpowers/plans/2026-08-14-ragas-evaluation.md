# RAGAS 评测实现方案

- **日期**: 2026-08-14
- **状态**: implementation checkpoint（离线实现与回归已通过；真实模型验收待凭证）
- **范围**: RAGAS 0.4 依赖、评审模型、RAGAS Golden Set、RAGAS runner、RAGAS report、
  CLI/Dashboard 的 RAGAS 分支和真实验收
- **关联代码**: `src/paper_rag/evaluation/ragas.py`、`ragas_models.py`、
  `ragas_schema.py`、`ragas_runner.py`、`ragas_gates.py`、`scripts/evaluate.py` 的 RAGAS
  分支和 Dashboard 的 RAGAS 分支

## 0. 实施检查点

截至 2026-08-14，本方案的离线实现已经落地：

- RAGAS 0.4.3、Instructor 和 LangChain Community 兼容组合已锁定；
- 五种现代 collection metric 均可在不发起网络请求时完成真实对象构造；
- RAGAS 专用 schema、Golden Set loader、runner、report、coverage 和 tag 聚合已完成；
- CLI 已支持 RAGAS 独立默认数据集、并发限制、fail-under、min-coverage 和 baseline
  regression gate；
- Dashboard 的 RAGAS 分支已接入独立 runner 和报告展示；
- `scripts/accept_ragas_evaluation.py` 已提供 adapter-only 和 end-to-end 两档真实验收；
- 非 real 测试 `1003 passed`，RAGAS/Custom 聚焦测试、Custom 离线验收和 scoped Ruff 均通过；
- 禁止修改的 Custom 核心文件 SHA-256 与实施前一致。

尚未完成的是外部模型调用验收：当前环境没有配置 `RAGAS_BASE_URL`、`RAGAS_API_KEY`、
`RAGAS_MODEL` 和 `RAGAS_EMBEDDING_MODEL`。验收入口会返回结构化配置错误，不会伪装成功。

## 1. 强制范围边界

本方案只负责 RAGAS 评测。现有 Custom 评测是稳定边界，不允许借 RAGAS 接入进行重构。

### 1.1 禁止修改

以下文件和行为不属于本方案，实施时禁止修改：

- `src/paper_rag/evaluation/custom.py`；
- `src/paper_rag/evaluation/retrieval.py`；
- `src/paper_rag/evaluation/base.py` 的现有抽象契约；
- `src/paper_rag/evaluation/runner.py` 的 Custom 执行与报告语义；
- `src/paper_rag/evaluation/composite.py`；
- `scripts/accept_evaluation.py`；
- `tests/fixtures/evaluation/golden.json`；
- `tests/fixtures/evaluation/retrieval_golden.json`；
- Custom 指标名称、公式、默认值、聚合方式、状态和报告结构；
- Custom、retrieval 和 composite 后端的 CLI/Dashboard 现有行为。

如果 RAGAS 需求与上述边界冲突，必须在 RAGAS 专用模块内解决，不得通过扩展或修改
Custom 核心来解决。

### 1.2 允许修改

本方案只允许以下变更：

- 修正 `[project.optional-dependencies].evaluation` 的 RAGAS 依赖兼容性；
- 在现有 `evaluation` 配置下新增隔离的 `ragas` 子配置；
- 修改 `src/paper_rag/evaluation/ragas.py`；
- 新增 `ragas_models.py`、`ragas_schema.py`、`ragas_runner.py`；
- 新增独立 RAGAS Golden Set，不覆盖现有 Custom fixture；
- 修改 `scripts/evaluate.py` 中 `backend == "ragas"` 的分支；
- 修改 Dashboard service/page 中 `backend == "ragas"` 的分支和展示；
- 新增 RAGAS 单元测试、依赖契约测试和真实验收脚本；
- 增加只读回归测试，证明 Custom 核心行为未变化。

## 2. 背景与现状

项目已经有 `RagasEvaluator` 初版，可以把 chunk 的 `text/content/page_content` 转成
RAGAS contexts，并通过现有 QA runner 逐题评分。但它还不能真实投入使用：

1. 当前环境锁定 `ragas==0.4.3` 和 `langchain-community==0.4.2`，`import ragas` 会因
   缺少 `langchain_community.chat_models.vertexai` 失败；
2. 适配器使用 `ragas.metrics` 旧指标单例和 `datasets.Dataset`；
3. RAGAS 0.4 推荐 `ragas.metrics.collections` 和指标 `ascore()`；
4. 没有显式注入评审 LLM 和 embedding，无法可靠使用项目的 OpenAI-compatible 服务；
5. `answer_correctness` 虽被声明支持，却没有实际执行；
6. 每题单独调用 RAGAS，模型复用、并发、重试、缓存和成本不可控；
7. 当前单元测试 mock 了真正的 RAGAS 调用，没有覆盖真实 import、结构化输出和 embedding；
8. 现有 Custom Golden Set 不是 RAGAS 的所有物，不能为接入 RAGAS 而改变其 schema。

## 3. 目标与非目标

### 3.1 目标

1. 使用 RAGAS 0.4 现代 API 评估回答忠实度、相关性、上下文质量和正确性；
2. 显式配置独立 judge LLM 与 embedding，支持 OpenAI-compatible endpoint；
3. 新建 RAGAS 专用 Golden Set、runner 和 report，不影响 Custom 核心；
4. 一次 RAGAS run 只执行一次 QA 采集，并复用一组 judge/embedding client；
5. 建立 reference、空上下文、拒答、NaN、部分失败和 coverage 的严格规则；
6. 支持 CLI 和 Dashboard 的纯 RAGAS 执行；
7. 单元测试完全离线，真实 RAGAS 验收显式 opt-in；
8. 报告记录 RAGAS、模型、Golden Set 和语料快照版本，保证可比较。

### 3.2 非目标

- 不实现或调整 hit rate、MRR、ID recall、citation、abstain 等 Custom 指标；
- 不修改 Custom、retrieval 或 composite runner；
- 不让 RAGAS 与 Custom 共用新的批量协议；
- 不把 RAGAS 结果写入 Custom report；
- 不自动生成 Golden Set 或自动生成未经审核的 reference answer；
- 不在普通 PR 测试中调用付费服务；
- 不将 RAGAS 放进线上单请求路径；
- 不在本阶段引入 RAGAS testset generator 或外部观测平台。

## 4. RAGAS 独立架构

RAGAS 采用旁路架构，不改变 Custom 调用链：

```text
tests/fixtures/evaluation/ragas_golden.json
                    |
                    v
          RagasGoldenSetLoader
          严格校验 + 文件哈希
                    |
                    v
             RagasEvalRunner
                    |
       qa_agentic.answer() 每题一次
                    |
                    v
              RagasSample[]
                    |
                    v
             RagasEvaluator
       judge LLM + embedding + metrics
                    |
                    v
           ragas-report.v1 JSON
              |             |
              v             v
             CLI        Dashboard
```

现有 `EvalRunner -> CustomEvaluator`、`RetrievalEvalRunner` 和 `CompositeEvaluator` 不进入
这条链路，也不为这条链路增加接口。

## 5. 指标设计

### 5.1 指标与输入

| 项目指标名 | RAGAS 0.4 类 | 必需输入 | 作用 |
| --- | --- | --- | --- |
| `faithfulness` | `Faithfulness` | query、answer、contexts | 回答陈述是否受证据支持 |
| `answer_relevancy` | `AnswerRelevancy` | query、answer、embedding | 回答是否直接回应问题 |
| `context_precision` | `ContextPrecisionWithReference` | query、contexts、reference | 相关上下文是否排在前面 |
| `context_recall` | `ContextRecall` | query、contexts、reference | 上下文是否覆盖参考答案 |
| `answer_correctness` | `AnswerCorrectness` | query、answer、reference、embedding | 回答与参考答案是否一致 |

适配器显式导入：

```python
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
```

禁止继续使用 `ragas.metrics` 的弃用单例。`context_precision` 固定使用 WithReference，不能
在缺 reference 时静默改变同名指标语义。

### 5.2 适用条件

1. `context_precision`、`context_recall`、`answer_correctness` 要求非空 reference；
2. `answer_relevancy` 和默认权重的 `answer_correctness` 要求 embedding；
3. faithfulness 和 context 指标要求非空 contexts；
4. `expected_abstain=true` 的样本不适合 RAGAS 五项指标，标记 `not_applicable`；
5. 正样本意外返回空 answer 或空 contexts 时标记 `missing_input`，不能记 0 或静默跳过；
6. NaN、无穷值或超出 `[0, 1]` 的结果记为 metric error，不进入均值；
7. 单个指标失败不得丢弃同一样本的其他成功指标；
8. 每个指标聚合必须包含 `mean`、`count`、`eligible_count` 和 `coverage`。

coverage 定义为：

```text
成功产生有限分数的 eligible 样本数 / eligible 样本总数
```

`not_applicable` 不进入分母；`missing_input`、模型异常和非法数值进入 eligible 分母但不进入
成功 count，防止跳过失败样本抬高均分。

## 6. RAGAS Golden Set

### 6.1 独立文件与 Schema

新增：

```text
tests/fixtures/evaluation/ragas_golden.json
```

使用独立 `ragas-eval.v1`，不修改现有 `golden.json`：

```json
{
  "schema_version": "ragas-eval.v1",
  "corpus": {
    "selection": "all_indexed"
  },
  "test_cases": [
    {
      "id": "ragas-001",
      "query": "论文提出的方法解决了什么问题？",
      "paper_ids": ["paper-id"],
      "reference_answer": "依据论文原文人工整理的参考答案。",
      "expected_abstain": false,
      "reference_chunk_ids": ["chunk-id"],
      "tags": ["factual", "zh"],
      "notes": "参考答案的审核说明"
    }
  ]
}
```

`reference_chunk_ids` 用于审计 reference answer 的来源，不参与 Custom 指标，也不复用
`expected_chunk_ids` 的 Custom 语义。

### 6.2 严格校验

在任何 QA、judge 或 embedding 调用前检查：

1. schema version 正确且 test cases 非空；
2. case ID 唯一，query 非空；
3. 正样本 `expected_abstain=false` 且 reference answer、paper IDs、reference chunk IDs 非空；
4. 负样本显式设置 `expected_abstain=true`，允许 reference 和 reference chunks 为空；
5. 所有 paper 状态为 `done`；
6. reference chunk 存在，并属于 case 的 paper IDs；
7. tags 是非空字符串列表；
8. reference answer 必须经人工审核，不能由被评测模型未经审核地产生。

Golden Set 结构或语料引用错误时整次 RAGAS run 失败，不能降级成单题错误。

### 6.3 初始数据集

先建立 30–50 条可审核基线，后续扩展到 100 条以上：

| 类型 | 建议比例 |
| --- | ---: |
| 单篇事实、方法和实验问答 | 35% |
| 跨段推理 | 20% |
| 多论文比较 | 20% |
| 术语解释 | 10% |
| 无答案/越界问题 | 10% |
| 中英混合和歧义边界 | 5% |

RAGAS 聚合结果同时按 `tags` 和语言分组，避免总均值掩盖中文、多论文或负样本问题。

## 7. 配置设计

在现有 `evaluation` 下只追加 `ragas` 子配置；原有 Custom 字段和值不改：

```yaml
evaluation:
  enabled: false
  provider: custom
  backends: [custom]
  metrics: [hit_rate, mrr, recall, paper_hit_rate, citation_precision, citation_recall]
  golden_set: tests/fixtures/evaluation/golden.json
  top_k: 8
  fail_on_error: false
  retrieval_golden_set: tests/fixtures/evaluation/retrieval_golden.json

  ragas:
    golden_set: tests/fixtures/evaluation/ragas_golden.json
    metrics: [faithfulness, answer_relevancy, context_precision, context_recall]
    base_url: $RAGAS_BASE_URL
    api_key: $RAGAS_API_KEY
    judge_model: $RAGAS_MODEL
    embedding_base_url: $RAGAS_EMBEDDING_BASE_URL
    embedding_api_key: $RAGAS_EMBEDDING_API_KEY
    embedding_model: $RAGAS_EMBEDDING_MODEL
    max_concurrency: 4
    timeout_sec: 120
    max_retries: 3
    cache_dir: data/evaluation/ragas_cache
    judge_options: {}
```

`src/paper_rag/config.py` 新增 `_EvaluationRagas`，仅作为 `_Evaluation.ragas` 字段。约束：

- 默认 evaluation/custom 配置加载行为不变；
- 只有构造 RAGAS runner 时才要求 RAGAS 凭证；
- embedding endpoint/key 可回退到 judge endpoint/key，但 embedding model 必须显式配置；
- `max_concurrency` 为 `1..16`，`timeout_sec > 0`，`max_retries >= 0`；
- `judge_options` 承载供应商参数，例如关闭 thinking；
- 报告和日志禁止记录 API key。

`.env.example` 只增加变量名和说明，不包含真实凭证。

## 8. 依赖策略

当前已验证可导入的组合：

```toml
evaluation = [
    "ragas==0.4.3",
    "datasets>=4,<5",
    "langchain-community>=0.3.31,<0.4",
    "instructor>=1.7,<2",
]
```

`langchain-community<0.4` 是 RAGAS 0.4.3 顶层导入旧 VertexAI 模块的临时兼容限制。
`instructor>=1.7` 则保证 `llm_factory` 依赖的 `instructor.from_openai` 可用；RAGAS 0.4.3
自身没有声明这一最低版本。升级 RAGAS 时必须先运行 import contract 和真实 adapter
smoke，再调整版本范围。

新增 evaluation-extra 依赖契约：

```bash
uv run python -c \
  "import ragas; from ragas.metrics.collections import Faithfulness"
```

基础环境仍保留 lazy import，未安装 evaluation extra 时不得影响 Custom 模块导入和运行。

## 9. RAGAS 模型工厂

新增 `src/paper_rag/evaluation/ragas_models.py`：

```python
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

judge_client = AsyncOpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
    timeout=settings.timeout_sec,
)
judge_llm = llm_factory(
    settings.judge_model,
    provider="openai",
    client=judge_client,
    **settings.judge_options,
)

embedding_client = AsyncOpenAI(
    api_key=embedding_api_key,
    base_url=embedding_base_url,
    timeout=settings.timeout_sec,
)
embeddings = embedding_factory(
    "openai",
    model=settings.embedding_model,
    client=embedding_client,
    interface="modern",
)
```

实现约束：

1. 一个 RAGAS run 只创建一组 judge 和 embedding client；
2. judge 模型必须通过真实结构化输出验收，不能按模型名推断；
3. judge 与 QA 生成模型相互独立；
4. 同一基线固定 RAGAS patch、judge model、embedding model 和 judge options；
5. 只允许一层网络重试，避免多层指数重试；
6. cache key 包含指标、模型、RAGAS 版本和完整输入；
7. cache 位于已忽略的 `data/` 下，且不保存 secret。

## 10. RAGAS 专用数据模型

新增 `src/paper_rag/evaluation/ragas_schema.py`：

```python
@dataclass(frozen=True)
class RagasCase:
    id: str
    query: str
    paper_ids: list[str]
    reference_answer: str | None
    reference_chunk_ids: list[str]
    expected_abstain: bool
    tags: list[str]
    notes: str | None


@dataclass
class RagasSample:
    id: str
    query: str
    response: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    citations: list[str]
    reference: str | None
    expected_abstain: bool
    actual_abstain: str | None
    tags: list[str]
```

同时定义 RAGAS 专用 metric result、sample result 和 report dataclass。不得把这些字段加入
`BaseEvaluator`、`GoldenCase`、`QueryResult` 或现有 `EvalReport`。

## 11. RagasEvaluator

`src/paper_rag/evaluation/ragas.py` 保留现有 `RagasEvaluator` 类名，但实现只服务 RAGAS：

1. 初始化时校验五种支持指标；
2. lazy import RAGAS 0.4 collection classes；
3. 通过 `ragas_models.py` 注入 judge 和 embedding；
4. 新增 `evaluate_batch(samples: list[RagasSample])`；
5. 同步边界内部只执行一次 `asyncio.run(_aevaluate_batch(...))`；
6. 使用 `asyncio.Semaphore(max_concurrency)` 控制调用；
7. 每个 sample/metric 独立捕获异常；
8. 从 `MetricResult.value` 读取分数并执行 finite/range 校验；
9. 保留现有单题 `evaluate()` 入口，仅作为兼容包装，不修改基类契约；
10. contexts 过滤空字符串，支持 `text/content/page_content`，但不改变 chunk 本身。

异步执行结构：

```text
evaluate_batch(samples)
→ asyncio.run(_aevaluate_batch(samples))
→ 计算每个 sample 的 eligible metrics
→ semaphore 包裹 metric.ascore()
→ gather(return_exceptions=True)
→ 校验 MetricResult.value
→ 返回与输入 sample ID 对齐的结果
```

禁止在每个 sample 或 metric 内调用 `asyncio.run()`。

## 12. RagasEvalRunner

新增 `src/paper_rag/evaluation/ragas_runner.py`，不修改现有 `EvalRunner`。

执行流程：

1. 使用 `ragas_schema.py` 加载并校验 `ragas-eval.v1`；
2. 固定 Golden Set SHA-256 和当前 corpus manifest；
3. 对每个 case 调用一次 `qa_agentic.answer(query, paper_ids=case.paper_ids)`；
4. 将 QA 输出转换为 `RagasSample`；
5. QA 失败记录为 `qa_error`，不伪装成 metric error；
6. 一次调用 `RagasEvaluator.evaluate_batch()`；
7. 按 sample ID 合并指标、metric status、errors 和 latency；
8. 计算整体及 tag 分组的 mean/count/eligible count/coverage；
9. 生成独立 `ragas-report.v1`；
10. 评测期间显式绕过 QA cache，或按缓存 chunk IDs 回捞固定版本上下文。

第一阶段 QA 保持串行，降低实现范围；只有 RAGAS metric 调用使用独立并发。后续若需要 QA
并发，也只能在 `RagasEvalRunner` 内增加，不能修改共享 Runner。

## 13. RAGAS 报告

报告使用独立 schema，不替换现有 Custom 报告：

```json
{
  "schema_version": "ragas-report.v1",
  "run_id": "...",
  "created_at": "...",
  "evaluation": {
    "mode": "ragas",
    "ragas_version": "0.4.3",
    "judge_model": "...",
    "embedding_model": "...",
    "golden_set_sha256": "...",
    "corpus_manifest_sha256": "..."
  },
  "aggregate_metrics": {
    "faithfulness": {
      "mean": 0.86,
      "count": 42,
      "eligible_count": 45,
      "coverage": 0.9333
    }
  },
  "tag_metrics": {},
  "query_results": []
}
```

逐题结果至少包含：

- query、response、retrieved chunk IDs、citations；
- reference、reference chunk IDs、expected/actual abstain；
- 每个指标的 value、status、latency 和 error；
- QA latency、RAGAS latency 和总 latency；
- `ok`、`partial`、`qa_error`、`not_applicable` 状态；
- 不包含 API key 或完整带参数 endpoint。

报告默认写入 `data/evaluation/ragas-<timestamp>.json`，不写入 Custom history schema。

## 14. CLI 接入

只修改 `scripts/evaluate.py` 的纯 RAGAS 分支：

```python
if args.backend == "ragas":
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner

    # 使用 RAGAS 专用 Golden Set、runner 和 report
```

RAGAS 默认指标：

```text
faithfulness answer_relevancy context_precision context_recall
```

RAGAS 分支新增或消费：

```text
--test-set
--metrics
--max-concurrency
--fail-on-error / --no-fail-on-error
--fail-under metric=value
--min-coverage metric=value
--compare-baseline path
--max-regression metric=value
--output
```

边界要求：

1. `backend=custom` 继续走原有代码和默认指标；
2. `mode=retrieval` 继续走原有 `RetrievalEvalRunner`；
3. `backend=composite` 保持现状，本方案不增强它；
4. RAGAS 的默认 test set 是 `ragas_golden.json`，不覆盖 Custom 默认路径；
5. RAGAS 参数校验在 QA 执行前完成；
6. baseline 的模型或 RAGAS 版本不同则拒绝直接比较。

## 15. Dashboard 接入

Dashboard 只调整 `backend == "ragas"` 分支：

1. 使用 `RagasEvalRunner` 和 RAGAS 专用 Golden Set；
2. 显示配置就绪状态，只展示 endpoint host、judge 和 embedding model；
3. 提供五种 RAGAS 指标，包括 `answer_correctness`；
4. 展示 mean 和 coverage；
5. 展示 `ok/partial/qa_error/not_applicable`；
6. 展示 RAGAS、judge、embedding、Golden Set 和 corpus 版本；
7. 展示逐指标错误和 latency；
8. 页面不允许输入或回显 API key。

Custom、retrieval 和 composite 的 service 分支、控件默认值、运行参数、历史格式和结果渲染
不得改变。RAGAS history 可以使用独立 JSONL 文件或带独立 schema 的记录，不能要求迁移
Custom 历史记录。

## 16. 测试策略

### 16.1 RAGAS 单元测试

扩展 `tests/test_evaluation_ragas.py`，新增独立 schema/runner tests：

- 五种现代 metric 构造和字段映射；
- `answer_correctness` 确实执行；
- reference 缺失在昂贵调用前失败；
- expected abstain 标记 `not_applicable`；
- 空 answer/contexts 产生 `missing_input` 并降低 coverage；
- 单指标异常不丢弃其他结果；
- NaN、inf、负数和大于 1 的值被拒绝；
- contexts 过滤空文本并支持项目字段；
- semaphore 不超过配置并发；
- 一个 batch 只创建一次模型工厂和事件循环；
- 异步完成顺序不影响 sample ID 对齐；
- Golden Set schema、重复 ID、chunk/paper 归属校验；
- report 聚合、tag 聚合和 JSON 序列化；
- report 与异常不泄漏 API key；
- baseline 模型不一致时拒绝比较。

单元测试允许 mock 外部 metric 的 `ascore()`，但不 mock 项目自己的字段映射、eligibility、
coverage、错误处理和聚合逻辑。

### 16.2 Custom 不变性回归

本方案不修改 Custom 测试。实施后运行现有测试作为回归门禁：

```bash
uv run pytest \
  tests/test_evaluation_custom.py \
  tests/test_evaluation_runner.py \
  tests/test_evaluation_retrieval.py \
  tests/test_evaluation_composite.py -q

uv run python scripts/accept_evaluation.py
```

如果这些测试因 RAGAS 变更失败，必须修复 RAGAS 隔离层；不得通过修改 Custom 实现、放宽
断言、更新 Custom fixture 或改变 Custom 期望来让测试通过。

### 16.3 依赖契约

在安装 `--extra evaluation` 的独立环境中执行：

```bash
uv run python -c \
  "import ragas; from ragas.metrics.collections import Faithfulness"
```

同时验证未安装 extra 的基础环境仍能导入并执行 Custom 模块，证明 RAGAS lazy import 没有
污染现有核心。

## 17. 真实验收

新增：

```text
scripts/accept_ragas_evaluation.py
```

不得修改或替换 `scripts/accept_evaluation.py`。

### 17.1 Adapter smoke

使用固定 query、answer、contexts 和 reference，真实调用：

- `import ragas`；
- OpenAI-compatible judge endpoint；
- judge 结构化输出；
- embedding endpoint；
- 五种 RAGAS metric。

不得 mock RAGAS metric、judge client、embedding client 或 `MetricResult`。只断言指标存在、
有限且位于 `[0, 1]`，不对外部模型的精确浮点值作断言。

### 17.2 RAGAS end-to-end

使用至少两篇状态为 `done` 的真实论文和 5–10 条 RAGAS Golden cases：

```text
真实 SQLite/Qdrant/FTS5/BM25
→ qa_agentic 检索和生成
→ RagasSample
→ 真实 judge/embedding
→ ragas-report.v1
```

不得 mock 检索、生成、judge 或 embedding。脚本输出 run ID、语料哈希、Golden Set 哈希、
模型名、coverage 和报告路径。缺凭证或语料时明确失败，不能伪装成功。

建议命令：

```bash
uv sync --extra evaluation

uv run python scripts/accept_ragas_evaluation.py --adapter-only

uv run python scripts/evaluate.py \
  --backend ragas \
  --test-set tests/fixtures/evaluation/ragas_golden.real.json \
  --metrics faithfulness answer_relevancy context_precision \
    context_recall answer_correctness \
  --output data/evaluation/ragas-baseline.json
```

## 18. CI 与质量门禁

### 18.1 每个 PR

运行无网络的 RAGAS 单元测试、依赖契约和 Custom 不变性回归：

```bash
uv run pytest tests/test_evaluation_ragas*.py -q
uv run pytest tests/test_evaluation_custom.py tests/test_evaluation_runner.py \
  tests/test_evaluation_retrieval.py tests/test_evaluation_composite.py -q
uv run python scripts/accept_evaluation.py
uv run ruff check .
uv run ruff format --check .
```

### 18.2 Nightly / 手动门禁

使用受保护 secret 运行真实 RAGAS Golden Set。先执行三次建立波动范围，再设置：

```text
faithfulness coverage >= 0.95
context_recall coverage >= 0.95
关键指标相对同版本基线回退不超过校准容差
无 qa_error 或配置错误
无 secret 泄漏
```

judge、embedding、prompt、RAGAS 版本或 Golden Set 变化时建立新基线，不与旧基线直接比较。

## 19. TDD 实施切片

每个切片严格执行 RED → GREEN → 聚焦回归 → Custom 不变性回归 → 重构 → 完整回归。
导入失败、缺环境变量或测试自身错误不算有效 RED。

### 切片 1：依赖与 import contract

1. 增加能复现真实 import 问题的 evaluation-extra 测试；
2. 固定兼容版本并更新 lockfile；
3. 在干净 extra 环境验证 collection import；
4. 验证基础环境的 Custom lazy import 不受影响。

### 切片 2：RAGAS 配置与模型工厂

1. 先测试 disabled、缺配置、endpoint 回退和脱敏；
2. 只增加 `_Evaluation.ragas`，不调整现有 Custom 字段；
3. 实现 lazy `ragas_models.py`；
4. mock 工厂验证 OpenAI-compatible 参数映射。

### 切片 3：现代指标适配

1. 为五种指标和输入条件写失败测试；
2. 移除 RAGAS 适配器中的旧 singleton/Dataset 路径；
3. 实现 collection metric、`ascore()` 和 batch；
4. 实现 finite/range 校验、并发和逐指标错误；
5. 不修改 `BaseEvaluator`。

### 切片 4：RAGAS Schema、Golden Set 和 Runner

1. 新建 `ragas_schema.py` 和 `ragas-eval.v1` 测试；
2. 新建 `ragas_runner.py`，锁定每题 QA 只执行一次；
3. 实现语料预检、样本转换、coverage 和 tag 聚合；
4. 生成 `ragas-report.v1`；
5. 不修改共享 `runner.py` 和现有 Golden Set。

### 切片 5：RAGAS CLI 分支

1. 测试 RAGAS 默认 fixture、指标和错误退出码；
2. 接入 `RagasEvalRunner`；
3. 实现 RAGAS baseline/coverage 门禁；
4. 回归证明 Custom/retrieval/composite 分支输出未变化。

### 切片 6：RAGAS Dashboard 分支

1. 测试 RAGAS report、coverage、partial 和配置诊断；
2. 接入 RAGAS 专用 runner/history；
3. 显示版本、指标和逐题错误；
4. 回归证明其他后端的 service/page 行为未变化。

### 切片 7：真实验收与基线

1. 运行 adapter-only，确认 import、结构化输出和 embedding；
2. 建立 5–10 条真实 RAGAS Golden cases；
3. 运行纯 RAGAS end-to-end；
4. 扩展到首批 30–50 条并运行三次；
5. 记录成本、波动和门禁，不改 Custom baseline。

## 20. 文件变更清单

| 文件 | 允许的 RAGAS 变更 |
| --- | --- |
| `pyproject.toml` / `uv.lock` | 固定 RAGAS 0.4 可导入组合 |
| `config/default.yaml` | 仅追加 `evaluation.ragas` |
| `.env.example` | 仅追加 RAGAS judge/embedding 变量 |
| `src/paper_rag/config.py` | 仅增加 `_EvaluationRagas` 和 `ragas` 字段 |
| `evaluation/ragas.py` | 现代 metrics、async batch、错误处理 |
| `evaluation/ragas_models.py` | 新增 judge/embedding 工厂 |
| `evaluation/ragas_schema.py` | 新增 RAGAS 专用 schema/report |
| `evaluation/ragas_runner.py` | 新增 RAGAS 专用 runner |
| `scripts/evaluate.py` | 仅调整 `backend=ragas` 分支 |
| `scripts/accept_ragas_evaluation.py` | 新增真实 RAGAS 验收 |
| Dashboard service/page | 仅调整 RAGAS 分支和 RAGAS 展示 |
| `tests/fixtures/evaluation/ragas_*.json` | 新增 RAGAS 专用 fixtures |
| `tests/test_evaluation_ragas*.py` | 新增 RAGAS 测试 |

以下核心文件明确不在变更清单：`custom.py`、`retrieval.py`、`base.py`、`runner.py`、
`composite.py`、`scripts/accept_evaluation.py` 和现有 Custom/retrieval Golden Sets。

## 21. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| RAGAS 侵入 Custom 核心 | 独立 schema/runner/report；Custom 文件列入禁止修改清单 |
| 依赖漂移破坏基础环境 | lazy import、精确锁定、extra import contract、Custom 回归 |
| 模型不支持结构化输出 | adapter-only 真实验收，不按模型名推断 |
| 跳过失败样本抬高均分 | eligible count + coverage + coverage 门禁 |
| LLM judge 波动 | 固定版本和 options，三次校准后设置容差 |
| 成本和限流不可控 | batch、semaphore、cache、nightly 执行 |
| 中文/学术评分偏差 | 多语 judge/embedding、按 tag/语言聚合、人工抽检 |
| reference 质量差 | reference chunk 绑定和独立审核 |
| secret 泄漏 | 脱敏模型、日志/report 测试、Dashboard 不接收 key |
| QA cache 没有 contexts | RAGAS runner 禁用 cache 或按固定 chunk IDs 回捞 |

## 22. 回滚策略

1. 默认仍由现有 Custom 配置和路径运行；
2. RAGAS 失败只影响 `backend=ragas`，不改变其他后端；
3. 删除新增 RAGAS runner/schema/report 即可退出，不需要迁移 Custom 数据；
4. RAGAS report 使用独立 schema 和文件，不覆盖 Custom 历史；
5. 依赖失败时回退到已验证的 RAGAS 0.4.3 组合；
6. `data/evaluation/ragas_*` cache/report 可删除，不影响索引或 Custom Golden Set。

## 23. 完成定义

全部满足后才能标记完成：

1. evaluation-extra 环境能真实导入 RAGAS 0.4 collection metrics；
2. 五种指标有成功、缺输入、异常、非法数值和并发测试；
3. RAGAS 使用独立 Golden Set、runner 和 report；
4. RAGAS run 对每个 case 只执行一次 QA；
5. report 包含 coverage、模型版本、Golden Set 和 corpus 哈希；
6. CLI 和 Dashboard 的 RAGAS 分支能运行和展示结果；
7. adapter-only 真实验收通过；
8. 至少 5–10 条真实样本的纯 RAGAS end-to-end 通过；
9. 完整 pytest、Ruff check 和 format check 通过；
10. Custom、retrieval、composite 现有测试和 `accept_evaluation.py` 原样通过；
11. `git diff` 确认禁止修改的 Custom 核心文件没有变更；
12. 报告、日志和 Dashboard 不包含 API key。

## 24. 参考资料

- RAGAS RAG evaluation：<https://docs.ragas.io/en/stable/getstarted/rag_eval/>
- RAGAS OpenAI-compatible quickstart：<https://docs.ragas.io/en/stable/getstarted/quickstart/>
- RAGAS 0.3 到 0.4 迁移：
  <https://docs.ragas.io/en/latest/howtos/migrations/migrate_from_v03_to_v04/>
- RAGAS available metrics：
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
