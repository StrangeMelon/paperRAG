# paper_retrieve_evidence 功能与技术设计规格

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 文档状态 | Draft |
| 目标项目 | `paper-rag-agent-rebuild` |
| 目标能力 | Paper RAG MCP Server 核心证据检索工具 |
| MCP Tool | `paper_retrieve_evidence` |
| 主要调用方 | DeerFlow |
| 最终回答生成方 | DeerFlow 中的 LLM |

## 2. 开发目标

在 `/home/user_kyh/paper-rag-agent-rebuild` 中实现异步 MCP 工具：

```text
paper_retrieve_evidence
```

该工具接收自包含的自然语言查询，负责执行：

```text
严格证据范围校验
-> 意图识别
-> Wiki 术语预解析
-> 查询改写
-> Dense + Sparse 混合检索
-> RRF 融合
-> Cross-encoder Rerank
-> 检索充分性反思
-> 必要时迭代检索
-> 拒答判断
-> 证据选择
-> 关联 Wiki 补充
-> 返回最小结构化证据包
```

工具不生成最终答案。最终回答由 DeerFlow 中的 LLM 生成。

## 3. 设计原则

1. MCP Server 负责检索质量、证据边界和来源追踪。
2. DeerFlow 负责最终语言生成和用户交互。
3. 论文 chunk 是可引用证据，Wiki 只能作为理解背景。
4. 生成回答所必需的信息应在一次工具调用中返回，不能依赖 DeerFlow 再次主动调用 `wiki_lookup`。
5. `paper_ids` 是证据范围约束，不是可被放宽的普通过滤条件。
6. 基础设施故障与确实没有证据必须严格区分。
7. 模型生成所需数据与可观测性数据必须走不同通道。
8. 默认响应必须控制 token 占用，不返回内部检索细节。
9. MCP 适配层保持轻薄，检索编排位于独立领域服务中。
10. 系统目标是有界并发和资源保护，而不是无限并行。

## 4. 非目标

本工具不负责：

- 生成最终回答。
- 校验尚未生成的回答引用。
- 从外部学术平台发现或下载论文。
- 修改、创建或合并 Wiki 词条。
- 维护 DeerFlow 对话历史。
- 将 Wiki 内容作为事实证据。
- 返回全部候选 chunk、完整内部 prompt 或完整 trace。
- 自动修正、忽略或放宽用户提供的论文证据范围。

## 5. 总体架构

```text
DeerFlow 并发请求
        |
        | MCP tools/call
        v
异步 MCP Adapter
        |
        +-- 参数与 paper_ids 严格校验
        +-- 请求准入、限流和分层超时
        |
        v
Evidence Retrieval Service（同步领域服务）
        |
        +-- Wiki query context
        +-- Intent / Rewrite / Reflect
        +-- Hybrid / Rerank / Abstain
        +-- Evidence selection
        +-- Evidence Wiki enrichment
        |
        +-- Public result ----------> DeerFlow
        |
        +-- Internal trace ---------> TTL Trace Store / Logs
```

MCP 层只负责协议、调度、错误映射和结果裁剪，不包含检索算法。

## 6. MCP Tool 定义

工具名称：

```text
paper_retrieve_evidence
```

建议工具描述：

> Retrieve a compact, citation-ready evidence set from indexed academic papers. The tool performs query understanding, rewriting, hybrid retrieval, reranking, sufficiency reflection, abstention, and Wiki background enrichment. It returns evidence but does not generate the final answer. Wiki content is background only and must not be cited as paper evidence.

从用户数据视角，该工具为只读工具。内部允许写入日志、指标、短期 trace 和去重后的复核信号，但不得修改论文内容或 Wiki 词条。

## 7. 输入契约

### 7.1 Pydantic 模型

```python
class RetrieveEvidenceInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] | None = Field(default=None, max_length=20)
    max_evidence: int = Field(default=4, ge=1, le=8)
    include_wiki: bool = True
    wiki_max_entries: int = Field(default=3, ge=0, le=5)
```

### 7.2 字段说明

| 字段 | 类型 | 默认值 | 约束 |
|---|---|---:|---|
| `query` | string | 必填 | 1 到 2000 字符，必须是自包含问题 |
| `paper_ids` | string array 或 null | null | 最多 20 个 |
| `max_evidence` | integer | 4 | 1 到 8 |
| `include_wiki` | boolean | true | 是否自动附带关联 Wiki |
| `wiki_max_entries` | integer | 3 | 0 到 5 |

不向 DeerFlow 暴露 `top_k` 和 `max_iter`。这两个参数由意图识别与服务端配置决定，避免 Host 破坏检索策略。

`user_id` 或 `tenant_id` 不允许由 LLM 作为工具参数传入，必须从 MCP 调用上下文获得。

MCP Server 不保存对话记忆。对于“它使用了什么方法”一类指代问题，DeerFlow 应结合对话上下文生成自包含查询后再调用工具。

## 8. paper_ids 严格证据范围

`paper_ids` 表达用户要求的证据范围，必须采用原子化严格校验。

### 8.1 语义规则

- `paper_ids=None`：允许搜索调用者可访问的整个论文库。
- `paper_ids=[]`：无效输入，直接返回 MCP Tool Error。
- `paper_ids` 为非空数组：所有 ID 必须存在、可访问且已完成索引。
- 只要一个 ID 不满足要求，整个调用失败。
- 禁止忽略错误 ID。
- 禁止只使用剩余有效 ID 继续检索。
- 禁止自动退回全库检索。
- 重复 ID 可以在全部校验通过后按首次出现顺序去重。

论文存在但状态为 `failed`、`parsed`、`chunked`、`embedded` 或 `indexed` 时，尚不能作为稳定检索范围使用，应返回 Tool Error，而不是返回 `no_evidence`。

### 8.2 建议错误

```json
{
  "code": "invalid_paper_scope",
  "message": "One or more paper IDs do not exist, are inaccessible, or are not indexed.",
  "paper_ids": ["arxiv:wrong-id"]
}
```

在多租户环境下，不区分“不存在”和“无权访问”，统一报告 `invalid_or_inaccessible`，避免通过错误信息枚举其他用户的论文。

### 8.3 实现接口

建议在 `sqlite_store.py` 增加批量接口：

```python
def get_papers_by_ids(paper_ids: list[str]) -> list[Paper]:
    ...
```

领域层实现：

```python
def validate_paper_scope(
    paper_ids: list[str] | None,
    *,
    principal: Principal,
) -> list[str] | None:
    ...
```

校验必须发生在 Embedding、Qdrant 查询和内部 LLM 调用之前。

## 9. 最小输出契约

### 9.1 有证据响应

```json
{
  "decision": "confident",
  "retrieval_id": "r_8f31ab",
  "evidence": [
    {
      "citation": "[chunk:abc123]",
      "paper_id": "arxiv:2402.00789",
      "title": "Graph-Mamba",
      "section": "Method",
      "page": 6,
      "modality": "text",
      "text": "Graph-Mamba uses..."
    }
  ],
  "wiki": [
    {
      "name": "State Space Model",
      "definition": "..."
    }
  ]
}
```

### 9.2 弱证据响应

```json
{
  "decision": "weak_evidence",
  "retrieval_id": "r_a194bc",
  "evidence": [
    {
      "citation": "[chunk:def456]",
      "paper_id": "arxiv:2402.00789",
      "title": "Graph-Mamba",
      "section": "Experiments",
      "page": 9,
      "modality": "table",
      "text": "..."
    }
  ],
  "wiki": []
}
```

### 9.3 拒答响应

```json
{
  "decision": "no_evidence",
  "retrieval_id": "r_2c17de",
  "evidence": []
}
```

拒答响应保留 `retrieval_id`，以支持拒答审计和 `paper_get_retrieval_trace` 排障。它不返回 `wiki`、候选 chunk、分数或拒答解释。

### 9.4 Pydantic 模型

```python
class EvidenceItem(BaseModel):
    citation: str
    paper_id: str
    title: str
    section: str | None
    page: int | None
    modality: str
    text: str


class WikiItem(BaseModel):
    name: str
    definition: str


class RetrieveEvidenceSuccess(BaseModel):
    decision: Literal["confident", "weak_evidence"]
    retrieval_id: str
    evidence: list[EvidenceItem]
    wiki: list[WikiItem]


class RetrieveEvidenceAbstained(BaseModel):
    decision: Literal["no_evidence"]
    retrieval_id: str
    evidence: list[EvidenceItem] = []
```

序列化使用 `exclude_none=True`。

## 10. 默认响应禁止返回的数据

以下数据仅进入内部 trace，不进入 DeerFlow 默认上下文：

- 意图分类结果及参数。
- 查询改写列表和 HyDE 内容。
- 每轮检索查询。
- Dense、BM25、RRF、rerank 原始分数。
- 候选 chunk 全集。
- 每轮反思结果。
- Evidence selection 逐候选打分。
- Wiki 候选和关联排序。
- 各阶段耗时。
- 内部 LLM 调用次数。
- 降级组件和内部 warning。

`evidence` 中也不返回检索分数。分数对 DeerFlow 生成答案没有直接价值，并会增加上下文噪声。

## 11. 内部领域模型

领域服务返回比 MCP 公共响应更完整的内部结果：

```python
@dataclass
class RetrievalExecution:
    retrieval_id: str
    public_decision: str
    evidence_chunks: list[dict]
    wiki_entries: list[dict]
    allowed_chunk_ids: list[str]
    trace: dict[str, Any]
```

MCP Adapter 从 `RetrievalExecution` 构造最小响应，完整 trace 写入 Trace Store。

## 12. 检索领域服务

公开同步接口：

```python
def retrieve_evidence(
    query: str,
    *,
    paper_ids: list[str] | None,
    max_evidence: int,
    include_wiki: bool,
    wiki_max_entries: int,
    principal: Principal,
) -> RetrievalExecution:
    ...
```

执行阶段：

1. 生成 `retrieval_id`。
2. 严格校验 `paper_ids`。
3. 检测查询语言。
4. 解析检索前 Wiki context。
5. 执行意图分类。
6. 计算内部 `top_k` 和 `max_iter`。
7. 执行多轮 retrieve + reflect。
8. 聚合并去重最终候选。
9. 执行 abstain。
10. 执行 evidence selection。
11. 关联最终 evidence 对应 Wiki。
12. 构造完整 trace。
13. 返回内部领域结果。

## 13. 与现有 qa_agentic 的关系

当前 `rag/qa_agentic.py` 已包含大部分检索流程，不能复制出一份 MCP 专用实现。

应将以下逻辑迁移到 `rag/evidence_retrieval.py`：

- 检索循环。
- Intent 调度。
- Final chunks 聚合。
- Abstain。
- Evidence selection。
- Wiki context 解析。

改造后：

```text
qa_agentic.answer()
  -> retrieve_evidence()
  -> 构造回答 prompt
  -> 调用回答 LLM
  -> citation_check
```

```text
MCP paper_retrieve_evidence()
  -> retrieve_evidence()
  -> 写入 trace
  -> 最小序列化
```

CLI、现有 QA 和 MCP 因此共用同一检索实现，避免行为分叉。

## 14. 公共 Decision 映射

内部决策：

```text
no_chunks
no_evidence
weak_evidence
confident
```

MCP 公共决策：

```text
no_chunks     -> no_evidence
no_evidence   -> no_evidence
weak_evidence -> weak_evidence
confident      -> confident
```

当内部结果为 `no_chunks` 或 `no_evidence`：

- 公共 `evidence=[]`。
- 不返回低相关候选。
- 不执行 Wiki 后置关联。
- Trace 保留候选和真实拒答原因。

基础设施故障不能映射为 `no_evidence`。

## 15. Wiki 预解析与后置关联

### 15.1 检索前 Wiki

检索前 Wiki 用于识别：

- 概念别名。
- 中英文名称。
- 缩写。
- 变体。
- 关键论文。

这些内容只参与查询改写和召回，不进入最终 evidence。

### 15.2 检索后 Wiki

建议在 `wiki/context.py` 增加：

```python
def resolve_evidence_wiki_context(
    question: str,
    evidence_chunks: list[dict],
    *,
    max_entries: int = 3,
) -> dict[str, Any]:
    ...
```

候选来源依次为：

1. Wiki 词条直接关联 evidence 的 `chunk_id`。
2. Wiki 词条关键论文包含 evidence 的 `paper_id`。
3. Wiki label 与查询精确匹配。
4. Qdrant Wiki 语义召回。

排序采用分层优先级，不在第一版引入未经标定的混合权重。

返回 DeerFlow 前必须：

- 解析 `merged_into` 重定向。
- 按 `entry_id` 去重。
- 限制条目数量。
- 剥离定义中的 `[chunk:...]`。
- 每条只保留 `name` 和 `definition`。
- 对超长定义执行字符预算裁剪。
- Wiki 异常时返回 `wiki=[]`，不影响论文 evidence。

工具描述和 DeerFlow 的 paper-research prompt 必须明确：

> `wiki` is background context only. Factual claims must cite tokens from `evidence`.

## 16. 上下文预算

默认限制：

- 最多 4 个 evidence。
- 每篇论文最多 2 个 evidence。
- 最多 3 个 Wiki 词条。
- Wiki 定义设置单条字符预算。
- 总响应设置序列化字符或 token 预算。

预算不足时按以下顺序处理：

1. 删除最低优先级 Wiki。
2. 删除最低优先级 evidence。
3. 单个超长 chunk 按安全边界裁剪，并在内部 trace 记录裁剪行为。

不能为了保留 Wiki 而裁掉更高优先级的论文 evidence。

## 17. Trace Store

### 17.1 目的

完整 trace 与模型生成数据分离，用于：

- 故障排查。
- 拒答审计。
- 性能分析。
- 后续引用校验。
- 评测数据采集。

### 17.2 记录结构

```json
{
  "retrieval_id": "r_8f31ab",
  "tenant_id": "user_x",
  "query": "...",
  "paper_scope": ["arxiv:2402.00789"],
  "intent": {},
  "rewrites": [],
  "iterations": [],
  "candidate_scores": [],
  "abstain": {},
  "evidence_chunk_ids": ["abc123"],
  "wiki_matches": [],
  "timings": {},
  "degraded_components": [],
  "created_at": "...",
  "expires_at": "..."
}
```

### 17.3 接口

```python
class RetrievalTraceStore:
    def put(self, execution: RetrievalExecution, principal: Principal) -> None:
        ...

    def get(self, retrieval_id: str, principal: Principal) -> dict:
        ...

    def purge_expired(self) -> int:
        ...
```

第一版使用线程安全的进程内 TTL/LRU：

- 使用 `threading.RLock`。
- 最大记录数配置化。
- TTL 配置化。
- 超限淘汰最旧记录。
- 每次 `put/get` 顺便清理过期记录。
- Trace 与 tenant/user 绑定。
- 不记录 API Key、内部系统 prompt 或绝对路径。
- 保存 `allowed_chunk_ids`，供后续 `paper_validate_citations` 使用。

Server 重启导致记录丢失是第一版允许的行为。此时引用校验返回 `retrieval_expired`，不得降级为仅检查 chunk 是否存在。

## 18. 管理 Trace Tool

仅 admin profile 注册：

```text
paper_get_retrieval_trace(retrieval_id)
```

要求：

- 校验调用者权限。
- 校验 trace 所属 tenant。
- 不存在或过期返回 Tool Error。
- 默认 DeerFlow 配置不加载该工具。
- 返回前清理敏感字段和绝对路径。

仅在工具描述中写“不要自动调用”不够可靠。只要工具出现在 `tools/list`，LLM 就可能选择它，因此 admin 工具应从默认工具目录中物理移除。

## 19. 检索状态与记忆边界

Server 保存短期运行状态，但不保存对话记忆。

短期状态至少包含：

- `retrieval_id`。
- 调用者身份。
- 允许引用的 chunk IDs。
- 完整 trace。
- 创建和过期时间。

它服务于 `paper_get_retrieval_trace`、后续 `paper_validate_citations`、审计和诊断。

未来若部署多个 MCP Server 实例，需要将该状态迁移到 Redis 或共享存储。在单进程 GPU 部署下，第一版不引入分布式状态组件。

## 20. 异步接入与有界并发

```text
DeerFlow 并发请求
        |
        v
异步 MCP Server
        |
        +-- 请求准入控制
        |     +-- 全局并发上限
        |     +-- 有界等待队列
        |     +-- 排队超时
        |
        +-- 快速查询
        |     +-- SQLite 读取 limiter
        |     +-- anyio.to_thread.run_sync()
        |
        +-- 证据检索
        |     +-- LLM limiter
        |     +-- Embedding GPU semaphore
        |     +-- Reranker GPU semaphore
        |     +-- Qdrant/FTS5 limiter
        |
        +-- 论文入库任务
              +-- 持久化任务队列
              +-- MinerU semaphore
              +-- Vision semaphore
              +-- Embedding semaphore
              +-- SQLite writer limiter
```

核心规则：

- MCP Tool 入口全部使用 `async def`。
- 同步领域服务不得直接运行在事件循环。
- SQLite、Embedding、Reranker 等通过 `anyio.to_thread.run_sync()` 执行。
- 不为每个请求创建线程池。
- 使用进程级共享线程池和独立 CapacityLimiter。
- GPU 信号量必须覆盖真实模型执行周期。
- 请求取消不代表底层线程或 GPU 推理已经停止，不能提前释放 GPU 配额。
- MinerU 使用受控子进程，超时时可以终止子进程。
- 不通过增加 MCP 多进程数量提升 GPU 吞吐。

## 21. MCP Runtime

`runtime.py` 负责：

```python
class McpRuntime:
    request_limiter
    retrieval_limiter
    thread_limiter
    admission_timeout
    retrieval_timeout
    trace_store
    resource_guards
```

MCP 工具入口伪代码：

```python
async def paper_retrieve_evidence(args: RetrieveEvidenceInput) -> dict:
    async with runtime.admit_retrieval():
        execution = await anyio.to_thread.run_sync(
            partial(retrieve_evidence, ...),
            limiter=runtime.thread_limiter,
        )
        runtime.trace_store.put(execution, principal)
        return build_public_response(execution)
```

在进入线程池前能够完成的输入校验应尽早执行。依赖 SQLite 的证据范围校验在线程中完成。

## 22. 资源保护

`resource_guards.py` 提供进程级同步信号量：

```text
gpu_total
embedding
reranker
vision
mineru
llm
sqlite_write
external_http
```

同步领域代码在线程或后台 worker 内获取信号量，因此不会阻塞事件循环。

固定获取顺序：

```text
gpu_total -> component semaphore
```

所有调用必须按同一顺序获取，避免死锁。

建议第一版保守配置：

- `gpu_total=1`
- `embedding=1`
- `reranker=1`
- `mineru=1`
- `sqlite_write=1`
- 其他值根据硬件和外部服务限流配置

模型实例保持进程内单例。不得通过启动多个 MCP worker 重复加载 BGE、Reranker 或 Vision 模型。

## 23. 请求准入与过载

Runtime 维护：

- 最大并发执行数。
- 最大等待请求数。
- 排队超时。
- 近期平均耗时。

系统过载时返回 MCP Tool Error：

```json
{
  "code": "busy",
  "message": "Paper retrieval service is at capacity.",
  "retry_after": 5
}
```

禁止：

- 无限等待信号量。
- 无限扩张线程池。
- 过载时继续接收所有请求。
- 将 `busy` 映射为 `no_evidence`。

## 24. 分层超时

至少配置以下超时：

| 超时 | 作用 |
|---|---|
| `admission_timeout` | 等待进入系统的最长时间 |
| `sqlite_timeout` | SQLite 操作 |
| `qdrant_timeout` | 向量数据库请求 |
| `external_http_timeout` | arXiv、PDF 和元数据请求 |
| `llm_timeout` | 意图、改写、反思 |
| `embedding_timeout` | 查询嵌入 |
| `reranker_timeout` | 重排 |
| `retrieval_total_timeout` | 一次完整检索总预算 |
| `ingest_stage_timeout` | 入库各阶段 |
| `mineru_timeout` | PDF 解析子进程 |
| `vision_timeout` | 单次视觉请求 |

达到总超时后，不再开始新的检索轮次。但不能假设 Python 线程可以被强制终止，已经运行的模型调用仍需占用资源许可，直到真实结束。

## 25. 建议配置

在 `config/default.yaml` 增加：

```yaml
mcp:
  profile: default
  admission_timeout_sec: 2
  retrieval_timeout_sec: 90
  trace_ttl_sec: 1800
  trace_max_entries: 1000
  max_running_retrievals: 2
  max_queued_retrievals: 8
  thread_tokens: 8

  resources:
    gpu_total: 1
    embedding: 1
    reranker: 1
    vision: 2
    mineru: 1
    llm: 4
    sqlite_write: 1

  timeouts:
    sqlite_sec: 5
    qdrant_sec: 15
    llm_sec: 30
    embedding_sec: 30
    reranker_sec: 30
    external_http_sec: 60
```

这些数值是保守初始值，必须通过真实 GPU、外部端点和并发测试校准。

## 26. 错误体系

定义领域异常：

```text
InvalidPaperScopeError
PermissionDeniedError
RetrievalBusyError
RetrievalTimeoutError
EmbeddingUnavailableError
StoreUnavailableError
RetrievalUnavailableError
RetrievalExpiredError
```

MCP Tool Error 格式：

```json
{
  "code": "invalid_paper_scope",
  "message": "...",
  "details": {}
}
```

正常响应中不包含 `error` 或 `warnings` 字段。

映射规则：

| 场景 | 结果 |
|---|---|
| 无相关证据 | 正常 `decision=no_evidence` |
| paper ID 无效 | Tool Error |
| 无访问权限 | Tool Error |
| Qdrant 和 Sparse 同时故障 | Tool Error |
| Wiki 故障 | 正常返回，`wiki=[]` |
| Reranker 故障但 RRF 可用 | 内部降级，正常返回 |
| 系统过载 | Tool Error `busy` |
| 总检索超时 | Tool Error `timeout` |

## 27. SQLite 策略

- 保持 WAL。
- 读取允许有限并发。
- 写入通过独立 limiter 控制。
- 长事务禁止包含 Embedding、LLM、MinerU 或网络调用。
- Trace 必须按 TTL 清理。
- 任务状态更新采用短事务。
- 同一论文的索引替换必须保持明确状态，避免检索到半成品。
- 同一 `paper_id` 禁止并发重建。

## 28. 论文入库队列接口约束

虽然本规格聚焦检索工具，但 MCP Runtime 必须为后续 `paper_ingest` 预留统一资源治理能力。

`paper_ingest` 只负责：

1. 参数和权限校验。
2. 计算来源请求的幂等键。
3. 检查同一论文是否已有活动任务。
4. 创建持久化任务。
5. 返回 `job_id`。

```json
{
  "job_id": "ing_741ac2",
  "status": "queued"
}
```

后台 worker 执行：

```text
fetch
-> parse
-> chunk
-> vision
-> embed
-> SQLite/Qdrant/FTS5 index
-> enqueue wiki
-> done
```

同一 `paper_id` 只能存在一个 `queued/running` 任务。该约束应同时由数据库唯一约束和进程内锁保证。

## 29. MCP Server 注册

Default profile：

```text
paper_search
paper_retrieve_evidence
wiki_lookup
paper_get_section
paper_get_visual
```

Admin profile：

```text
default profile tools
paper_get_retrieval_trace
```

`pyproject.toml` 增加：

```toml
[project.optional-dependencies]
mcp = ["mcp>=..."]

[project.scripts]
paper-rag-mcp = "paper_rag.mcp.server:main"
```

第一版向 DeerFlow 提供 `stdio` transport，并保持单进程运行，以共享 GPU 模型和进程级资源限制。

## 30. 建议文件结构

```text
src/paper_rag/
├── rag/
│   └── evidence_retrieval.py
├── wiki/
│   └── context.py
└── mcp/
    ├── __init__.py
    ├── server.py
    ├── schemas.py
    ├── errors.py
    ├── runtime.py
    ├── resource_guards.py
    ├── trace_store.py
    └── tools/
        ├── __init__.py
        ├── retrieve_evidence.py
        └── retrieval_trace.py

tests/
├── test_evidence_retrieval.py
├── test_mcp_retrieve_evidence.py
├── test_mcp_runtime.py
├── test_mcp_trace_store.py
└── test_mcp_server.py

scripts/
└── demo_mcp_retrieve_evidence.py
```

## 31. 测试计划

### 31.1 严格范围

- `paper_ids=None` 允许全库。
- `paper_ids=[]` 返回 Tool Error。
- 一个正确 ID 正常检索。
- 正确和错误 ID 混合时整体失败。
- 不可访问 ID 整体失败。
- 未完成索引 ID 整体失败。
- 范围校验失败后不调用 Embedding、Qdrant 或 LLM。
- 任何错误都不会触发全库回退。

### 31.2 最小响应

- `confident` 只包含四个顶层字段。
- `weak_evidence` 只包含四个顶层字段。
- `no_evidence` 只包含 `decision`、`retrieval_id`、`evidence`。
- Evidence 不包含分数和内部路径。
- Wiki 只包含 `name`、`definition`。
- Trace、intent、rewrites 不出现在公共结果。

### 31.3 检索行为

- Intent 正确控制检索轮数。
- Rewrite 失败时使用原始查询。
- Dense/Sparse 正确融合。
- Reranker 失败时使用 RRF。
- Reflect 不足时触发下一轮。
- `no_evidence` 不泄漏低相关 chunk。
- Evidence 数量和单篇限额有效。
- 中英文和多模态查询有效。

### 31.4 Wiki

- Chunk 直接关联优先。
- Paper 关联次之。
- 标签和语义召回兜底。
- 重定向正确解析。
- Wiki 引用被剥离。
- Wiki 故障不影响 evidence。
- Wiki 不会占用 evidence 的引用令牌。

### 31.5 Trace

- 成功调用写入 trace。
- `no_evidence` 也写入 trace。
- 过期后不可读取。
- 不同 tenant 不能读取对方 trace。
- LRU 容量限制生效。
- Trace 不进入公共响应。

### 31.6 并发

- 并发请求不阻塞事件循环。
- 超过运行和排队上限返回 `busy`。
- GPU semaphore 限制真实并发。
- SQLite 写操作被串行保护。
- 超时不会提前释放仍在执行的 GPU 许可。
- 多次请求不会重复加载模型。

## 32. 真实验收

按照重建项目的强制验收协议：

1. MCP 边界测试。
2. 领域服务实现。
3. 真实 Qdrant、BGE-M3、Reranker、LLM Demo。
4. 无 mock 的 MCP 集成测试。
5. DeerFlow 实际连接并调用。

真实 Demo 至少验证：

- 全库中文查询。
- 指定单篇论文查询。
- 指定多篇论文查询。
- 中英双语查询。
- Wiki 自动附带。
- 域外问题返回 `no_evidence`。
- 错误 paper ID 返回 Tool Error。
- 正确与错误 paper ID 混合时整体失败。
- 两个并发请求不会阻塞 MCP Server。
- 输出不包含 trace 和候选分数。

## 33. 实施顺序

1. 增加 MCP 配置模型和错误类型。
2. 增加输入输出 Pydantic schema。
3. 实现严格 paper scope 批量查询和测试。
4. 抽取 `rag/evidence_retrieval.py`。
5. 让现有 `qa_agentic` 复用新领域服务。
6. 实现 evidence Wiki 反向关联。
7. 实现 Trace Store。
8. 实现 Resource Guards。
9. 实现异步 MCP Runtime 和准入控制。
10. 实现 MCP Tool Adapter。
11. 实现 admin trace tool/profile。
12. 增加 MCP console script。
13. 完成真实 Demo、集成测试和 DeerFlow 联调。

## 34. 第一阶段完成标准

第一阶段完成必须同时满足：

- DeerFlow 可以通过 MCP 发送自包含查询。
- DeerFlow 只收到最小 `decision + retrieval_id + evidence + wiki` 响应。
- Wiki 自动附带且明确不可引用。
- `paper_ids` 使用原子化严格校验。
- 错误输入使用 MCP Tool Error。
- `no_evidence` 不返回低相关候选。
- 完整检索过程仅能通过 admin trace 工具查看。
- 并发请求受到线程、GPU、队列和分层超时保护。
- 系统过载时返回 `busy + retry_after`，不无限排队。
- 现有 `qa_agentic` 与 MCP 共用同一检索领域服务。
