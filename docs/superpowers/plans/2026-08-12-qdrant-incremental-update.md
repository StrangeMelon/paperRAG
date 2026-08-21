# Qdrant 论文 Chunk 增量更新开发方案

- **日期**: 2026-08-12
- **状态**: implementation checkpoint（增量代码与两篇 PDF 验收已通过；未执行全量迁移）
- **范围**: `paper_chunks` collection、SQLite `chunk` 快照、论文入库流水线
- **关联代码**: `src/paper_rag/chunk/builder.py`、`src/paper_rag/store/sqlite_store.py`、
  `src/paper_rag/store/qdrant_store.py`、`src/paper_rag/store/ingest_pipeline.py`、
  `scripts/init_store.py`

## 1. 背景与问题

当前 Qdrant 入库路径在每次强制重建论文时执行：

```text
按 paper_id 删除全部旧 Point
→ 对全部新 Chunk 生成 embedding
→ 全量 upsert 新 Point
```

当前 `chunk_id` 是位置型 ID，由 `paper_id`、章节序号、模态和块序号组成。它适合
做引用和 SQLite 主键，但不能证明 chunk 内容没有变化：同一位置上的正文、标题前缀、
视觉摘要或 payload 都可能已经改变。

目标是保留现有 `chunk_id` 的外部语义，同时让 Qdrant 更新只处理真正变化的 Point。
更新操作暂时保持同步（`wait=True`），先保证入库状态和检索可见性的一致性。

## 2. 目标与非目标

### 2.1 目标

1. 为 `paper_chunks.paper_id` 创建可用于精确过滤的 Qdrant payload index。
2. 保留现有 `chunk_id`，新增内容、embedding 版本和 payload 指纹。
3. 使用确定性的 UUIDv5 Point ID，重复执行同一版本入库必须幂等。
4. 在 embedding 前完成差量分类，只对必要的 Chunk 生成向量。
5. 对新增、向量变化、payload-only 变化和删除分别执行最小操作。
6. 对现有整数 Point ID、缺少指纹的旧数据提供可重试迁移路径。
7. 同一 `paper_id` 的并发入库不能产生两个版本的混合结果。

### 2.2 非目标

- 不在本阶段修改 `chunk_id` 的算法或已有引用协议。
- 不在本阶段引入 `wait=False`、后台 operation worker 或最终一致性状态机。
- 不解决分块边界对前部插入的连锁变化；该问题属于后续稳定分块设计。
- 不改变 Qdrant collection 的向量维度、距离函数或 HNSW 参数迁移策略。

## 3. 设计原则

### 3.1 ID、内容和存储版本是三个不同概念

`chunk_id` 表示逻辑位置和外部引用；它不是内容版本。

`content_id` 表示 embedding 的实际输入内容。当前 embedding 输入是
`chunk["context_text"]`，因此 `content_id` 必须由该字符串的 UTF-8 字节计算，而不是
仅由裸 `text` 计算。

`embedding_version` 表示生成向量的协议版本，至少包括模型供应商、固定模型 revision、
向量维度、最大长度、pooling/归一化方式、上下文模板版本和 embedding 实现版本。
`batch_size`、设备类型等不改变语义时不应触发全量重嵌入；若数值精度被视为协议的一部分，
必须明确纳入版本定义。

`payload_fingerprint` 表示业务 payload 的规范化哈希，不包含 vector 和自身。它用于
区分“向量不变但展示、定位或 metadata 变化”的情况。

### 3.2 指纹必须基于实际存储契约

指纹不能依赖 Python `dict` 的自然顺序或不可序列化对象。规范化过程必须：

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

路径、枚举、整数、浮点、`None` 和列表顺序都要有明确的序列化规则。运行时字段（例如
更新时间、临时缓存路径）若不应触发更新，应在计算 fingerprint 前排除；若它们属于实际
对外 payload，则必须保留并接受其变化会触发 payload 更新。

### 3.3 Point ID 必须稳定且与现有位置 ID 对齐

使用项目内固定 namespace：

```text
point_id = UUIDv5(PROJECT_POINT_NAMESPACE, paper_id + "\0" + chunk_id)
```

namespace 一旦投入生产不得改变。UUIDv5 实际可用的信息位少于完整 128 位；由于现有
`chunk_id` 已截断为 80 位，UUIDv5 不能恢复被截断的熵。迁移实现仍应检查同一论文内的
重复 `chunk_id`，跨论文则在 UUID 输入中显式加入 `paper_id`。

## 4. 数据模型变更

### 4.1 SQLite `Chunk`

在 `src/paper_rag/store/sqlite_store.py` 的 `Chunk` 中新增：

```python
content_id: str | None = Field(default=None, index=True)
embedding_version: str | None = None
payload_fingerprint: str | None = None
```

建议旧数据迁移为可空字段。旧数据没有指纹时视为 dirty，首次处理该论文时强制同步；
不要把空字符串当作有效 fingerprint。

`_CHUNK_COLUMN_MIGRATIONS` 必须同时补充这三列。`upsert_sections_and_chunks()` 的
payload 映射、SQLite round-trip 测试和旧数据库启动迁移测试必须同步更新。

### 4.2 Qdrant payload

`upsert_chunks()` 写入的 payload 必须包含：

```text
chunk_id
paper_id
content_id
embedding_version
payload_fingerprint
以及现有检索、展示和溯源字段
```

向量不放进 payload。Qdrant 的旧 Point 如果缺少任一指纹字段，不能进入 skip 分支。

### 4.3 Qdrant payload index

`scripts/init_store.py` 的 `init_qdrant()` 需要拆分为：

1. collection 不存在时创建 collection。
2. 无论 collection 是否已经存在，都幂等创建：

```python
client.create_payload_index(
    collection_name=config.qdrant.collection_chunks,
    field_name="paper_id",
    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    wait=True,
)
```

已有 collection 的初始化不能因为“collection 已存在”而跳过 index。`paper_id` 是精确
字符串过滤字段，使用 keyword index；不需要为三个 fingerprint 建 index，因为它们只在
已经按 `paper_id` scroll 后由应用层比较。

## 5. 增量同步流程

### 5.1 阶段 A：解析、分块和计算指纹

1. 按当前流程解析 PDF、读取语言和构建 chunks。
2. 插入 metadata chunk。
3. 为每个 chunk 计算三个字段。
4. 计算预期 UUID Point ID。
5. 在本地检查：
   - `chunk_id` 不为空；
   - 同一论文内 `chunk_id` 唯一；
   - `content_id`、`embedding_version`、`payload_fingerprint` 非空；
   - chunks 数量与后续向量输入一一对应。

此时不要执行全量 embedding。

### 5.2 阶段 B：读取旧 Qdrant 快照

使用 `scroll()` 按 `paper_id` 过滤并分页读取。必须处理 scroll offset，不能依赖客户端
默认的少量 limit。只读取必要 payload，不读取 vectors：

```text
point_id
chunk_id
content_id
embedding_version
payload_fingerprint
```

旧快照建立以下映射：

```text
old_by_point_id
old_by_chunk_id
```

如果发现同一 `chunk_id` 对应多个旧 Point，保留预期 UUID 对应者，其余 Point 进入
`delete_ids`，避免历史失败产生的重复点永久存在。

### 5.3 阶段 C：分类

对于每个新 chunk，按以下优先级分类：

| 条件 | 分类 | 操作 |
|---|---|---|
| 预期 Point ID 不存在 | `vector_updates` | 生成 embedding 并 upsert |
| `content_id` 不同 | `vector_updates` | 生成 embedding 并 upsert |
| `embedding_version` 不同 | `vector_updates` | 生成 embedding 并 upsert |
| 以上相同，仅 payload 指纹不同 | `payload_updates` | 仅覆盖 payload |
| 三个指纹均相同 | `skipped` | 不操作 |

旧 Point 中没有对应新 `chunk_id` 的 Point ID 进入 `delete_ids`。旧整数 Point ID、缺失
指纹 Point 和错误 UUID Point 也必须进入迁移清理路径，而不能被误判为 unchanged。

### 5.4 阶段 D：只生成必要向量

将 `vector_updates` 按原顺序映射到待编码文本：

```python
texts = [chunk["context_text"] for chunk in vector_updates]
vectors = bge_m3.encode(texts) if texts else []
```

必须验证 `len(vectors) == len(vector_updates)`。未变化和 payload-only 的 chunk 不得进入
embedding 调用。

### 5.5 阶段 E：同步写入顺序

使用 `wait=True`，按以下顺序执行：

1. 对 `vector_updates` 批量 upsert，写入完整 payload 和新向量。
2. 对 `payload_updates` 使用 `overwrite_payload` 完整替换 payload，而不是合并式
   `set_payload`，避免新 payload 已删除的字段残留。
3. 对 `delete_ids` 按 Point ID 精确删除。

先 upsert、后删除可以在中途失败时优先保留新内容。所有操作都必须幂等；失败后重试同一
论文应得到同一最终状态。

当前 `delete_chunks_for_paper()` 会吞掉异常并继续流程。增量同步中，删除异常必须向上抛出
或将论文标记为 failed；不能在“删除失败但状态 done”的情况下继续运行。

### 5.6 阶段 F：SQLite、FTS5 和 Wiki 状态

SQLite 先保存新的 chunk 快照，但只有 Qdrant 同步成功后才能将论文标记为 `indexed` 和
`done`。失败时保留新快照并记录错误，下一次任务仍以 Qdrant 实际快照重新计算差量。

Qdrant 成功后执行现有的单篇 FTS5 同步。Wiki 任务内容指纹不能再只依赖排序后的
`chunk_id`，应使用：

```text
SHA-256(sorted((chunk_id, content_id) pairs))
```

否则位置型 `chunk_id` 不变时，正文变化不会触发 Wiki 重建。持久化 Wiki 证据还应保存
`content_id` 或论文版本，以避免旧证据 ID 静默重新绑定到新文本。

## 6. 并发、失败与一致性

### 6.1 论文级互斥

同一 `paper_id` 的两个入库任务不能同时执行“读取旧快照 → 计算差量 → 写入”。需要
跨线程、跨进程有效的论文级锁或数据库租约。单纯 Python 进程内 Lock 不足以覆盖多进程
worker。

### 6.2 部分失败

所有写操作都使用确定性 Point ID，因此重复重试不会产生新版本副本。部分失败场景：

- upsert 成功、delete 失败：旧 Point 可能暂时残留，重试会再次精确删除；论文不能标记 done。
- overwrite_payload 失败：向量可能已更新，重试会根据 payload 指纹补齐。
- scroll 失败：不得执行删除，直接失败并保留旧 Qdrant 数据。
- SQLite 成功、Qdrant 失败：状态为 failed，后续以 Qdrant 快照为准重试。

### 6.3 检索一致性

本方案阶段内保持 `wait=True`，Qdrant 写入返回后才允许将状态推进到 `done`。这不构成
跨 SQLite、FTS5、Qdrant 的事务；因此状态更新必须放在 Qdrant 和 FTS5 成功之后，并保留
可重试错误记录。

## 7. 旧数据迁移

迁移不要求一次性重建整个 collection，可以按论文渐进完成：

1. 先创建 `paper_id` payload index。
2. 新代码读取旧整数 Point 和无指纹 payload。
3. 将其视为 dirty，生成预期 UUID Point。
4. upsert UUID Point。
5. 精确删除旧整数 Point。
6. 写入三个新字段。
7. 成功后记录论文迁移完成。

如果迁移过程在第 4 或第 5 步中断，下一次仍能通过 Point ID 和 fingerprint 差量修复。
建议提供一个仅扫描 dirty 论文的迁移 CLI，而不是依赖普通检索请求触发隐式迁移。

## 8. 测试计划

### 8.1 单元测试

- `content_id` 对相同 `context_text` 稳定，对任意字符变化敏感。
- `embedding_version` 变化会进入 `vector_updates`。
- payload 变化但内容和 embedding 版本不变时，只进入 `payload_updates`。
- 固定 namespace 下 UUID Point ID 重复计算相同。
- 同一论文重复 `chunk_id` 被拒绝。
- payload fingerprint 序列化顺序稳定。
- 旧 payload 缺字段时不会进入 skip 分支。
- `old - new` 只产生精确 point ID 删除。

### 8.2 Qdrant store 测试

- 初始化对已存在 collection 仍创建 `paper_id` index。
- scroll 正确处理多页结果。
- upsert 使用 UUID Point ID、完整 payload 和 `wait=True`。
- payload-only 更新使用 overwrite 语义。
- 删除使用精确 Point ID 列表，而不是按 `paper_id` 全量删除。
- Qdrant 异常不会被静默吞掉并标记为 done。

### 8.3 入库流水线测试

至少覆盖以下五种重建：

1. 完全相同论文：不调用 embedding，不执行 Qdrant 写操作。
2. 新增 chunk：只 embedding 和 upsert 新 chunk。
3. 删除 chunk：只精确删除旧 Point。
4. 同一位置内容变化：只更新对应 Point，不删除整篇论文。
5. 标题、章节前缀或视觉摘要变化：`content_id` 变化并重新 embedding。

另外测试 embedding 失败、scroll 失败、删除失败和同一论文并发任务。

### 8.4 迁移与真实服务测试

- 旧整数 Point、缺字段 Point、重复 Point 混合存在时可完成迁移。
- Embedded Qdrant 和远程 Qdrant 各执行一次真实同步。
- 同步完成后按 `paper_id` 检索，确认不存在旧 chunk，且新 chunk 可检索。
- 运行完整测试、Ruff 和初始化脚本：

```bash
uv run pytest tests/test_qdrant_store.py tests/test_ingest_pipeline.py tests/test_sqlite_store.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## 9. 观测指标与验收标准

每次论文同步记录：

```text
old_points
new_chunks
vector_updates
payload_updates
skipped
deleted
embedding_count
qdrant_operation_count
elapsed_ms
```

验收标准：

- 相同论文重跑时 `embedding_count == 0`，且不会发生整篇删除。
- 论文内容变化后，Qdrant 中最终 Point 集合与新 chunk 集合完全一致。
- 任意失败重试后达到同一最终状态，无重复 Point、无旧 payload 残留。
- 既有引用的 `chunk_id` 仍可解析；Wiki 旧证据不会在内容变化后无标记地复用。
- 在目标规模数据上，更新成本主要随变化 chunk 数和论文解析成本增长，而不是随整个
  collection 的 Point 总数线性增长。

## 10. 外部参考

- [Qdrant Filtering](https://qdrant.tech/documentation/search/filtering/)
- [Qdrant Payload Indexing](https://qdrant.tech/documentation/manage-data/indexing/)
- [Qdrant Points](https://qdrant.tech/documentation/manage-data/points/)
- [Qdrant Consistency Guarantees](https://qdrant.tech/documentation/scaling/consistency-guarantees/)

## 11. 当前实现与验收记录

本阶段只完成必要的增量更新代码，不处理已有生产 collection 的全量迁移。旧整数
Point 和缺少指纹的迁移逻辑已在纯逻辑层保留设计边界，但尚未对正式库执行迁移。

已实现：

- `src/paper_rag/store/incremental.py`：内容/embedding/payload 指纹、UUIDv5 Point ID、
  差量计划。
- `Chunk.content_id`、`Chunk.embedding_version`、`Chunk.payload_fingerprint` 及旧 SQLite
  表的启动迁移。
- `paper_id` keyword payload index 的幂等初始化。
- Qdrant 分页 snapshot、向量 upsert、完整 payload 覆盖和精确 Point ID 删除。
- 入库前差量分类，embedding 仅处理 `vector_updates`。
- 批量入库报告的 parse、chunk、vision、embedding、incremental update 和 total 耗时。
- `scripts/accept_incremental_ingest.py` 真实双论文验收脚本。

真实验收命令：

```bash
.venv/bin/python scripts/accept_incremental_ingest.py
```

验收脚本使用 `demo-ingest-batch-data/pdfs` 下的两篇 PDF 和独立 embedded Qdrant/SQLite，
执行首轮完整入库与第二轮 `--force` 增量重跑，不修改现有 demo 或正式库。2026-08-12
验收结果：

| 轮次 | 论文 | chunks | embedding | vector_updates | skipped | deleted |
|---|---|---:|---:|---:|---:|---:|
| 首轮 | Graph-Mamba | 41 | 28.060s | 41 | 0 | 0 |
| 首轮 | 综合能源服务区块链 | 60 | 20.169s | 60 | 0 | 0 |
| 增量重跑 | Graph-Mamba | 41 | 0.001s | 0 | 41 | 0 |
| 增量重跑 | 综合能源服务区块链 | 60 | 0.001s | 0 | 60 | 0 |

2026-08-12 03:52 的重新验收中，Qdrant 与 SQLite 的最终 Chunk 数一致，脚本输出
`ACCEPTANCE PASSED`。真实验收环境中
vision 配置为 disabled，因此 vision 阶段耗时为 0；启用 vision 后，builder 会单独记录
`vision_seconds`。

下一阶段才执行旧 collection 的按论文迁移，并需要为迁移 CLI 增加独立的失败恢复和
断点验收；本阶段不将迁移失败混入普通入库流程。

## 12. 全能力验收

`scripts/accept_full_incremental_ingest.py` 是开启 MinerU 与 Vision 的严格验收入口：

```bash
.venv/bin/python scripts/accept_full_incremental_ingest.py
```

它使用独立的 `demo-full-incremental-update-data/`，强制以下配置：

```text
mineru.mode = local
mineru.method = ocr
mineru.fallback_to_pymupdf = false
vision.enabled = true
vision.cache = true
```

脚本会失败于 MinerU 降级、缺少 Vision 凭据、没有真实 figure/table 资产、任意 Vision
失败、阶段耗时缺失或 Qdrant/SQLite chunk 数不一致。它对每篇论文输出：

```text
parse_seconds
chunk_seconds
vision_seconds
embedding_seconds
incremental_update_seconds
total_seconds
vector_updates / payload_updates / skipped / deleted
```

MinerU OCR 和图表邻近上下文存在运行间差异，因此全能力重跑不能错误地以
`vector_updates == 0` 作为唯一成功条件。严格验收改为验证：每篇均有被跳过的 chunk，
`vector_updates < chunks`，分类数等于新 chunks 数，Vision 至少命中一项缓存，且未发生
Vision 失败。这证明变化 point 被局部更新，而非整篇删除重写。

2026-08-12 真实结果：

| 轮次 | 论文 | parser | vision | embedding | vector_updates | skipped |
|---|---|---|---:|---:|---:|---:|
| 首轮 | Graph-Mamba | mineru+complete | 21.784s | 3.590s | 49 | 0 |
| 首轮 | 综合能源服务区块链 | mineru+complete | 24.271s | 0.229s | 82 | 0 |
| 增量重跑 | Graph-Mamba | mineru+complete | 5.974s | 3.481s | 17 | 28 |
| 增量重跑 | 综合能源服务区块链 | mineru+complete | 20.511s | 0.167s | 29 | 52 |

该轮共有 29 个图表 chunk；首轮 Vision 成功 29 个，第二轮 15 个命中缓存、14 个因 OCR
输出或邻近上下文变化而重新摘要，所有请求成功。脚本输出 `FULL ACCEPTANCE PASSED`。
