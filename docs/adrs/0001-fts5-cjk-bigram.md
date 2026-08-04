# ADR-0001 · FTS5 稀疏检索的中文适配与同步机制

- **日期**: 2026-08-04
- **状态**: proposed（fts5 课验收通过后转 accepted）
- **关联**: 重建课程 P6 第二课 `src/paper_rag/retrieve/fts5.py`；
  全局约束"重建版必须支持中文论文"（LEARNING_STATE.md, 2026-08-01）

## 背景：我们碰到了什么问题

混合检索需要稀疏（关键词）这条腿来补稠密检索的短板（专有名词、缩写、数字、
罕见术语）。基准实现用 SQLite FTS5 做默认稀疏后端（`retrieve.sparse_backend:
fts5`）。动手重建前按课程约定实证核对基准行为，发现三个问题：

### 问题 1：unicode61 分词器把连续中文收为单一 token（致命）

基准 docstring 声称 CJK 会被 unicode61 逐字切分（per-character），实测不符。
用 `fts5vocab` 直读索引词表，存入
`图神经网络在长程依赖建模上的局限` 后索引里只有**一个**中文 token——整段原文：

```
['dependencies', 'graph', 'long', 'networks', 'neural', 'range',
 'struggle', 'with', '图神经网络在长程依赖建模上的局限']
```

后果：`长程依赖`、`图神经网络` 等子串查询 **0 命中**，只有查询串在文档里恰好
被标点切成独立完整 run 时才能靠巧合命中（真实中文期刊实测：字面含 `区块链`
的 26 个 chunk 中，基准索引仅命中 1 个——该块里有一处 "系统。区块链、物联网"
被句号/顿号恰好切出独立 token，召回率 4%）。中文论文在基准 FTS5 里的稀疏检索
能力**接近于零**；且 `hybrid.py` 出错回退
rank_bm25 的路径只在**抛异常**时触发，"合法地返回空"不会回退——中文混合检索
**静默退化**为纯稠密。

### 问题 2：docstring 承诺的 porter 词干器没有出现在建表语句里

`tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"` 中没有 `porter`。
实测：文档含 `dependencies`，查询 `depend` → 0 命中。英文侧漏召回。

### 问题 3：索引从不被填充（先入库、后首查必空）

`chunks_fts` 表与三个同步触发器**懒建**于第一次 `search()`；触发器只对创建之后
的 INSERT 生效。真实时序是 ingest 先写 `chunk` 表、检索后发生，故首查时镜像表
恒为空。全仓无任何调用 `reindex_all()` 的代码（scripts/store/tools 均无）。
稀疏腿在基准的真实部署时序下从未工作过。

## 决策：技术选型

### 选型 1：中文分词用 CJK bigram，索引存分词镜像

- 新增纯函数 `segment_cjk(text)`：连续 CJK 串展开为空格分隔的相邻二字对
  （`长程依赖建模` → `长程 程依 依赖 赖建 建模`），单字退化 unigram，
  英文/数字原样保留。**写入索引前**与**构造查询时**用同一函数，两侧自动对齐。
- `chunks_fts.text` 因此存的是分词镜像而非原文；`search()` JOIN `chunk` 表
  取回原文返回，上层看不到 bigram 串。
- 开关配置化：`retrieve.fts5_cjk_bigram: true`（default.yaml + 配置模型）。

**否决的备选**：
- jieba 词典分词——需新增依赖（违背"与基准同依赖"约束）；Python `sqlite3`
  不支持注册自定义 C tokenizer，只能预分词，词典对学术术语误切多，收益不稳。
- FTS5 `trigram` tokenizer——原生支持子串，但同时改变英文行为（三元组匹配、
  索引膨胀、分数体系不同），为修中文伤英文，不取。
- 逐字 unigram——与下一课 rank_bm25 后端的 `[一-鿿]` 逐字方案一致性
  最好，但单字 IDF 判别力弱、误召回高；bigram 是中文检索标准做法
  （Lucene CJKAnalyzer 同款）。记账：两个稀疏后端 zh 粒度不一致（fts5=bigram,
  rank_bm25=unigram），留待 hybrid/评测课用真实数据对比后统一。

### 选型 2：加 porter 词干器

`tokenize = "porter unicode61 remove_diacritics 2 tokenchars '_'"`。兑现基准
docstring 的既定意图，英文词形变化互相召回。词干 token 不可读无影响——原文
经 JOIN 返回，FTS5 列不对外。porter 只作用于 ASCII，中文路径不受影响。

### 选型 3：删触发器，Python 侧同步 + search() 自愈

- 删除基准的 AI/AU/AD 触发器：SQL 触发器无法调用 Python 分词函数，保留它们
  必然出现"半分词半原文"的混合索引状态。
- `fts5.py` 公开 `reindex_all()`（全量重建）与 `sync_paper(paper_id)`（单篇
  增量：删旧插新）。
- `search()` 内置自愈：索引行数 != `chunk` 表行数时自动 `reindex_all()`，
  修复问题 3 的"先入库后首查必空"，也覆盖历史存量数据。
- 记账的已知边界：行数相等但内容已变（原地 UPDATE）检测不到——当前入库
  流水线的替换语义是先删后插，行数必变，暂不构成风险。

## 后果

- 相对基准的结构性偏离共三处（分词镜像 + JOIN 回原文、porter、去触发器改
  Python 同步），全部记入测试断言与 LEARNING_STATE.md。
- 索引体积约增大 2 倍（bigram），对课程规模（10^1–10^2 篇）无感。
- 行为不再与基准逐字节可比：英文分数因 porter 改变、中文从 0 命中变为可用。
  等价标准按课程约定取"运行行为正确"而非"与基准同缺陷"。
