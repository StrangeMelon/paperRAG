# Paper RAG Agent 重写学习课程

## 目标

在本目录从零重建当前 `paper-rag-agent-main` 的全部后端能力，排除 DeerFlow
前端及与 Paper RAG 无关的 DeerFlow 上游后端。最终实现使用相同依赖、模型、
配置、提示词、检索流程、引用约束和评测集，并达到不低于基准仓库的质量门槛。

本课程不依赖模型记忆。每次继续之前，先阅读 `LEARNING_STATE.md` 和
`SOURCE_MANIFEST.md`；每次结束时更新学习状态和验证证据。

## 范围

包含：

- `src/paper_rag/` 的全部 Python 模块。
- 根目录的 Python 打包、配置、脚本、Docker、运维与评测资产。
- `tests/` 与 `tests/eval/` 的质量门禁。
- DeerFlow 中专属于 Paper RAG 的 Harness Tool、`paper-research` subagent、Gateway
  router、认证/流式/指标集成及对应测试。

不逐文件重建：

- `integrations/deer-flow/frontend/`。
- DeerFlow 后端中与 Paper RAG 无关的上游功能。该后端作为复制进来的外部运行时
  依赖，但其项目相关改动必须完整理解、重写和验证。

## 固定学习协议

1. 每次只处理一个源文件或一个紧密绑定的非代码配置文件，绝不一次输出一个模块。
2. 先说明该文件解决的需求、它的输入输出契约、依赖和替代设计，再给出与基准一致的代码。
3. 先运行或阅读对应测试，随后写入代码，运行聚焦验证，再记录结果。
4. 每个阶段结束运行阶段测试，并把目标文件与基准仓库作 diff 或 SHA-256 校验。
5. 外部 LLM 的自然语言不可逐字稳定复现；等价性以 prompt、参数、检索证据、citation、
   trace 分支、API schema 和评测门禁为准。
6. 每完成一个可运行阶段建立一次 Git 提交。提交前不覆盖用户的本地修改。

## 阶段路线

### P0：课程控制面

建立本课程文件、源清单、状态记录和 Git checkpoint 规则。完成后进入项目文件重建。

### P1：工程骨架与配置

依次重建 `pyproject.toml`、`.env.example`、`Makefile`、Docker/Compose 资产和
`config/`。学习 Python extras、环境变量覆盖、Pydantic Settings、配置优先级。

### P2：包基础与领域契约

重建包入口、`config.py`、`utils/`、`ingest/schema.py` 与 ID/去重规则。产出稳定的
Paper、ParsedDocument、Chunk 等数据契约。

### P3：双存储

重建 SQLite/SQLModel 与 Qdrant 的存储适配器、初始化脚本和数据表/collection 契约。
对应 ADR-0004、ADR-0019。

### P4：采集、解析和切块

从 arXiv、URL、本地 PDF、OpenAlex、Semantic Scholar 获取论文；以 MinerU 为主、
PyMuPDF 为降级；完成 section-aware、contextual、多模态 chunk。

### P5：嵌入与入库状态机

接入 BGE-M3，重建 ingest pipeline 与单篇/批量/重建索引脚本，完成从 `created` 到
`done` 的离线闭环。

### P6：混合检索

重建 dense、BM25、FTS5、RRF、rerank、格式化与降级路径，先用纯逻辑测试锁定结果。

### P7：基础 QA 与 Agentic QA

先完成 simple QA，再增量加入 LLM client、intent、rewrite、HyDE、reflect、evidence
selection、citation check、abstain、cache、history、research memory、async 和 stream。

### P8：质量、可观测与工具接口

重建 trace、metrics、chaos、性能、retrieval/citation/claim eval；完成 search、section、
qa、compare、index、BibTeX 等 Python 工具接口。

### P9：Discovery 与 Wiki

实现论文发现候选闭环和 self-evolving Wiki，保持“候选 metadata 不是最终证据”与
patch-only、限频、版本化等安全边界。

### P10：视觉、交付物、反馈与主动 Agent

依次完成 figure/table vision enrichment、Markdown/PPTX/DOCX/LaTeX/PDF 交付物、
反馈与 hard case、订阅/inbox/digest/stale scan/auto-ingest。

### P11：DeerFlow 后端集成

将完整 DeerFlow 后端作为依赖复制；只重建 Paper RAG 的 LangChain Tool wrappers、
paper-research subagent、Gateway routers、user_id 边界、SSE、指标及测试。

### P12：部署与最终等价验收

完成 Qdrant Docker、可选 observability stack、迁移、备份恢复、secret scan、MinerU
doctor、DeerFlow smoke test。用同一论文集、配置、命令和评测门禁比较两套项目。

## 完成标准

- 所有 `SOURCE_MANIFEST.md` 中的项目文件均标记为完成并通过内容校验。
- `make lint`、纯逻辑测试、核心服务测试、DeerFlow Paper RAG 集成测试通过。
- Golden retrieval、citation audit、claim eval 和 Gateway/DeerFlow smoke 结果不低于基准。
- 全流程部署后能够 ingest、检索、QA、引用、discovery、wiki、deliver、feedback、
  proactive，并在 DeerFlow 后端调用。
