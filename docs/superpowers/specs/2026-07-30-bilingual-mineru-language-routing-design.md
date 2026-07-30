# MinerU 中英文 OCR 语言路由与降级设计

## 背景

论文集合同时包含中文和英文论文，其中一部分是没有文字层的纯扫描 PDF。纯扫描论文会由
人工在标准元数据中标注语言；普通 PDF 不要求人工标注。项目强制使用 MinerU OCR 和 CUDA，
但不能因为语言无法判断或单篇论文解析失败而终止整个批处理流程。

PaddleOCR 使用 `ch` 表示中英文通用 OCR 模型，使用 `en` 表示英文专用模型。领域元数据不应
暴露这个供应商约定，因此元数据使用 ISO 风格的 `zh` 和 `en`，只在 MinerU 适配边界完成
`zh -> ch` 映射。

## 目标

- 自动为每篇论文选择 `ch` 或 `en` OCR 模型。
- 将扫描论文的人工语言标记定义成可验证的一等元数据字段。
- 普通 PDF 在没有人工元数据时通过本地文字采样判断语言。
- 语言判断失败时稳定回退 `ch`，不抛出导致批处理终止的异常。
- 单篇 MinerU 失败时记录失败或降级状态，并继续处理下一篇论文。
- 同时下载和诊断中英文 OCR 权重，保证自动路由不会在运行时才暴露缺失模型。
- 保持已有 `ch` 或 `en` 全局强制模式，便于调试和特殊部署。

## 非目标

- 本阶段不支持中英文以外的 OCR 语言。
- 不调用 LLM 或外部语言检测 API。
- 不通过中英文各执行一次完整 OCR 再比较置信度。
- 不把 PyMuPDF 对扫描件产生的空文本当成成功解析结果。

## 元数据契约

`PaperMeta` 增加以下可选字段：

```python
language: Literal["zh", "en"] | None = None
```

人工标注写入论文目录下 `meta.json` 的顶层：

```json
{
  "paper_id": "local:example",
  "title": "示例论文",
  "language": "zh"
}
```

采集器重复采集同一论文时必须保留已有的非空 `language`，不能用来源返回的空值覆盖人工
标注。采集源未来返回可信语言时可以填充空字段，但不能覆盖人工值。该合并规则由统一元数据
持久化函数负责，避免五类采集器各自实现不同策略。

## 配置契约

应用层配置接受三个值：

```yaml
mineru:
  lang: auto  # auto | ch | en
```

- `auto`：按本文的语言决策链逐篇选择。
- `ch`：所有论文强制使用中英文通用模型。
- `en`：所有论文强制使用英文专用模型。

`auto` 是应用层值，不会原样传给 `magic-pdf`。执行 CLI 时只能传入 `-l ch` 或 `-l en`。

## 语言决策组件

新增独立模块 `src/paper_rag/parse/language.py`，提供一个不依赖 MinerU 的纯本地决策接口。
返回对象至少包含：

```python
OcrLanguageDecision(
    document_language="zh",
    mineru_language="ch",
    source="metadata",
    reason="valid_meta_language",
    model_fallback=False,
)
```

其中：

- `document_language` 为 `zh`、`en` 或无法确认时的 `None`。
- `mineru_language` 始终为可执行的 `ch` 或 `en`。
- `source` 为 `forced`、`metadata`、`pdf_text` 或 `fallback`。
- `reason` 保存稳定、可测试、可记录的决策原因。
- `model_fallback` 表示是否因为首选模型权重缺失而切换语言模型。

## 自动决策顺序

1. `mineru.lang` 为 `ch` 或 `en` 时直接返回强制结果。
2. 查找 PDF 同目录的 `meta.json`。顶层 `language` 为 `zh` 时映射到 `ch`，为 `en` 时
   映射到 `en`。
3. 元数据不存在、字段缺失或值无效时，用 PyMuPDF 从前五页提取文字，最多采样
   20,000 个字符。
4. 统计 CJK 统一表意字符和拉丁字母：CJK 至少 20 个且占两类字符总数的比例不低于
   5% 时判为 `zh -> ch`；否则拉丁字母至少 50 个时判为 `en -> en`。
5. 文本不足、PDF 无文字层、元数据损坏或 PyMuPDF 提取异常时，返回
   `fallback -> ch`，并保存具体 `reason`。

阈值用于避免英文论文中少量中文姓名或参考文献触发中文模型。它们作为模块常量集中定义，
后续可通过中英文评测集校准，但本阶段不增加配置项。

## 模型可用性降级

语言决策完成后，在启动 MinerU 子进程前检查所选语言的检测和识别权重：

- 选择 `en` 但英文权重缺失、且 `ch` 权重完整时，改用 `ch`，设置
  `model_fallback=True`，并在 `reason` 中记录降级原因。
- 选择 `ch` 但中文权重缺失时，不能降级到英文模型处理中文扫描件，抛出可分类的
  `MineruError`。
- `auto` 模式的 Doctor 严格检查中英文两套权重；缺任意一套时报告未完全就绪。
- 下载步骤必须准备布局模型、LayoutReader，以及 `ch`、`en` 两套 OCR 检测和识别权重。

## 解析执行降级

语言判断中的预期故障不能抛出异常；它们都返回 `ch` 降级决策。真正的解析执行仍可能因
模型缺失、CUDA OOM、超时或 MinerU 错误而失败。

后续解析调度器按单篇论文隔离这些失败：

1. MinerU 成功且标准化产物非空：状态为 `succeeded`。
2. MinerU 失败且 PDF 有可提取文字：调用 PyMuPDF，产物非空时状态为 `degraded`，记录
   MinerU 原因和降级后端。
3. 扫描 PDF 的 MinerU 失败，或 PyMuPDF 降级结果为空：状态为 `failed`，保留错误分类，
   但批处理继续下一篇。
4. 禁止把空 Markdown 或只有页标记的结果记录成成功。

本阶段的 `mineru_local.parse_pdf()` 继续以 `MineruError` 表示真实执行失败；批次不中断的
职责属于下一步解析调度器，避免单一后端同时承担批处理编排。

## 数据流

```text
PDF + sibling meta.json
        |
        v
language resolver
  forced -> metadata -> text sample -> ch fallback
        |
        v
model availability check
  en missing -> ch when available
        |
        v
magic-pdf -m ocr -l ch|en
        |
        +-- success -> standardized parse output
        |
        +-- failure -> parser dispatcher
                         +-- text PDF -> PyMuPDF degraded
                         +-- scanned/empty -> failed, continue batch
```

## 可观测性

每篇论文至少记录：

- 最终领域语言和 MinerU 语言；
- 决策来源与原因；
- 是否发生模型语言降级；
- 实际解析后端；
- `succeeded`、`degraded` 或 `failed` 状态；
- 失败分类和可执行提示。

Demo 必须逐步打印这些信息，便于人工确认自动判断没有静默选择错误模型。

## 测试与真实验收

### 边界和纯逻辑测试

- `PaperMeta.language` 只接受 `zh`、`en` 或空值。
- 重复采集保留人工语言字段。
- 有效、缺失、损坏和非法 `meta.json` 的决策行为。
- 使用 PyMuPDF 创建真实中文、英文、空白和混合文字 PDF，验证字符统计与阈值。
- PyMuPDF 打开失败时回退 `ch`，不让语言解析异常外泄。
- 强制 `ch/en` 绕过自动判断。
- `auto` 绝不作为参数传给 `magic-pdf`。
- 英文模型缺失时降级 `ch`；中文模型缺失时产生可分类错误。
- Doctor 在 `auto` 模式检查两套 OCR 权重。

### 真实 Demo

- 用户分别输入一篇中文和一篇英文 PDF。
- 扫描件使用真实 `meta.json` 人工语言标记。
- 普通 PDF 不添加语言元数据，必须由文字采样自动判断。
- Demo 打印判断来源、最终 `-l` 参数、GPU 配置和标准化产物摘要。

### 无 mock 集成测试

- 使用项目自己的真实模型目录和 CUDA GPU。
- 中文、英文各执行一次真实 OCR。
- 验证非空 Markdown、图片引用、布局数据和语言决策记录。
- 执行期间通过 GPU 指标确认实际使用 GPU。
- 单独构造一次 MinerU 失败，验证批处理继续且状态不是伪成功。

## 实现顺序

1. 修正当前错误的固定 `en` 测试契约，改为 `auto`。
2. 扩展 `PaperMeta.language` 及其验证测试。
3. 建立统一元数据持久化与人工语言保留规则，再迁移各采集器。
4. 新增独立语言决策模块及真实临时 PDF 测试。
5. 将 MinerU 配置与 Doctor 扩展为 `auto` 和双模型检查。
6. 将语言决策及模型可用性降级接入 `mineru_local.parse_pdf()`。
7. 下载项目独立的布局、LayoutReader、中英文 OCR 模型。
8. 完成中英文真实 GPU OCR Demo 和无 mock 集成测试。
9. 在下一文件实现解析调度器的单篇失败隔离及 PyMuPDF 降级。

## 兼容性与迁移

- 旧 `meta.json` 没有 `language` 时仍可加载，并进入自动文字判断。
- 现有配置的 `lang: ch` 或 `lang: en` 保持强制模式语义。
- 原有 `lang: null` 在迁移时改为 `auto`；配置模型需要拒绝其他未知值。
- 已经人工编辑的语言字段在重新采集时必须保留。
- 运行时模型文件继续位于重建项目自己的 `data/index/mineru_models/`，不得引用基准项目。
