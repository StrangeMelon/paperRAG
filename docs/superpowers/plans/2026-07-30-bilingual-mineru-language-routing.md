# MinerU Bilingual OCR Language Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为中英文混合论文集合实现逐篇 OCR 语言选择、人工扫描件元数据、模型可用性降级、真实 GPU OCR 验收和单篇解析失败隔离。

**Architecture:** 标准元数据使用 `zh/en`，独立语言决策模块把它映射为 MinerU 的 `ch/en`；普通 PDF 通过 PyMuPDF 文字采样判断，无法判断时回退 `ch`。MinerU 适配器负责模型权重检查和 `en -> ch` 模型降级，解析调度器负责 MinerU 到 PyMuPDF 的后端降级并保留基准项目的 `(Path, parser_name)` 接口。

**Tech Stack:** Python 3.10-3.13、Pydantic、PyMuPDF、magic-pdf 1.3.12、PaddleOCR PyTorch 权重、CUDA、huggingface_hub、Pytest、Ruff、uv。

---

## 执行约束

- 正式功能文件由学习者逐个编写；测试和 Demo 由助手创建。
- 每个任务先观察 RED，再编写最小实现，最后运行聚焦测试和 Ruff。
- 不把模型、解析产物、Demo 数据或 `.hf-cache` 提交到 Git。
- 本计划固定模型仓库修订
  `a4f6a8d29a4d96730f90ea174a9322e842b93552`，不得改用浮动 `main`。
- 中英文真实 GPU OCR 未通过前，不进入切块实现。

## 文件职责

- `src/paper_rag/ingest/schema.py`：定义领域语言 `zh/en`。
- `src/paper_rag/ingest/metadata.py`：统一写入 `meta.json` 并保留人工语言。
- 五个 `src/paper_rag/ingest/*_source.py`：改用统一持久化边界。
- `src/paper_rag/parse/language.py`：不依赖 MinerU 的逐篇语言判断。
- `src/paper_rag/parse/mineru_local.py`：Doctor、权重可用性、语言路由和 CLI 执行。
- `scripts/download_mineru_models.py`：从固定官方修订下载最小必需权重集合。
- `scripts/demo_mineru_local.py`：打印语言决策并执行用户提供 PDF 的真实 GPU OCR。
- `src/paper_rag/parse/dispatcher.py`：MinerU/PyMuPDF 降级和空结果拒绝。

### Task 1: 将 MinerU 语言配置改为 `auto | ch | en`

**Files:**
- Modify: `tests/test_mineru_gpu_config.py`
- Modify: `tests/test_config.py`
- Modify: `src/paper_rag/config.py`
- Modify: `config/default.yaml`

- [ ] **Step 1: 修正 GPU 配置测试中的临时 `en` 契约**

将 `tests/test_mineru_gpu_config.py` 中默认 MinerU 字典的语言改成：

```python
"lang": "auto",
```

- [ ] **Step 2: 增加配置值域测试**

在 `tests/test_config.py` 末尾加入：

```python
@pytest.mark.parametrize("language", ["auto", "ch", "en"])
def test_mineru_language_accepts_supported_modes(language: str) -> None:
    assert config._MinerU(lang=language).lang == language


def test_mineru_language_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        config._MinerU(lang="fr")
```

若文件尚未导入 `pytest`，在导入区加入 `import pytest`。

- [ ] **Step 3: 运行 RED**

Run:

```bash
uv run pytest -vv -s tests/test_mineru_gpu_config.py tests/test_config.py
```

Expected: 默认配置仍是 `null`，且 `_MinerU(lang="fr")` 未拒绝未知值。

- [ ] **Step 4: 实现类型化配置**

在 `src/paper_rag/config.py` 导入：

```python
from typing import Any, Literal
```

把 `_MinerU.lang` 改为：

```python
lang: Literal["auto", "ch", "en"] = "auto"
```

把 `config/default.yaml` 改为：

```yaml
mineru:
  mode: local
  cli: magic-pdf
  method: ocr
  lang: auto
  timeout_sec: 600
  fallback_to_pymupdf: true
```

- [ ] **Step 5: 运行 GREEN 和 Ruff**

```bash
uv run pytest -vv -s tests/test_mineru_gpu_config.py tests/test_config.py
uv run ruff check src/paper_rag/config.py tests/test_config.py tests/test_mineru_gpu_config.py
```

Expected: 配置测试全部通过，Ruff 输出 `All checks passed!`。

- [ ] **Step 6: 提交配置契约**

```bash
git add config/default.yaml src/paper_rag/config.py tests/test_config.py tests/test_mineru_gpu_config.py
git commit -m "feat(config): 支持 MinerU OCR 语言自动路由"
```

### Task 2: 为 `PaperMeta` 增加领域语言字段

**Files:**
- Modify: `tests/test_ingest_schema.py`
- Modify: `src/paper_rag/ingest/schema.py`

- [ ] **Step 1: 写入字段和值域测试**

在 `tests/test_ingest_schema.py` 加入：

```python
@pytest.mark.parametrize("language", ["zh", "en", None])
def test_paper_meta_accepts_supported_document_languages(
    language: str | None,
) -> None:
    schema = _schema_module()
    meta = schema.PaperMeta(
        paper_id="paper:language",
        title="Language Paper",
        language=language,
    )

    assert meta.language == language


def test_paper_meta_rejects_unknown_document_language() -> None:
    schema = _schema_module()

    with pytest.raises(ValidationError):
        schema.PaperMeta(
            paper_id="paper:language",
            title="Language Paper",
            language="ch",
        )
```

并在最小元数据测试中加入：

```python
assert meta.language is None
```

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_ingest_schema.py
```

Expected: `PaperMeta` 尚无 `language` 字段。

- [ ] **Step 3: 实现领域字段**

在 `src/paper_rag/ingest/schema.py` 导入 `Literal`：

```python
from typing import Any, Literal
```

在 `PaperMeta` 的 `abstract` 后加入：

```python
language: Literal["zh", "en"] | None = None
```

- [ ] **Step 4: 运行 GREEN 和 Ruff**

```bash
uv run pytest -vv -s tests/test_ingest_schema.py
uv run ruff check src/paper_rag/ingest/schema.py tests/test_ingest_schema.py
```

- [ ] **Step 5: 提交领域模型**

```bash
git add src/paper_rag/ingest/schema.py tests/test_ingest_schema.py
git commit -m "feat(ingest): 为论文元数据增加语言字段"
```

### Task 3: 建立人工语言保留的统一持久化边界

**Files:**
- Create: `tests/test_ingest_metadata.py`
- Create: `src/paper_rag/ingest/metadata.py`

- [ ] **Step 1: 写入真实文件行为测试**

创建 `tests/test_ingest_metadata.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

from paper_rag.ingest.metadata import persist_paper_meta
from paper_rag.ingest.schema import PaperMeta


def _meta(language: str | None) -> PaperMeta:
    return PaperMeta(
        paper_id="paper:manual-language",
        title="Manual Language",
        language=language,
        source="local",
    )


def test_persist_paper_meta_writes_new_language(tmp_path: Path) -> None:
    saved = persist_paper_meta(tmp_path, _meta("zh"))
    payload = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))

    assert saved.language == "zh"
    assert payload["language"] == "zh"


def test_persist_paper_meta_preserves_existing_manual_language(tmp_path: Path) -> None:
    persist_paper_meta(tmp_path, _meta("zh"))

    saved = persist_paper_meta(tmp_path, _meta(None))

    assert saved.language == "zh"
    assert json.loads(
        (tmp_path / "meta.json").read_text(encoding="utf-8")
    )["language"] == "zh"


def test_persist_paper_meta_replaces_malformed_existing_json(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text("{broken", encoding="utf-8")

    saved = persist_paper_meta(tmp_path, _meta("en"))

    assert saved.language == "en"
    assert json.loads(
        (tmp_path / "meta.json").read_text(encoding="utf-8")
    )["language"] == "en"
```

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_ingest_metadata.py
```

Expected: `ModuleNotFoundError: paper_rag.ingest.metadata`。

- [ ] **Step 3: 实现统一持久化函数**

创建 `src/paper_rag/ingest/metadata.py`：

```python
"""标准论文元数据的持久化与人工标注保护。"""

from __future__ import annotations

import json
from pathlib import Path

from ..utils.logger import get_logger
from .schema import PaperMeta

log = get_logger(__name__)


def persist_paper_meta(target: Path, meta: PaperMeta) -> PaperMeta:
    """写入 meta.json，并保留已有的非空人工语言标记。"""

    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "meta.json"
    existing_language: str | None = None

    if meta_path.is_file():
        try:
            existing = PaperMeta.model_validate_json(
                meta_path.read_text(encoding="utf-8")
            )
            existing_language = existing.language
        except Exception as exc:
            log.warning(
                f"invalid existing metadata will be replaced: "
                f"{type(exc).__name__}: {exc}"
            )

    if existing_language is not None:
        meta = meta.model_copy(update={"language": existing_language})

    meta_path.write_text(
        json.dumps(
            meta.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta
```

- [ ] **Step 4: 运行 GREEN 和 Ruff**

```bash
uv run pytest -vv -s tests/test_ingest_metadata.py
uv run ruff check src/paper_rag/ingest/metadata.py tests/test_ingest_metadata.py
```

- [ ] **Step 5: 提交持久化边界**

```bash
git add src/paper_rag/ingest/metadata.py tests/test_ingest_metadata.py
git commit -m "feat(ingest): 保留论文人工语言标注"
```

### Task 4: 迁移本地 PDF 采集器

**Files:**
- Modify: `src/paper_rag/ingest/local_source.py`
- Modify: `tests/test_local_source.py`

- [ ] **Step 1: 在现有本地采集测试追加重复采集断言**

在已有成功采集用例完成第一次 `fetch()` 后执行：

```python
meta_path = target_dir / "meta.json"
payload = json.loads(meta_path.read_text(encoding="utf-8"))
payload["language"] = "zh"
meta_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

second = module.LocalSource().fetch(str(source_pdf))

assert second.meta.language == "zh"
assert json.loads(meta_path.read_text(encoding="utf-8"))["language"] == "zh"
```

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_local_source.py
```

Expected: 第二次采集把人工语言覆盖成空值。

- [ ] **Step 3: 改用统一持久化函数**

删除 `import json`，加入：

```python
from .metadata import persist_paper_meta
```

把直接写 `meta.json` 的代码块替换为：

```python
meta = persist_paper_meta(target, meta)
```

- [ ] **Step 4: 运行 GREEN、真实本地回归和 Ruff**

```bash
uv run pytest -vv -s tests/test_local_source.py tests/test_local_source_real.py
uv run ruff check src/paper_rag/ingest/local_source.py tests/test_local_source.py
```

- [ ] **Step 5: 提交本地采集迁移**

```bash
git add src/paper_rag/ingest/local_source.py tests/test_local_source.py
git commit -m "refactor(ingest): 统一本地论文元数据持久化"
```

### Task 5: 迁移 URL PDF 采集器

**Files:**
- Modify: `src/paper_rag/ingest/url_source.py`
- Verify: `tests/test_url_source.py`
- Verify: `tests/test_url_source_real.py`

- [ ] **Step 1: 运行迁移前基线**

```bash
uv run pytest -vv -s tests/test_url_source.py
```

- [ ] **Step 2: 改用统一持久化函数**

删除 `import json`，加入：

```python
from .metadata import persist_paper_meta
```

把直接写 `meta.json` 的代码块替换为：

```python
meta = persist_paper_meta(target, meta)
```

- [ ] **Step 3: 运行边界、真实公网回归和 Ruff**

```bash
uv run pytest -vv -s tests/test_url_source.py
uv run pytest -vv -s tests/test_url_source_real.py
uv run ruff check src/paper_rag/ingest/url_source.py tests/test_url_source.py
```

- [ ] **Step 4: 提交 URL 采集迁移**

```bash
git add src/paper_rag/ingest/url_source.py
git commit -m "refactor(ingest): 统一 URL 论文元数据持久化"
```

### Task 6: 迁移 arXiv 采集器

**Files:**
- Modify: `src/paper_rag/ingest/arxiv_source.py`
- Verify: `tests/test_arxiv_source.py`
- Verify: `tests/test_arxiv_source_real.py`

- [ ] **Step 1: 运行迁移前基线**

```bash
uv run pytest -vv -s tests/test_arxiv_source.py
```

- [ ] **Step 2: 让 `_persist_meta` 返回合并后的元数据**

删除 `import json`，加入：

```python
from .metadata import persist_paper_meta
```

调用处改为：

```python
meta = _persist_meta(
    target=target,
    meta=meta,
    source_query=identifier,
)
```

函数签名和函数体改为：

```python
def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准化元数据和原始采集参数。"""

    meta = persist_paper_meta(target, meta)
    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )
    return meta
```

- [ ] **Step 3: 运行边界、真实 arXiv 回归和 Ruff**

```bash
uv run pytest -vv -s tests/test_arxiv_source.py
uv run pytest -vv -s tests/test_arxiv_source_real.py
uv run ruff check src/paper_rag/ingest/arxiv_source.py tests/test_arxiv_source.py
```

- [ ] **Step 4: 提交 arXiv 迁移**

```bash
git add src/paper_rag/ingest/arxiv_source.py
git commit -m "refactor(ingest): 统一 arXiv 元数据持久化"
```

### Task 7: 迁移 OpenAlex 采集器

**Files:**
- Modify: `src/paper_rag/ingest/openalex_source.py`
- Verify: `tests/test_openalex_source.py`
- Verify: `tests/test_openalex_source_real.py`

- [ ] **Step 1: 运行迁移前基线**

```bash
uv run pytest -vv -s tests/test_openalex_source.py
```

- [ ] **Step 2: 映射可信 OpenAlex 语言并统一持久化**

删除 `import json`，加入：

```python
from .metadata import persist_paper_meta
```

在构造 `PaperMeta` 前加入：

```python
reported_language = data.get("language")
language = reported_language if reported_language in {"zh", "en"} else None
```

在构造 `PaperMeta` 的关键字参数中加入：

```python
language=language,
```

在调用处写成：

```python
meta = _persist_meta(
    target=target,
    meta=meta,
    source_query=identifier,
)
```

把本文件 `_persist_meta` 完整替换为：

```python
def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准化元数据和原始采集参数。"""

    meta = persist_paper_meta(target, meta)
    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )
    return meta
```

- [ ] **Step 3: 运行边界、真实 OpenAlex 回归和 Ruff**

```bash
uv run pytest -vv -s tests/test_openalex_source.py
uv run pytest -vv -s tests/test_openalex_source_real.py
uv run ruff check src/paper_rag/ingest/openalex_source.py tests/test_openalex_source.py
```

- [ ] **Step 4: 提交 OpenAlex 迁移**

```bash
git add src/paper_rag/ingest/openalex_source.py
git commit -m "refactor(ingest): 保留 OpenAlex 论文语言元数据"
```

### Task 8: 迁移 Semantic Scholar 采集器

**Files:**
- Modify: `src/paper_rag/ingest/semantic_scholar_source.py`
- Verify: `tests/test_semantic_scholar_source.py`

- [ ] **Step 1: 运行迁移前基线**

```bash
uv run pytest -vv -s tests/test_semantic_scholar_source.py
```

- [ ] **Step 2: 统一持久化并返回保留语言后的对象**

删除 `import json`，加入：

```python
from .metadata import persist_paper_meta
```

把调用改为：

```python
meta = _persist_meta(
    target=target,
    meta=meta,
    source_query=identifier,
)
```

把本文件 `_persist_meta` 完整替换为：

```python
def _persist_meta(
    *,
    target: Path,
    meta: PaperMeta,
    source_query: str,
) -> PaperMeta:
    """持久化标准化元数据和原始采集参数。"""

    meta = persist_paper_meta(target, meta)
    (target / "source.txt").write_text(
        f"source={meta.source}\nquery={source_query}\n",
        encoding="utf-8",
    )
    return meta
```

- [ ] **Step 3: 运行 GREEN 和 Ruff**

```bash
uv run pytest -vv -s tests/test_semantic_scholar_source.py
uv run ruff check src/paper_rag/ingest/semantic_scholar_source.py tests/test_semantic_scholar_source.py
```

Expected: 4 个边界用例通过；真实 API 验收仍保留为取得 API key 后的课程门禁。

- [ ] **Step 4: 提交 Semantic Scholar 迁移**

```bash
git add src/paper_rag/ingest/semantic_scholar_source.py
git commit -m "refactor(ingest): 统一 Semantic Scholar 元数据持久化"
```

### Task 9: 实现不依赖 MinerU 的语言决策模块

**Files:**
- Create: `tests/test_parse_language.py`
- Create: `src/paper_rag/parse/language.py`

- [ ] **Step 1: 创建真实 PDF 决策测试**

测试文件必须使用 PyMuPDF 创建：

```python
def _write_pdf(path: Path, text: str, *, font_name: str = "helv") -> None:
    import fitz

    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text, fontname=font_name, fontsize=12)
    document.save(path)
    document.close()
```

至少加入以下断言：

```python
def test_metadata_language_has_priority(tmp_path: Path) -> None:
    pdf = tmp_path / "raw.pdf"
    _write_pdf(pdf, "English text " * 20)
    (tmp_path / "meta.json").write_text(
        json.dumps({"language": "zh"}),
        encoding="utf-8",
    )

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "zh"
    assert decision.mineru_language == "ch"
    assert decision.source == "metadata"


def test_english_text_selects_english_model(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "Retrieval augmented generation research paper. " * 20)

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "en"
    assert decision.mineru_language == "en"
    assert decision.source == "pdf_text"


def test_chinese_text_selects_bilingual_model(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf, "检索增强生成能够处理中文学术论文。" * 20, font_name="china-s")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language == "zh"
    assert decision.mineru_language == "ch"


def test_blank_scanned_pdf_falls_back_without_raising(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf, "")

    decision = resolve_ocr_language(pdf, "auto")

    assert decision.document_language is None
    assert decision.mineru_language == "ch"
    assert decision.source == "fallback"
    assert decision.reason == "no_extractable_text"
```

还要覆盖损坏 `meta.json`、损坏 PDF、强制 `ch` 和强制 `en`。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_parse_language.py
```

Expected: `ModuleNotFoundError: paper_rag.parse.language`。

- [ ] **Step 3: 实现语言决策模块**

接口和常量必须保持：

```python
MAX_PAGES = 5
MAX_CHARACTERS = 20_000
MIN_CJK_CHARACTERS = 20
MIN_LATIN_CHARACTERS = 50
MIN_CJK_RATIO = 0.05


@dataclass(frozen=True)
class OcrLanguageDecision:
    document_language: Literal["zh", "en"] | None
    mineru_language: Literal["ch", "en"]
    source: Literal["forced", "metadata", "pdf_text", "fallback"]
    reason: str
    model_fallback: bool = False


def resolve_ocr_language(
    pdf_path: str | Path,
    configured_language: Literal["auto", "ch", "en"] = "auto",
    *,
    meta_path: str | Path | None = None,
) -> OcrLanguageDecision:
    pdf = Path(pdf_path).expanduser().resolve()

    if configured_language == "ch":
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="forced",
            reason="forced_ch",
        )
    if configured_language == "en":
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="forced",
            reason="forced_en",
        )
    if configured_language != "auto":
        raise ValueError(f"unsupported OCR language mode: {configured_language}")

    resolved_meta_path = (
        Path(meta_path).expanduser().resolve()
        if meta_path is not None
        else pdf.parent / "meta.json"
    )
    metadata_language = _read_metadata_language(resolved_meta_path)
    if metadata_language == "zh":
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="metadata",
            reason="valid_meta_language",
        )
    if metadata_language == "en":
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="metadata",
            reason="valid_meta_language",
        )

    try:
        text = _sample_pdf_text(pdf)
    except Exception as exc:
        return OcrLanguageDecision(
            document_language=None,
            mineru_language="ch",
            source="fallback",
            reason=f"pdf_text_error:{type(exc).__name__}",
        )

    if not text.strip():
        return OcrLanguageDecision(
            document_language=None,
            mineru_language="ch",
            source="fallback",
            reason="no_extractable_text",
        )

    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin_count = sum(char.isascii() and char.isalpha() for char in text)
    language_characters = cjk_count + latin_count
    cjk_ratio = cjk_count / language_characters if language_characters else 0.0

    if cjk_count >= MIN_CJK_CHARACTERS and cjk_ratio >= MIN_CJK_RATIO:
        return OcrLanguageDecision(
            document_language="zh",
            mineru_language="ch",
            source="pdf_text",
            reason="cjk_text_detected",
        )
    if latin_count >= MIN_LATIN_CHARACTERS:
        return OcrLanguageDecision(
            document_language="en",
            mineru_language="en",
            source="pdf_text",
            reason="latin_text_detected",
        )
    return OcrLanguageDecision(
        document_language=None,
        mineru_language="ch",
        source="fallback",
        reason="insufficient_language_signal",
    )
```

同一文件还要实现：

```python
def _read_metadata_language(
    meta_path: Path,
) -> Literal["zh", "en"] | None:
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    language = payload.get("language")
    return language if language in {"zh", "en"} else None


def _sample_pdf_text(pdf_path: Path) -> str:
    import fitz

    document = fitz.open(str(pdf_path))
    parts: list[str] = []
    characters = 0
    try:
        for page_index in range(min(len(document), MAX_PAGES)):
            page_text = document[page_index].get_text("text") or ""
            remaining = MAX_CHARACTERS - characters
            parts.append(page_text[:remaining])
            characters += len(parts[-1])
            if characters >= MAX_CHARACTERS:
                break
    finally:
        document.close()
    return "\n".join(parts)
```

文件导入区必须包含 `json`、`dataclass`、`Path` 和 `Literal`。元数据和 PDF 读取异常都
转换为稳定 `reason`，只有无效应用配置会抛 `ValueError`。

- [ ] **Step 4: 运行 GREEN 和 Ruff**

```bash
uv run pytest -vv -s tests/test_parse_language.py
uv run ruff check src/paper_rag/parse/language.py tests/test_parse_language.py
```

- [ ] **Step 5: 提交语言决策模块**

```bash
git add src/paper_rag/parse/language.py tests/test_parse_language.py
git commit -m "feat(parse): 实现中英文 OCR 语言自动判断"
```

### Task 10: 让 Doctor 在 `auto` 模式检查双语权重

**Files:**
- Modify: `tests/test_mineru_local.py`
- Modify: `src/paper_rag/parse/mineru_local.py`

- [ ] **Step 1: 写入 `auto` 双语权重测试**

复用现有临时 `magic-pdf.json` 测试夹具，增加：

```python
checks = module._ocr_model_weight_checks(config_path, "auto")

assert [check.name for check in checks] == [
    "OCR detection weight:ch",
    "OCR recognition weight:ch",
    "OCR detection weight:en",
    "OCR recognition weight:en",
]
```

再创建临时 `Layout/LayoutReader/config.json` 与 `model.safetensors`，断言
`_layout_reader_weight_checks(config_path)` 返回两个同名 `[OK]` 检查；删除模型文件后对应
检查必须变为失败。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_mineru_local.py -k "ocr_weight"
```

Expected: `auto` 被当成不支持的 OCR 语言。

- [ ] **Step 3: 提取权重路径函数并支持 `auto`**

新增完整权重路径函数：

```python
def _ocr_weight_paths(
    config_path: Path,
    language: str,
) -> tuple[Path, Path]:
    """返回指定语言的检测与识别权重路径。"""

    import yaml

    mineru_config = json.loads(config_path.read_text(encoding="utf-8"))
    magic_pdf = import_module("magic_pdf")
    package_root = Path(magic_pdf.__file__).resolve().parent
    ocr_config_path = (
        package_root
        / "model"
        / "sub_modules"
        / "ocr"
        / "paddleocr2pytorch"
        / "pytorchocr"
        / "utils"
        / "resources"
        / "models_config.yml"
    )
    ocr_config = yaml.safe_load(ocr_config_path.read_text(encoding="utf-8"))
    selected = (ocr_config.get("lang") or {}).get(language)
    if not isinstance(selected, dict):
        raise ValueError(f"unsupported OCR language: {language}")

    detection_name = selected.get("det")
    recognition_name = selected.get("rec")
    if not isinstance(detection_name, str) or not isinstance(recognition_name, str):
        raise ValueError(f"incomplete OCR model config: {language}")

    configured_models_dir = mineru_config.get("models-dir") or "/tmp/models"
    models_dir = Path(configured_models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = (cfg.PROJECT_ROOT / models_dir).resolve()
    ocr_dir = models_dir / "OCR" / "paddleocr_torch"
    return ocr_dir / detection_name, ocr_dir / recognition_name
```

把 `_ocr_model_weight_checks()` 替换为：

```python
def _ocr_model_weight_checks(
    config_path: Path,
    language: str | None,
) -> list[MineruCheck]:
    """检查一个语言或 auto 模式所需的真实 OCR 权重。"""

    if not config_path.exists():
        return []
    if not language:
        return [
            MineruCheck(
                name="OCR language",
                ok=False,
                detail="not configured",
                hint="将 mineru.lang 设置为 auto、ch 或 en。",
            )
        ]

    languages = ("ch", "en") if language == "auto" else (language,)
    checks: list[MineruCheck] = []
    for selected_language in languages:
        try:
            detection_path, recognition_path = _ocr_weight_paths(
                config_path,
                selected_language,
            )
        except Exception as exc:
            checks.append(
                MineruCheck(
                    name=f"OCR weights:{selected_language}",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                    hint="检查 magic-pdf OCR 模型配置与语言值。",
                )
            )
            continue

        for role, weight_path in (
            ("detection", detection_path),
            ("recognition", recognition_path),
        ):
            checks.append(
                MineruCheck(
                    name=f"OCR {role} weight:{selected_language}",
                    ok=weight_path.is_file(),
                    detail=str(weight_path),
                    hint=(
                        "下载对应语言的 MinerU OCR 权重，"
                        "并放入 OCR/paddleocr_torch/。"
                    ),
                )
            )
    return checks
```

再增加：

```python
def _ocr_weights_available(config_path: Path, language: str) -> bool:
    try:
        return all(
            path.is_file()
            for path in _ocr_weight_paths(config_path, language)
        )
    except Exception:
        return False
```

增加 LayoutReader 检查：

```python
def _layout_reader_weight_checks(config_path: Path) -> list[MineruCheck]:
    """检查阅读顺序模型的配置与权重。"""

    if not config_path.is_file():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        configured_dir = payload.get("layoutreader-model-dir")
        if not isinstance(configured_dir, str) or not configured_dir.strip():
            raise ValueError("layoutreader-model-dir not configured")
        model_dir = Path(configured_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = (cfg.PROJECT_ROOT / model_dir).resolve()
    except Exception as exc:
        return [
            MineruCheck(
                name="LayoutReader config",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                hint="配置 layoutreader-model-dir。",
            )
        ]

    checks: list[MineruCheck] = []
    for filename in ("config.json", "model.safetensors"):
        path = model_dir / filename
        checks.append(
            MineruCheck(
                name=f"LayoutReader weight:{filename}",
                ok=path.is_file(),
                detail=str(path),
                hint="下载 LayoutReader 配置和 safetensors 权重。",
            )
        )
    return checks
```

在 `diagnose()` 的 `_enabled_model_weight_checks(config_path)` 后加入：

```python
checks.extend(_layout_reader_weight_checks(config_path))
```

- [ ] **Step 4: 运行完整 MinerU 边界回归**

```bash
uv run pytest -vv -s tests/test_mineru_local.py
uv run pytest -vv -s tests/test_mineru_doctor_script.py
uv run ruff check src/paper_rag/parse/mineru_local.py tests/test_mineru_local.py
```

- [ ] **Step 5: 提交 Doctor 双语诊断**

```bash
git add src/paper_rag/parse/mineru_local.py tests/test_mineru_local.py
git commit -m "feat(parse): 诊断 MinerU 中英文 OCR 权重"
```

### Task 11: 将自动语言与模型降级接入 MinerU CLI

**Files:**
- Modify: `tests/test_mineru_local.py`
- Modify: `src/paper_rag/parse/mineru_local.py`

- [ ] **Step 1: 写入 CLI 不接收 `auto` 的测试**

先在 `tests/test_mineru_local.py` 加入测试隔离夹具，避免不涉及模型文件的边界测试读取真实
权重目录：

```python
@pytest.fixture(autouse=True)
def _assume_ocr_weights_for_boundary_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _mineru_module()
    monkeypatch.setattr(
        module,
        "_ocr_weights_available",
        lambda *_args: True,
        raising=False,
    )
```

把现有 `_parser_config()` 的语言改成 `auto`，并在成功解析测试中替换语言决策：

```python
decision = module.OcrLanguageDecision(
    document_language="en",
    mineru_language="en",
    source="pdf_text",
    reason="latin_text_detected",
    model_fallback=False,
)
monkeypatch.setattr(module, "resolve_ocr_language", lambda *args, **kwargs: decision)
monkeypatch.setattr(module, "_ocr_weights_available", lambda *args: True)
```

继续断言命令包含 `"-l", "en"`，并断言命令中没有字符串 `auto`。

增加英文权重缺失测试：英文不可用、中文可用时，命令必须变成 `-l ch`，输出目录的
`language.json` 必须包含 `"model_fallback": true`。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_mineru_local.py -k "parse_pdf"
```

- [ ] **Step 3: 实现语言选择和记录**

在 `mineru_local.py` 导入：

```python
from dataclasses import asdict, dataclass, replace
from .language import OcrLanguageDecision, resolve_ocr_language
```

新增：

```python
def _select_available_ocr_language(
    decision: OcrLanguageDecision,
    config_path: Path,
) -> OcrLanguageDecision:
    if _ocr_weights_available(config_path, decision.mineru_language):
        return decision
    if (
        decision.mineru_language == "en"
        and _ocr_weights_available(config_path, "ch")
    ):
        return replace(
            decision,
            mineru_language="ch",
            reason=f"{decision.reason};english_weights_missing",
            model_fallback=True,
        )
    raise MineruError(
        f"OCR model weights missing for {decision.mineru_language}"
    )
```

在 `parse_pdf()` 创建输出目录后、构造命令前执行：

```python
config_path = _mineru_config_path()
decision = resolve_ocr_language(
    resolved_pdf_path,
    config.mineru.lang,
)
decision = _select_available_ocr_language(decision, config_path)
(output_dir / "language.json").write_text(
    json.dumps(asdict(decision), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

命令始终追加：

```python
command.extend(["-l", decision.mineru_language])
```

删除原先直接使用 `config.mineru.lang` 的条件分支。

- [ ] **Step 4: 运行 GREEN 和完整回归**

```bash
uv run pytest -vv -s tests/test_mineru_local.py tests/test_parse_language.py
uv run ruff check src/paper_rag/parse/mineru_local.py tests/test_mineru_local.py
```

- [ ] **Step 5: 提交 MinerU 语言接入**

```bash
git add src/paper_rag/parse/mineru_local.py tests/test_mineru_local.py
git commit -m "feat(parse): 按论文选择 MinerU OCR 语言"
```

### Task 12: 创建可复现的 MinerU 模型下载脚本

**Files:**
- Create: `tests/test_download_mineru_models.py`
- Create: `scripts/download_mineru_models.py`

- [ ] **Step 1: 写入固定修订与路径映射测试**

测试必须断言：

```python
assert module.REVISION == "a4f6a8d29a4d96730f90ea174a9322e842b93552"
assert module.MODEL_FILES[
    "models/OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth"
][0] == Path("OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth")
assert module.MODEL_FILES[
    "models/ReadingOrder/layout_reader/model.safetensors"
][0] == Path("Layout/LayoutReader/model.safetensors")
```

再用 monkeypatch 替换 `hf_hub_download`，返回真实临时文件，验证脚本把所有文件复制到
`--models-dir` 下且第二次运行复用已有非空文件。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_download_mineru_models.py
```

- [ ] **Step 3: 实现下载清单**

脚本常量必须包含以下七个远端文件和本地映射：

```python
REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
REVISION = "a4f6a8d29a4d96730f90ea174a9322e842b93552"
MODEL_FILES = {
    "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt": (
        Path("Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"),
        100_000,
    ),
    "models/ReadingOrder/layout_reader/config.json": (
        Path("Layout/LayoutReader/config.json"),
        100,
    ),
    "models/ReadingOrder/layout_reader/model.safetensors": (
        Path("Layout/LayoutReader/model.safetensors"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth": (
        Path("OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth": (
        Path("OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth": (
        Path("OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/en_PP-OCRv4_rec_infer.pth": (
        Path("OCR/paddleocr_torch/en_PP-OCRv4_rec_infer.pth"),
        100_000,
    ),
}
```

每个文件使用：

```python
cached = hf_hub_download(
    repo_id=REPO_ID,
    filename=remote_path,
    revision=REVISION,
    cache_dir=models_dir / ".hf-cache",
)
```

复制到临时 `.part` 文件，检查大小后用 `Path.replace()` 原子替换目标；存在且大小达标时
打印 `reuse` 并跳过。

- [ ] **Step 4: 运行边界验证和 Ruff**

```bash
uv run pytest -vv -s tests/test_download_mineru_models.py
uv run ruff check scripts/download_mineru_models.py tests/test_download_mineru_models.py
uv run python scripts/download_mineru_models.py --help
```

- [ ] **Step 5: 提交下载入口，不提交权重**

```bash
git add scripts/download_mineru_models.py tests/test_download_mineru_models.py
git commit -m "feat(parse): 增加 MinerU 双语模型下载入口"
```

### Task 13: 下载真实模型并让 Doctor 严格通过

**Files:**
- Runtime only: `data/index/mineru_models/`
- Verify: `scripts/mineru_doctor.py`

- [ ] **Step 1: 执行独立模型下载**

```bash
uv run python scripts/download_mineru_models.py
```

Expected: 七个文件均打印 `downloaded` 或 `reuse`，进程返回 `0`。

- [ ] **Step 2: 检查真实文件清单**

```bash
find data/index/mineru_models -type f -printf '%P %s bytes\n' | sort
```

Expected: 布局模型、LayoutReader 两个文件和中英文四个 OCR 权重均非空。

- [ ] **Step 3: 运行真实严格 Doctor**

```bash
uv run python scripts/mineru_doctor.py --strict
echo $?
```

Expected: 所有依赖、布局、中英文 OCR 权重检查为 `[OK]`，退出码为 `0`。

- [ ] **Step 4: 确认运行数据未进入 Git**

```bash
git status --short
```

Expected: `data/index/mineru_models/` 不出现在待提交列表中。

### Task 14: 完成中英文真实 GPU OCR Demo 与集成测试

**Files:**
- Modify: `scripts/demo_mineru_local.py`
- Create: `tests/test_mineru_bilingual_real.py`

- [ ] **Step 1: Demo 打印语言决策**

把 `--lang` 参数改成：

```python
parser.add_argument(
    "--lang",
    choices=("auto", "ch", "en"),
    default="auto",
    help="OCR 语言路由模式。",
)
```

在调用生产解析器前调用 `resolve_ocr_language()`，打印：

```text
document_language=<zh|en|None>
mineru_language=<ch|en>
source=<forced|metadata|pdf_text|fallback>
reason=<stable reason>
```

解析后读取并验证 `language.json` 与标准化 `paper.md`。

- [ ] **Step 2: 创建双语真实集成测试**

测试通过环境变量接收两份真实文件：

```text
PAPER_RAG_REAL_ENGLISH_PDF=/absolute/english.pdf
PAPER_RAG_REAL_CHINESE_PDF=/absolute/chinese.pdf
```

若文件缺失，测试必须明确失败并说明变量名，不能 `skip`。中文扫描件旁必须有
`meta.json` 且包含 `{"language": "zh"}`；英文普通 PDF 不放语言元数据。分别调用真实
`mineru_local.parse_pdf()`，断言：

```python
assert (output_dir / "paper.md").stat().st_size > 100
assert (output_dir / "language.json").is_file()
assert language_payload["mineru_language"] in {"ch", "en"}
```

中文必须选择 `ch` 且来源为 `metadata`；英文必须选择 `en` 且来源为 `pdf_text`。

- [ ] **Step 3: 运行 Ruff 和 Demo 帮助**

```bash
uv run ruff check scripts/demo_mineru_local.py tests/test_mineru_bilingual_real.py
uv run python scripts/demo_mineru_local.py --help
```

- [ ] **Step 4: 分别运行用户可见 Demo**

```bash
uv run python scripts/demo_mineru_local.py /absolute/english.pdf
uv run python scripts/demo_mineru_local.py /absolute/chinese.pdf
```

在另一终端观察：

```bash
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
```

- [ ] **Step 5: 运行无 mock 双语集成测试**

```bash
PAPER_RAG_REAL_ENGLISH_PDF=/absolute/english.pdf \
PAPER_RAG_REAL_CHINESE_PDF=/absolute/chinese.pdf \
uv run pytest -vv -s tests/test_mineru_bilingual_real.py
```

Expected: 两个真实 GPU OCR 用例通过，Markdown、布局和语言记录非空。

- [ ] **Step 6: 提交 Demo 和真实测试，不提交产物**

```bash
git add scripts/demo_mineru_local.py tests/test_mineru_bilingual_real.py
git commit -m "test(parse): 验收 MinerU 中英文 GPU OCR"
```

### Task 15: 实现解析调度器的后端降级与空结果拒绝

**Files:**
- Create: `tests/test_parse_dispatcher.py`
- Create: `src/paper_rag/parse/dispatcher.py`
- Create: `scripts/demo_parse_dispatcher.py`

- [ ] **Step 1: 写入调度边界测试**

覆盖以下接口：

```python
output_dir, parser_name = dispatcher.parse_pdf(paper_id, pdf_path)
```

断言 MinerU 成功返回 `"mineru"`；MinerU 抛 `MineruError` 且 PyMuPDF 产生有效正文时返回
`"pymupdf"`；PyMuPDF 只有页标记时抛 `ParseError`；禁用 fallback 时重新抛 MinerU 错误。
每条路径都验证 `parsed/<paper_id>/parse_status.json` 的 `status` 为 `succeeded`、
`degraded` 或 `failed`。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest -vv -s tests/test_parse_dispatcher.py
```

- [ ] **Step 3: 实现兼容调度器**

必须保留：

```python
class ParseError(RuntimeError):
    """所有解析后端均未产生可用正文。"""


def parse_pdf(paper_id: str, pdf_path: str | Path) -> tuple[Path, str]:
    """返回标准化目录和实际解析后端名称。"""
```

增加 `_has_meaningful_markdown()`，读取 `paper.md` 后用正则删除
`<!-- page 1 -->` 形式的注释再判断是否有非空正文。MinerU 失败时仅在
`fallback_to_pymupdf=True` 执行 PyMuPDF；降级结果无正文时写入 `failed` 状态并抛
`ParseError`。状态文件使用 JSON，包含 `paper_id`、`status`、`parser`、`reason`。

完整实现骨架为：

```python
"""解析后端调度、降级和结果有效性检查。"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

from .. import config as cfg
from ..utils.logger import get_logger
from ..utils.paths import parsed_dir

log = get_logger(__name__)
_PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)


class ParseError(RuntimeError):
    """所有解析后端均未产生可用正文。"""


def _has_meaningful_markdown(output_dir: Path) -> bool:
    markdown_path = output_dir / "paper.md"
    if not markdown_path.is_file():
        return False
    markdown = markdown_path.read_text(encoding="utf-8")
    return bool(_PAGE_MARKER_RE.sub("", markdown).strip())


def _write_status(
    output_dir: Path,
    *,
    paper_id: str,
    status: str,
    parser: str | None,
    reason: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parse_status.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "status": status,
                "parser": parser,
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_pdf(paper_id: str, pdf_path: str | Path) -> tuple[Path, str]:
    """返回标准化目录和实际解析后端名称。"""

    config = cfg.load()
    status_dir = parsed_dir(paper_id)
    mineru_reason = ""

    if config.mineru.mode == "local":
        mineru_local = importlib.import_module("paper_rag.parse.mineru_local")
        try:
            output_dir = mineru_local.parse_pdf(paper_id, pdf_path)
            if not _has_meaningful_markdown(output_dir):
                raise mineru_local.MineruError("MinerU produced no meaningful text")
            _write_status(
                output_dir,
                paper_id=paper_id,
                status="succeeded",
                parser="mineru",
                reason="",
            )
            return output_dir, "mineru"
        except mineru_local.MineruError as exc:
            mineru_reason = str(exc)
            log.warning(f"mineru failed: {exc}")
            if not config.mineru.fallback_to_pymupdf:
                _write_status(
                    status_dir,
                    paper_id=paper_id,
                    status="failed",
                    parser="mineru",
                    reason=mineru_reason,
                )
                raise

    fallback = importlib.import_module("paper_rag.parse.fallback_pymupdf")
    try:
        output_dir = fallback.parse_pdf(paper_id, pdf_path)
    except Exception as exc:
        reason = f"pymupdf_failed:{type(exc).__name__}:{exc}"
        _write_status(
            status_dir,
            paper_id=paper_id,
            status="failed",
            parser="pymupdf",
            reason=reason,
        )
        raise ParseError(reason) from exc

    if not _has_meaningful_markdown(output_dir):
        reason = "pymupdf_produced_no_meaningful_text"
        _write_status(
            output_dir,
            paper_id=paper_id,
            status="failed",
            parser="pymupdf",
            reason=reason,
        )
        raise ParseError(reason)

    status = "degraded" if mineru_reason else "succeeded"
    _write_status(
        output_dir,
        paper_id=paper_id,
        status=status,
        parser="pymupdf",
        reason=mineru_reason,
    )
    return output_dir, "pymupdf"
```

- [ ] **Step 4: 运行边界与真实 PyMuPDF 回归**

```bash
uv run pytest -vv -s tests/test_parse_dispatcher.py tests/test_fallback_pymupdf.py
uv run ruff check src/paper_rag/parse/dispatcher.py tests/test_parse_dispatcher.py
```

- [ ] **Step 5: 创建真实降级 Demo**

`scripts/demo_parse_dispatcher.py` 接受用户 PDF，临时配置一个不存在的 MinerU CLI，调用
真实调度器并打印 `parse_status.json`。普通文字 PDF 必须显示 `degraded/pymupdf`；扫描件
无法产生正文时必须显示明确失败，不得伪装为通过。

- [ ] **Step 6: 运行真实降级 Demo**

```bash
uv run python scripts/demo_parse_dispatcher.py /absolute/text-paper.pdf
```

- [ ] **Step 7: 提交解析调度器 checkpoint**

```bash
git add src/paper_rag/parse/dispatcher.py tests/test_parse_dispatcher.py scripts/demo_parse_dispatcher.py
git commit -m "feat(parse): 实现解析后端降级与失败隔离"
```

## 最终阶段门禁

依次运行：

```bash
uv run pytest -q tests/test_ingest_schema.py tests/test_ingest_metadata.py
uv run pytest -q tests/test_local_source.py tests/test_url_source.py tests/test_arxiv_source.py
uv run pytest -q tests/test_openalex_source.py tests/test_semantic_scholar_source.py
uv run pytest -q tests/test_parse_language.py tests/test_mineru_local.py
uv run pytest -q tests/test_mineru_doctor_script.py tests/test_parse_dispatcher.py
uv run ruff check src/paper_rag/ingest src/paper_rag/parse tests scripts
```

真实门禁单独运行并保留完整输出：

```bash
uv run python scripts/mineru_doctor.py --strict
PAPER_RAG_REAL_ENGLISH_PDF=/absolute/english.pdf \
PAPER_RAG_REAL_CHINESE_PDF=/absolute/chinese.pdf \
uv run pytest -vv -s tests/test_mineru_bilingual_real.py
```

只有上述命令通过、真实 GPU 指标可见、运行数据未进入 Git，才允许进入切块模块。
