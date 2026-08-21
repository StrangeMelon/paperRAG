"""paper_rag 的统一配置加载器.

Reads `config/default.yaml`, expands `$ENV_VAR` placeholders, and exposes a
typed config object via `load()`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class _Paths(BaseModel):
    data_root: str
    papers_dir: str
    parsed_dir: str
    index_dir: str
    sqlite_path: str
    bm25_path: str
    models_dir: str


class _Embedding(BaseModel):
    provider: str = "bge-m3"
    model_name: str = "BAAI/bge-m3"
    version: str = "bge-m3:v1"
    dim: int = 1024
    batch_size: int = 32
    max_length: int = 8192
    device: str = "auto"


class _Reranker(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    cache_dir: str | None = None
    use_fp16: bool = True
    top_k: int = 8


class _Qdrant(BaseModel):
    url: str = "http://localhost:6333"
    local_path: str | None = None  # if set, use embedded Qdrant (no docker)
    collection_chunks: str = "paper_chunks"
    collection_wiki: str = "wiki_entries"
    distance: str = "Cosine"


class _MinerU(BaseModel):
    mode: str = "local"
    cli: str = "mineru"
    method: str = "auto"  # auto | txt | ocr
    lang: Literal["auto", "ch", "en"] = "auto"
    timeout_sec: int = 600
    fallback_to_pymupdf: bool = True


class _ChunkText(BaseModel):
    target_tokens: int = 500
    overlap_tokens: int = 50
    encoding: str = "cl100k_base"


class _Chunk(BaseModel):
    text: _ChunkText = Field(default_factory=_ChunkText)
    context_prefix: str = "[Title: {title}] [Section: {section}]\n"
    context_prefix_zh: str = "[标题: {title}] [章节: {section}]\n"


class _ReferencePolicy(BaseModel):
    enabled: bool = True
    penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    exclude_from_evidence: bool = True
    legacy_section_fallback: bool = True


class _Retrieve(BaseModel):
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 8
    sparse_backend: str = "fts5"
    fts5_cjk_bigram: bool = True
    fts5_phrase_max_run: int = 6
    bm25_max_chunks: int = 200000
    references: _ReferencePolicy = Field(default_factory=_ReferencePolicy)


class _Abstain(BaseModel):
    """Three-way abstain decision based on retrieval evidence quality.

    Calibrated by ``scripts/calibrate_abstain.py`` against a labeled eval run
    (positives + no-answer negatives). Defaults err on the conservative side
    (enabled but with low/high thresholds that only catch the most obvious
    no-evidence cases — the typical fpr=0 operating point on the M6 33-question
    set).
    """

    enabled: bool = True
    threshold_low: float = 0.20  # < low      -> no_evidence (LLM skipped)
    threshold_high: float = 0.40  # >= high    -> confident (normal flow)
    min_chunks: int = 3  # avg top-N chunk scores for decision
    no_evidence_message: str = (
        "未在已索引文献中找到与该问题相关的内容。请确认问题与已入库的论文主题"
        "相符，或考虑通过 paper_ingest_tool 扩充语料库。"  # noqa: RUF001
    )
    # 英文问题的拒答文案(qa_agentic 按问题语言路由; 基准单中文文案的确认偏离)
    no_evidence_message_en: str = (
        "No relevant content was found in the indexed papers. Please check that "
        "the question matches the indexed corpus, or ingest more papers via "
        "paper_ingest_tool."
    )


class _IntentTier(BaseModel):
    """单档意图的检索力度。"""

    top_k: int = Field(default=10, ge=1)
    max_iter: int = Field(default=2, ge=1)
    rrf_k: int = Field(default=60, ge=1)


class _Intent(BaseModel):
    """意图分类的三档检索力度。

    基准把 top_k 5/10/15 与 max_iter 1/2/3 硬编码在 `rag/intent_classifier.py`
    的 `_DEFAULTS` 里; 搬到配置以遵守"不硬编码可调项"。缺省值与基准逐项一致,
    偏离只搬家不改行为。`enabled: false` 时跳过 LLM 调用直接走本地启发式。
    """

    enabled: bool = True
    factual: _IntentTier = Field(default_factory=lambda: _IntentTier(top_k=5, max_iter=1))
    reasoning: _IntentTier = Field(default_factory=lambda: _IntentTier(top_k=10, max_iter=2))
    explore: _IntentTier = Field(default_factory=lambda: _IntentTier(top_k=15, max_iter=3))


class _Rag(BaseModel):
    max_inner_iters: int = 3
    max_inner_tokens: int = 8000
    enable_hyde: bool = True
    enable_reflect: bool = True
    qa_cache_enabled: bool = False
    qa_cache_ttl_hours: int = 24
    intent: _Intent = Field(default_factory=_Intent)
    abstain: _Abstain = Field(default_factory=_Abstain)


class _EvaluationRagas(BaseModel):
    golden_set: str = "tests/fixtures/evaluation/ragas_golden.json"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    judge_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    max_concurrency: int = Field(default=4, ge=1, le=16)
    timeout_sec: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cache_dir: str = "data/evaluation/ragas_cache"
    judge_options: dict[str, Any] = Field(default_factory=dict)


class _Evaluation(BaseModel):
    enabled: bool = False
    provider: str = "custom"
    backends: list[str] = Field(default_factory=lambda: ["custom"])
    metrics: list[str] = Field(default_factory=lambda: ["hit_rate", "mrr", "recall"])
    golden_set: str = "tests/fixtures/evaluation/golden.json"
    retrieval_golden_set: str = "tests/fixtures/evaluation/retrieval_golden.json"
    top_k: int = Field(default=8, ge=1)
    max_concurrency: int = Field(default=8, ge=1, le=32)
    fail_on_error: bool = False
    ragas: _EvaluationRagas = Field(default_factory=_EvaluationRagas)


class _LlmTemperatures(BaseModel):
    """Per-role LLM temperatures.

    Picked once based on offline calibration; centralised here so that a
    single config tweak rolls out to every call site (qa_agentic, qa_stream,
    deliver/survey, deliver/latex_bib, wiki/flow, query_rewrite).
    """

    answer: float = 0.2  # qa_agentic main answer
    stream: float = 0.2  # qa_stream main answer
    rewrite: float = 0.3  # query_rewrite — wider for paraphrase diversity
    wiki: float = 0.2  # wiki concept create / patch
    survey: float = 0.3  # deliver/survey_md narrative
    latex_bib: float = 0.3  # deliver/latex_bib narrative


class _Llm(BaseModel):
    provider: str = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    chat_model: str | None = None
    small_model: str | None = None
    temperatures: _LlmTemperatures = Field(default_factory=_LlmTemperatures)
    # OpenAI 兼容供应商的私有参数, 非空时由 rag/llm.py 透传给 chat.completions.create。
    # Qwen(DashScope)思考型模型非流式调用必须 {enable_thinking: false}, 否则 400。
    extra_body: dict[str, Any] = Field(default_factory=dict)


class _Vision(BaseModel):
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_sec: int = 60
    max_concurrency: int = Field(default=4, ge=1, le=32)
    # 智谱(GLM-4.6V)OpenAI 兼容口径: temperature 区间为 (0,1) 开区间, 0 不适用,
    # 故不照抄基准 api.py 写死的 temperature=0。
    temperature: float = Field(default=0.01, gt=0.0, lt=1.0)
    # OpenAI 兼容供应商私有参数, 非空时透传给 chat.completions.create。
    # GLM-V 属思考型系列, 若吐 reasoning_content 或报 400 则填
    # {"thinking": {"type": "disabled"}}。
    extra_body: dict[str, Any] = Field(default_factory=dict)
    fallback_local: bool = False
    local_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_max_new_tokens: int = Field(default=256, ge=1, le=700)
    max_images_per_paper: int = 40
    max_image_bytes: int = 8_000_000
    cache: bool = True
    cache_dir: str = "./data/index/vision_cache"


class _WikiResolve(BaseModel):
    """三级概念解析阈值。向量只负责召回, 合并判定权在 LLM 验证。"""

    recall_floor: float = 0.60  # 低于此相似度不进候选(novel)
    auto_merge_same_lang: float = 0.90  # 同语言且高于此值才允许免验证直接 match
    short_label_max_ascii_chars: int = 4  # RL/CL/GAN/BERT 级缩写: 不许单独触发合并
    short_label_max_cjk_chars: int = 1  # 中文单字才算短; 两字词已是完整概念


class _WikiExtract(BaseModel):
    max_concepts: int = 5
    char_budget_zh: int = 4000  # 中文信息密度高, 字符预算低于英文
    char_budget_en: int = 6000
    exclude_sections: list[str] = Field(
        default_factory=lambda: ["references", "bibliography", "参考文献"]
    )


class _WikiQualityGate(BaseModel):
    """入队前过滤解析质量差的文档(如 mineru+broken 的征文通知), 防垃圾词条。"""

    min_chunks: int = 15
    skip_parsed_with: list[str] = Field(default_factory=lambda: ["mineru+broken", "pymupdf+broken"])


class _WikiConsistency(BaseModel):
    min_definition_chars_en: int = 20
    min_definition_chars_zh: int = 12  # 中文信息密度高, 12 汉字已是完整定义


class _WikiWorker(BaseModel):
    batch_size: int = 10
    concurrency: int = Field(default=4, ge=1, le=32)
    max_attempts: int = 3
    retry_backoff_sec: list[int] = Field(default_factory=lambda: [60, 600, 3600])


class _WikiLlm(BaseModel):
    """Wiki 专用 OpenAI 兼容模型; 未配置的连接字段回退到全局 llm。"""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    timeout_sec: float = Field(default=120.0, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dedicated_endpoint(self) -> _WikiLlm:
        if bool(self.base_url) != bool(self.api_key):
            raise ValueError("wiki.llm.base_url and api_key must be configured together")
        if self.base_url and not self.model:
            raise ValueError("wiki.llm.model is required for a dedicated endpoint")
        return self


class _Wiki(BaseModel):
    enabled: bool = True
    self_eval_threshold: float = 0.7
    definition_rewrite_lock_hours: int = 24  # 只锁昂贵的定义重写; 关系新增不受限
    llm: _WikiLlm = Field(default_factory=_WikiLlm)
    resolve: _WikiResolve = Field(default_factory=_WikiResolve)
    extract: _WikiExtract = Field(default_factory=_WikiExtract)
    quality_gate: _WikiQualityGate = Field(default_factory=_WikiQualityGate)
    consistency: _WikiConsistency = Field(default_factory=_WikiConsistency)
    worker: _WikiWorker = Field(default_factory=_WikiWorker)


class _McpResources(BaseModel):
    gpu_total: int = Field(default=1, ge=1)
    embedding: int = Field(default=1, ge=1)
    reranker: int = Field(default=1, ge=1)
    vision: int = Field(default=2, ge=1)
    mineru: int = Field(default=1, ge=1)
    llm: int = Field(default=4, ge=1)
    sqlite_write: int = Field(default=1, ge=1)


class _McpTimeouts(BaseModel):
    sqlite_sec: float = Field(default=5, gt=0)
    qdrant_sec: float = Field(default=15, gt=0)
    llm_sec: float = Field(default=30, gt=0)
    embedding_sec: float = Field(default=30, gt=0)
    reranker_sec: float = Field(default=30, gt=0)
    external_http_sec: float = Field(default=60, gt=0)


class _Mcp(BaseModel):
    profile: Literal["default", "admin"] = "default"
    admission_timeout_sec: float = Field(default=2, gt=0)
    retrieval_timeout_sec: float = Field(default=90, gt=0)
    trace_ttl_sec: float = Field(default=1800, gt=0)
    trace_max_entries: int = Field(default=1000, ge=1)
    max_running_retrievals: int = Field(default=2, ge=1)
    max_queued_retrievals: int = Field(default=8, ge=0)
    thread_tokens: int = Field(default=8, ge=1)
    resources: _McpResources = Field(default_factory=_McpResources)
    timeouts: _McpTimeouts = Field(default_factory=_McpTimeouts)


class _Logging(BaseModel):
    level: str = "INFO"
    json_format: bool = Field(default=False, alias="json")

    model_config = {"populate_by_name": True}


class AppConfig(BaseModel):
    paths: _Paths
    embedding: _Embedding = Field(default_factory=_Embedding)
    reranker: _Reranker = Field(default_factory=_Reranker)
    qdrant: _Qdrant = Field(default_factory=_Qdrant)
    mineru: _MinerU = Field(default_factory=_MinerU)
    chunk: _Chunk = Field(default_factory=_Chunk)
    retrieve: _Retrieve = Field(default_factory=_Retrieve)
    rag: _Rag = Field(default_factory=_Rag)
    evaluation: _Evaluation = Field(default_factory=_Evaluation)
    llm: _Llm = Field(default_factory=_Llm)
    vision: _Vision = Field(default_factory=_Vision)
    wiki: _Wiki = Field(default_factory=_Wiki)
    mcp: _Mcp = Field(default_factory=_Mcp)
    logging: _Logging = Field(default_factory=_Logging)


def _expand_env(value: Any) -> Any:
    """Recursively replace `$VAR` strings with environment values (or None)."""
    if isinstance(value, str) and value.startswith("$"):
        return os.environ.get(value[1:])
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _resolve_paths(raw: dict[str, Any]) -> dict[str, Any]:
    """Make path strings absolute relative to project root."""
    paths = raw.get("paths", {})
    for k, v in list(paths.items()):
        if isinstance(v, str) and v.startswith("./"):
            paths[k] = str((PROJECT_ROOT / v[2:]).resolve())
    raw["paths"] = paths
    return raw


@lru_cache(maxsize=1)
def load(path: str | Path | None = None) -> AppConfig:
    """Load and cache application config.

    Override order:
      1. Explicit `path` argument
      2. Env var `PAPER_RAG_CONFIG` (absolute or relative to project root)
      3. `config/default.yaml`
    """
    if path is not None:
        cfg_path = Path(path)
    elif os.environ.get("PAPER_RAG_CONFIG"):
        env_path = Path(os.environ["PAPER_RAG_CONFIG"])
        cfg_path = env_path if env_path.is_absolute() else (PROJECT_ROOT / env_path)
    else:
        cfg_path = DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env(raw)
    raw = _resolve_paths(raw)
    return AppConfig.model_validate(raw)


__all__ = ["PROJECT_ROOT", "AppConfig", "load"]
