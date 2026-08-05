"""OpenAI 兼容的视觉摘要适配器(生产目标: 智谱 GLM-4.6V)。

相对基准三处确认偏离:
- **双语提示词**: zh 要求摘要正文用中文, 但 ``key_entities``(模型名/数据集/
  指标)保留原文形态——中文论文正文里这些术语本来就以拉丁形态出现, 译成中文
  反而与 BM25 词面脱节。这是 query_rewrite 课"zh 关键词中英混出"在语料侧的
  镜像。JSON 机器键始终英文, 解析逻辑不分叉。
- **temperature 不写死 0**: 智谱 OpenAI 兼容口径为 temperature ∈ (0,1) 开区间,
  ``temperature=0`` 会被拒, 故由 ``vision.temperature`` 提供(默认 0.01)。
- **extra_body 透传**: GLM-V 属思考型系列; 非空时透传, 空表则整个参数不出现,
  调用形参与基准逐键一致。

字段标签同样按语言渲染: 摘要文本会被追加进 chunk 并进入 FTS5/BM25, 英文标签
会在中文语料里形成词面噪声。
"""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .schema import STATUS_FAILED, STATUS_OK, VisualSummaryRequest, VisualSummaryResult

_PROMPT_EN = """You are summarizing a figure or visual table from an academic paper.
Use the image plus the provided caption and nearby paper context.
Do not invent exact numbers that are not legible.
If text or values are unclear, say "not legible".
Return compact JSON with:
- visual_type
- main_message
- axes_or_dimensions
- key_entities
- trends_or_comparisons
- supports_claim
- limitations_or_uncertainty
"""

_PROMPT_ZH = """你正在为一篇中文学术论文中的图或表格图片撰写摘要。
请结合图片本身、给出的图注与论文邻近上下文进行描述。
不要编造图中无法辨认的具体数值。
若文字或数值不清晰, 直接写"无法辨认"。
描述性内容一律用中文书写; 但模型名、数据集名、评价指标等专有术语必须保留
论文中的原文形态(例如 ResNet、ImageNet、BLEU), 不要翻译成中文。
以紧凑 JSON 返回, 键名保持英文:
- visual_type(视觉类型)
- main_message(主要信息)
- axes_or_dimensions(坐标轴或维度)
- key_entities(关键实体)
- trends_or_comparisons(趋势或对比)
- supports_claim(支持的论断)
- limitations_or_uncertainty(局限或不确定性)
"""

_FIELD_LABELS_EN = (
    ("visual_type", "Visual type"),
    ("main_message", "Main message"),
    ("axes_or_dimensions", "Axes or dimensions"),
    ("key_entities", "Key entities"),
    ("trends_or_comparisons", "Trends or comparisons"),
    ("supports_claim", "Supports claim"),
    ("limitations_or_uncertainty", "Limitations or uncertainty"),
)

_FIELD_LABELS_ZH = (
    ("visual_type", "视觉类型"),
    ("main_message", "主要信息"),
    ("axes_or_dimensions", "坐标轴或维度"),
    ("key_entities", "关键实体"),
    ("trends_or_comparisons", "趋势或对比"),
    ("supports_claim", "支持的论断"),
    ("limitations_or_uncertainty", "局限或不确定性"),
)

_REQUEST_LABELS_EN = {
    "paper": "Paper id",
    "chunk": "Chunk id",
    "modality": "Modality",
    "caption": "Caption",
    "context": "Nearby context",
    "none": "(none)",
}

_REQUEST_LABELS_ZH = {
    "paper": "论文 id",
    "chunk": "块 id",
    "modality": "元素类型",
    "caption": "图注",
    "context": "邻近上下文",
    "none": "(无)",
}


def prompt_for(language: str | None) -> str:
    """按语言取提示词; 未知语言不猜, 回落英文(与 contextual 课同款原则)。"""
    return _PROMPT_ZH if language == "zh" else _PROMPT_EN


def _field_labels(language: str | None):
    return _FIELD_LABELS_ZH if language == "zh" else _FIELD_LABELS_EN


def _request_labels(language: str | None) -> dict[str, str]:
    return _REQUEST_LABELS_ZH if language == "zh" else _REQUEST_LABELS_EN


class OpenAIVisionSummarizer:
    """把一张图片的摘要请求转成一次 OpenAI 兼容 chat.completions 调用。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int = 60,
        temperature: float = 0.01,
        extra_body: dict[str, Any] | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})
        self._client_factory = client_factory

    def summarize(self, request: VisualSummaryRequest) -> VisualSummaryResult:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [self._message(request)],
                "temperature": self.temperature,
                "max_tokens": 700,
                "timeout": self.timeout_sec,
            }
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            resp = self._client().chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            # 输出净化(真实 GLM-4.6V 实测): 默认思考模式下 reasoning 可能吃光
            # max_tokens, content 为空或 JSON 截断——两者都不得进索引。
            if not content:
                return VisualSummaryResult(
                    status=STATUS_FAILED,
                    provider="api",
                    model=self.model,
                    error=(
                        "empty content from vision api (thinking may have consumed "
                        "max_tokens; set vision.extra_body {thinking: {type: disabled}})"
                    ),
                )
            raw = _loads_json(content)
            if raw is None and content.startswith(("{", "```")):
                return VisualSummaryResult(
                    status=STATUS_FAILED,
                    provider="api",
                    model=self.model,
                    error="unparseable JSON from vision api (likely truncated by max_tokens)",
                )
            summary = summary_from_payload(raw, language=request.language) if raw else content
            return VisualSummaryResult(
                status=STATUS_OK,
                summary=summary,
                provider="api",
                model=self.model,
                raw=raw,
            )
        except Exception as exc:  # 视觉增强不得打断 ingest, 失败一律记账返回
            return VisualSummaryResult(
                status=STATUS_FAILED,
                provider="api",
                model=self.model,
                error=str(exc),
            )

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_sec)

    def _message(self, request: VisualSummaryRequest) -> dict[str, Any]:
        lb = _request_labels(request.language)
        text = (
            f"{prompt_for(request.language)}\n\n"
            f"{lb['paper']}: {request.paper_id}\n"
            f"{lb['chunk']}: {request.chunk_id}\n"
            f"{lb['modality']}: {request.modality}\n"
            f"{lb['caption']}: {request.caption or lb['none']}\n"
            f"{lb['context']}: {request.surrounding_context or lb['none']}"
        )
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": _data_url(request.asset_path)}},
            ],
        }


def _data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _loads_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def summary_from_payload(payload: dict[str, Any], *, language: str | None = None) -> str:
    """把 JSON 字段渲染成多行摘要; 标签按语言路由(标签本身也进 BM25 索引)。"""
    parts: list[str] = []
    for key, label in _field_labels(language):
        value = payload.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{label}: {value}")
    if parts:
        return "\n".join(parts)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
