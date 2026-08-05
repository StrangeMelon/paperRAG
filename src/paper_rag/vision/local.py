"""本地视觉模型兜底(Qwen2.5-VL 兼容), 默认关闭。

结构照抄基准, 两处收敛: 提示词经 ``build_local_prompt`` 与 ``api.py`` 同源
(双语模板不分叉), 渲染同样走 ``summary_from_payload(language=...)``。

重型依赖(transformers / torch / PIL)全在方法内导入; 依赖缺失时返回
``unavailable`` 而不是抛错——vision-local 是可选 extra, 未装不该打断 ingest。
真实 GPU 验收推迟(默认 ``vision.fallback_local: false``)。
"""

from __future__ import annotations

from .. import config as cfg
from .api import _loads_json, _request_labels, prompt_for, summary_from_payload
from .schema import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    VisualSummaryRequest,
    VisualSummaryResult,
)


def build_local_prompt(request: VisualSummaryRequest) -> str:
    """本地模型的纯文本提示词; 与 API 侧共用模板与语言路由。"""
    lb = _request_labels(request.language)
    return (
        f"{prompt_for(request.language)}\n\n"
        f"{lb['caption']}: {request.caption or lb['none']}\n"
        f"{lb['context']}: {request.surrounding_context or lb['none']}"
    )


class LocalVisionSummarizer:
    """Qwen2.5-VL 兼容兜底; 依赖不存在时保持 unavailable。"""

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = model
        self._model = None
        self._processor = None

    def summarize(self, request: VisualSummaryRequest) -> VisualSummaryResult:
        try:
            self._ensure_loaded()
        except Exception as exc:
            return VisualSummaryResult(
                status=STATUS_UNAVAILABLE,
                provider="local",
                model=self.model,
                error=str(exc),
            )

        try:
            from PIL import Image

            image = Image.open(request.asset_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": build_local_prompt(request)},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            inputs = inputs.to(self._model.device)
            output = self._model.generate(
                **inputs,
                max_new_tokens=cfg.load().vision.local_max_new_tokens,
            )
            decoded = self._processor.batch_decode(output, skip_special_tokens=True)[0]
            raw = _loads_json(decoded)
            return VisualSummaryResult(
                status=STATUS_OK,
                summary=(
                    summary_from_payload(raw, language=request.language) if raw else decoded.strip()
                ),
                provider="local",
                model=self.model,
                raw=raw,
            )
        except Exception as exc:
            return VisualSummaryResult(
                status=STATUS_FAILED,
                provider="local",
                model=self.model,
                error=str(exc),
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model,
            device_map="auto",
        )
