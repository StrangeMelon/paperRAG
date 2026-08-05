"""vision/local.py 边界契约: 依赖缺失即 unavailable, 语言路由与 api 同源。

本地视觉模型默认关闭(vision.fallback_local: false), 真实 GPU 验收推迟;
本模块只证"依赖不在时不炸穿 ingest"与提示词复用, 不加载任何权重。
"""

from __future__ import annotations

from pathlib import Path

from paper_rag.vision import api
from paper_rag.vision.local import LocalVisionSummarizer, build_local_prompt
from paper_rag.vision.schema import (
    STATUS_UNAVAILABLE,
    VisualSummaryRequest,
)


def _req(tmp_path: Path, **over) -> VisualSummaryRequest:
    p = tmp_path / "fig.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    base: dict = {
        "paper_id": "p1",
        "chunk_id": "c1",
        "modality": "figure",
        "asset_path": p,
        "caption": "图 1 准确率",
        "surrounding_context": "上下文",
    }
    base.update(over)
    return VisualSummaryRequest(**base)


def test_prompt_reuses_api_templates_zh(tmp_path):
    # 提示词单一来源: 与 api.py 同源, 避免双语模板分叉。
    prompt = build_local_prompt(_req(tmp_path, language="zh"))
    assert api.prompt_for("zh") in prompt
    assert "图注: 图 1 准确率" in prompt


def test_prompt_reuses_api_templates_en(tmp_path):
    prompt = build_local_prompt(_req(tmp_path, caption="Figure 1", surrounding_context=""))
    assert api.prompt_for(None) in prompt
    assert "Caption: Figure 1" in prompt
    assert "(none)" in prompt


def test_unavailable_when_deps_missing(tmp_path, monkeypatch):
    s = LocalVisionSummarizer("Qwen/Qwen2.5-VL-7B-Instruct")

    def boom() -> None:
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(s, "_ensure_loaded", boom)
    res = s.summarize(_req(tmp_path))
    assert res.status == STATUS_UNAVAILABLE
    assert res.provider == "local"
    assert res.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert "transformers" in (res.error or "")


def test_summarize_never_raises_on_runtime_failure(tmp_path, monkeypatch):
    s = LocalVisionSummarizer()
    monkeypatch.setattr(s, "_ensure_loaded", lambda: None)
    res = s.summarize(_req(tmp_path, asset_path=tmp_path / "missing.png"))
    # 加载成功但推理阶段失败 -> failed, 且不抛
    assert res.status in {"failed", "unavailable"}
    assert res.summary == ""
