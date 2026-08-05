"""vision/api.py 边界契约: 双语提示词、GLM 参数口径、JSON 渲染。

用 client_factory 注入假客户端, 不触网; 只证接口设计与语言路由。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from paper_rag.vision import api
from paper_rag.vision.schema import STATUS_FAILED, STATUS_OK, VisualSummaryRequest


class _FakeCompletions:
    def __init__(self, content: str, exc: Exception | None = None):
        self.content = content
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        msg = type("M", (), {"content": self.content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content: str = "{}", exc: Exception | None = None):
        self.completions = _FakeCompletions(content, exc)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _png(tmp_path: Path) -> Path:
    p = tmp_path / "fig.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    return p


def _req(tmp_path: Path, **over) -> VisualSummaryRequest:
    base: dict[str, Any] = {
        "paper_id": "p1",
        "chunk_id": "c1",
        "modality": "figure",
        "asset_path": _png(tmp_path),
        "caption": "Figure 1 accuracy vs depth",
        "surrounding_context": "we compare depths",
        "model": "glm-4.6v",
    }
    base.update(over)
    return VisualSummaryRequest(**base)


def _summarizer(client: _FakeClient, **over) -> api.OpenAIVisionSummarizer:
    kwargs: dict[str, Any] = {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key": "k",
        "model": "glm-4.6v",
        "client_factory": lambda: client,
    }
    kwargs.update(over)
    return api.OpenAIVisionSummarizer(**kwargs)


# --- 语言路由 ---------------------------------------------------------------


def test_prompt_for_zh_demands_chinese_prose_and_keeps_latin_terms():
    p = api.prompt_for("zh")
    assert "中文" in p
    assert "key_entities" in p  # JSON 机器键保持英文
    assert "无法辨认" in p  # not legible 的中文对应文案
    # 术语原文形态要求(query_rewrite 双语过桥在语料侧的镜像)
    assert "原文" in p


def test_prompt_for_en_and_none_are_identical_english():
    assert api.prompt_for("en") == api.prompt_for(None)
    assert "academic paper" in api.prompt_for(None)
    assert "not legible" in api.prompt_for(None)


def test_unknown_language_falls_back_to_english_without_guessing():
    assert api.prompt_for("de") == api.prompt_for(None)


def test_message_carries_zh_prompt_and_zh_field_labels(tmp_path):
    s = _summarizer(_FakeClient())
    msg = s._message(_req(tmp_path, language="zh", caption="图 1 准确率"))
    text = msg["content"][0]["text"]
    assert "中文" in text
    assert "图注: 图 1 准确率" in text
    assert "论文 id:" in text


def test_message_embeds_base64_data_url(tmp_path):
    s = _summarizer(_FakeClient())
    req = _req(tmp_path)
    msg = s._message(req)
    url = msg["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == req.asset_path.read_bytes()


def test_missing_caption_renders_language_specific_none_marker(tmp_path):
    s = _summarizer(_FakeClient())
    zh = s._message(_req(tmp_path, language="zh", caption="", surrounding_context=""))
    en = s._message(_req(tmp_path, caption="", surrounding_context=""))
    assert "(无)" in zh["content"][0]["text"]
    assert "(none)" in en["content"][0]["text"]


# --- GLM 参数口径 -----------------------------------------------------------


def test_temperature_is_strictly_positive_for_zhipu(tmp_path):
    # 智谱 OpenAI 兼容: temperature 区间 (0,1), 0 会被拒。
    client = _FakeClient(content="{}")
    _summarizer(client, temperature=0.01).summarize(_req(tmp_path))
    assert client.completions.calls[0]["temperature"] == 0.01


def test_extra_body_passed_through_when_non_empty(tmp_path):
    client = _FakeClient(content="{}")
    thinking = {"thinking": {"type": "disabled"}}
    _summarizer(client, extra_body=thinking).summarize(_req(tmp_path))
    assert client.completions.calls[0]["extra_body"] == thinking


def test_extra_body_absent_from_call_when_empty(tmp_path):
    # 空表缺省时调用形参与基准逐键一致。
    client = _FakeClient(content="{}")
    _summarizer(client).summarize(_req(tmp_path))
    call = client.completions.calls[0]
    assert "extra_body" not in call
    assert set(call) == {"model", "messages", "temperature", "max_tokens", "timeout"}


# --- 输出解析 ---------------------------------------------------------------


def test_json_payload_renders_zh_labels():
    payload = {"visual_type": "折线图", "main_message": "准确率随深度上升"}
    out = api.summary_from_payload(payload, language="zh")
    assert "视觉类型: 折线图" in out
    assert "主要信息: 准确率随深度上升" in out


def test_json_payload_renders_en_labels_by_default():
    out = api.summary_from_payload({"visual_type": "line chart"}, language=None)
    assert "Visual type: line chart" in out


def test_payload_list_values_joined_and_empty_dropped():
    out = api.summary_from_payload(
        {"key_entities": ["ResNet", "ImageNet"], "main_message": "", "supports_claim": None},
        language="en",
    )
    assert "Key entities: ResNet, ImageNet" in out
    assert "Main message" not in out
    assert "Supports claim" not in out


def test_payload_without_known_keys_falls_back_to_json_dump():
    out = api.summary_from_payload({"weird": "值"}, language="zh")
    assert json.loads(out) == {"weird": "值"}  # 中文不转义


def test_summarize_ok_with_fenced_json(tmp_path):
    client = _FakeClient(content='```json\n{"visual_type": "bar chart"}\n```')
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_OK
    assert "Visual type: bar chart" in res.summary
    assert res.provider == "api"
    assert res.raw == {"visual_type": "bar chart"}


def test_summarize_ok_with_plain_text_content(tmp_path):
    client = _FakeClient(content="  a line chart of accuracy  ")
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_OK
    assert res.summary == "a line chart of accuracy"
    assert res.raw is None


def test_summarize_failure_is_captured_not_raised(tmp_path):
    client = _FakeClient(exc=RuntimeError("400 temperature out of range"))
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_FAILED
    assert "temperature out of range" in (res.error or "")
    assert res.summary == ""


def test_empty_content_is_failed_with_diagnostic(tmp_path):
    # GLM-4.6V 默认思考: reasoning 吃光 max_tokens 时 content 为空,
    # 必须给出可排障的 error, 不得无声 failed。
    client = _FakeClient(content="")
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_FAILED
    assert "empty content" in (res.error or "")
    assert "thinking" in (res.error or "")


def test_truncated_json_content_is_failed_not_indexed(tmp_path):
    # finish_reason=length 的截断 JSON 不得作为摘要进索引。
    client = _FakeClient(content='{"visual_type": "diagram", "main_mess')
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_FAILED
    assert "truncated" in (res.error or "")
    assert res.summary == ""


def test_truncated_fenced_json_is_failed(tmp_path):
    client = _FakeClient(content='```json\n{"visual_type": "diag')
    res = _summarizer(client).summarize(_req(tmp_path))
    assert res.status == STATUS_FAILED


def test_missing_asset_file_is_captured_as_failure(tmp_path):
    client = _FakeClient()
    req = _req(tmp_path, asset_path=tmp_path / "nope.png")
    res = _summarizer(client).summarize(req)
    assert res.status == STATUS_FAILED
