"""vision/enrich.py 边界契约: 语言贯通、正门取料、守卫与 fail-open。

summarizer 全部打桩(不触网), 图片用真实临时文件。
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from paper_rag.vision import enrich as en
from paper_rag.vision.cache import VisionSummaryCache
from paper_rag.vision.schema import (
    STATUS_CACHED,
    STATUS_FAILED,
    STATUS_FALLBACK,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    VisualSummaryRequest,
    VisualSummaryResult,
)


@pytest.fixture
def png(tmp_path: Path) -> Path:
    p = tmp_path / "fig.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    return p


def _chunk(png: Path, **over) -> dict:
    base: dict = {
        "chunk_id": "c1",
        "paper_id": "p1",
        "modality": "figure",
        "text": "Figure: accuracy\nContext: we compare\nPath: images/fig.png",
        "context_text": "[Title: T] [Section: S]\nFigure: accuracy",
        "asset_path": str(png),
        "metadata": {},
    }
    base.update(over)
    return base


def _ok(summary: str = "Visual type: line chart"):
    return lambda req: VisualSummaryResult(
        status=STATUS_OK, summary=summary, provider="api", model="glm-4.6v"
    )


def _fail(err: str = "boom"):
    return lambda req: VisualSummaryResult(
        status=STATUS_FAILED, provider="api", model="glm-4.6v", error=err
    )


# --- 取料: metadata 正门优先, 文本反解兜底 ----------------------------------


def test_request_prefers_metadata_over_text_parsing(png):
    chunk = _chunk(
        png,
        metadata={"caption": "图 1 真实图注", "surrounding_context": "真实上下文"},
    )
    req = en.request_from_chunk(chunk, language="zh")
    assert req.caption == "图 1 真实图注"
    assert req.surrounding_context == "真实上下文"
    assert req.language == "zh"


def test_text_fallback_parses_en_prefixes(png):
    req = en.request_from_chunk(_chunk(png), language="en")
    assert req.caption == "accuracy"
    assert req.surrounding_context == "we compare"


def test_text_fallback_parses_zh_prefixes(png):
    chunk = _chunk(png, text="图: 准确率对比\n上下文: 我们比较了深度\n路径: images/fig.png")
    req = en.request_from_chunk(chunk, language="zh")
    assert req.caption == "准确率对比"
    assert req.surrounding_context == "我们比较了深度"


def test_text_fallback_accepts_fullwidth_colon(png):
    chunk = _chunk(png, text="图：准确率对比\n上下文：我们比较了深度")  # noqa: RUF001
    req = en.request_from_chunk(chunk, language="zh")
    assert req.caption == "准确率对比"
    assert req.surrounding_context == "我们比较了深度"


def test_table_caption_on_next_line_is_recovered(png):
    # 基准缺陷: compose_table_text 是 "表:\n{content}", 同行取值恒空。
    chunk = _chunk(
        png,
        modality="table",
        text="表:\n| 模型 | 准确率 |\n| GNN | 0.91 |\n上下文: 表 2 对比结果",
    )
    req = en.request_from_chunk(chunk, language="zh")
    assert "模型" in req.caption
    assert req.surrounding_context == "表 2 对比结果"


def test_context_fallback_truncates_when_no_prefix(png):
    chunk = _chunk(png, text="x" * 900)
    req = en.request_from_chunk(chunk, language=None)
    assert req.caption == ""
    assert len(req.surrounding_context) == 500


# --- 语言贯通 ---------------------------------------------------------------


def test_language_reaches_summarizer(png):
    seen: list[VisualSummaryRequest] = []

    def spy(req: VisualSummaryRequest) -> VisualSummaryResult:
        seen.append(req)
        return VisualSummaryResult(status=STATUS_OK, summary="视觉类型: 折线图", provider="api")

    en.enrich_chunks("p1", [_chunk(png)], summarizer=spy, cache_enabled=False, language="zh")
    assert seen[0].language == "zh"


def test_zh_summary_line_uses_chinese_label(png):
    chunks = en.enrich_chunks(
        "p1",
        [_chunk(png)],
        summarizer=_ok("视觉类型: 折线图"),
        cache_enabled=False,
        language="zh",
    )
    assert "视觉摘要: 视觉类型: 折线图" in chunks[0]["text"]
    assert "视觉摘要: 视觉类型: 折线图" in chunks[0]["context_text"]
    assert "Visual summary" not in chunks[0]["text"]


def test_en_summary_line_keeps_baseline_label(png):
    chunks = en.enrich_chunks(
        "p1", [_chunk(png)], summarizer=_ok(), cache_enabled=False, language="en"
    )
    assert "Visual summary: Visual type: line chart" in chunks[0]["text"]


def test_none_language_uses_english_label(png):
    chunks = en.enrich_chunks("p1", [_chunk(png)], summarizer=_ok(), cache_enabled=False)
    assert "Visual summary:" in chunks[0]["text"]


def test_summary_appended_once_on_repeated_enrich(png):
    chunk = _chunk(png)
    for _ in range(2):
        en.enrich_chunks("p1", [chunk], summarizer=_ok(), cache_enabled=False)
    assert chunk["text"].count("Visual summary:") == 1


# --- 守卫与 fail-open -------------------------------------------------------


def test_formula_chunk_is_untouched_byte_for_byte(png):
    formula = _chunk(png, modality="formula", chunk_id="f1", asset_path=None)
    before = dict(formula)
    en.enrich_chunks("p1", [formula], summarizer=_ok(), cache_enabled=False)
    assert formula["text"] == before["text"]
    assert formula["context_text"] == before["context_text"]
    assert formula["metadata"] == {}


def test_text_chunk_is_untouched(png):
    text_chunk = _chunk(png, modality="text", chunk_id="t1")
    before = text_chunk["text"]
    en.enrich_chunks("p1", [text_chunk], summarizer=_ok(), cache_enabled=False)
    assert text_chunk["text"] == before
    assert "visual_summary_status" not in text_chunk["metadata"]


def test_other_paper_chunks_skipped(png):
    other = _chunk(png, paper_id="p2")
    en.enrich_chunks("p1", [other], summarizer=_ok(), cache_enabled=False)
    assert other["metadata"] == {}


def test_failure_keeps_original_text_and_records_status(png):
    chunk = _chunk(png)
    before = chunk["text"]
    en.enrich_chunks(
        "p1",
        [chunk],
        summarizer=_fail("400 bad request"),
        fallback_summarizer=lambda r: VisualSummaryResult(
            status=STATUS_UNAVAILABLE, provider="local"
        ),
        cache_enabled=False,
    )
    assert chunk["text"] == before  # fail-open: 原文完整保留
    assert chunk["metadata"]["visual_summary_status"] == STATUS_FAILED
    assert "400 bad request" in chunk["metadata"]["visual_summary_error"]
    assert chunk["metadata"]["visual_summary_provider"] == "api"  # 失败也要能定位来源
    assert "visual_summary" not in chunk["metadata"]


def test_fallback_promoted_when_primary_fails(png):
    chunk = _chunk(png)
    en.enrich_chunks(
        "p1",
        [chunk],
        summarizer=_fail(),
        fallback_summarizer=lambda r: VisualSummaryResult(
            status=STATUS_OK, summary="local said so", provider="local"
        ),
        cache_enabled=False,
    )
    assert chunk["metadata"]["visual_summary_status"] == STATUS_FALLBACK
    assert "local said so" in chunk["text"]


def test_missing_asset_path_is_skipped(png):
    chunk = _chunk(png, asset_path=None)
    en.enrich_chunks("p1", [chunk], summarizer=_ok(), cache_enabled=False)
    assert chunk["metadata"]["visual_summary_status"] == STATUS_SKIPPED
    assert "asset_path" in chunk["metadata"]["visual_summary_error"]


def test_nonexistent_asset_is_skipped(tmp_path):
    chunk = _chunk(tmp_path / "gone.png")
    en.enrich_chunks("p1", [chunk], summarizer=_ok(), cache_enabled=False)
    assert chunk["metadata"]["visual_summary_status"] == STATUS_SKIPPED


def test_oversized_image_is_skipped(png):
    chunk = _chunk(png)
    en.enrich_chunks("p1", [chunk], summarizer=_ok(), cache_enabled=False, max_image_bytes=1)
    assert chunk["metadata"]["visual_summary_status"] == STATUS_SKIPPED
    assert "max_image_bytes" in chunk["metadata"]["visual_summary_error"]


def test_max_images_per_paper_enforced(png):
    a, b = _chunk(png), _chunk(png, chunk_id="c2")
    en.enrich_chunks("p1", [a, b], summarizer=_ok(), cache_enabled=False, max_images_per_paper=1)
    assert a["metadata"]["visual_summary_status"] == STATUS_OK
    assert b["metadata"]["visual_summary_status"] == STATUS_SKIPPED
    assert "max_images_per_paper" in b["metadata"]["visual_summary_error"]


def test_summarizer_exception_does_not_break_ingest(png):
    chunk = _chunk(png)

    def boom(req):
        raise RuntimeError("network down")

    en.enrich_chunks("p1", [chunk], summarizer=boom, cache_enabled=False)
    assert chunk["metadata"]["visual_summary_status"] == STATUS_FAILED
    assert "network down" in chunk["metadata"]["visual_summary_error"]


def _stub_vision_cfg(monkeypatch, **over) -> None:
    """钉住 vision 配置, 不依赖环境——生产配置已 enabled=true 且凭据在场,
    纯逻辑测试若读环境配置会真打 API(2026-08-05 用户开启生产增强后实翻车)。"""
    from types import SimpleNamespace

    base = {
        "enabled": False,
        "base_url": None,
        "api_key": None,
        "model": None,
        "timeout_sec": 60,
        "temperature": 0.01,
        "extra_body": {},
        "fallback_local": False,
        "local_model": "stub",
        "cache": False,
        "cache_dir": "unused",
        "max_images_per_paper": 40,
        "max_image_bytes": 8_000_000,
        "max_concurrency": 4,
    }
    base.update(over)
    stub = SimpleNamespace(vision=SimpleNamespace(**base))
    monkeypatch.setattr(en.cfg, "load", lambda: stub)


def test_disabled_config_without_injection_returns_chunks_untouched(png, monkeypatch):
    _stub_vision_cfg(monkeypatch, enabled=False)
    chunk = _chunk(png)
    before = chunk["text"]
    out = en.enrich_chunks("p1", [chunk])
    assert out[0]["text"] == before
    assert chunk["metadata"] == {}


def test_enabled_config_with_missing_credentials_skips_untouched(png, monkeypatch):
    # enabled=true 但凭据不全: 打 warning 后原样返回, 不得半途构造 summarizer。
    _stub_vision_cfg(monkeypatch, enabled=True, base_url="https://x", api_key=None, model="m")
    chunk = _chunk(png)
    before = chunk["text"]
    out = en.enrich_chunks("p1", [chunk])
    assert out[0]["text"] == before
    assert chunk["metadata"] == {}


# --- 缓存联动 ---------------------------------------------------------------


def test_cache_hit_marks_cached_and_skips_summarizer(tmp_path, png):
    cache = VisionSummaryCache(tmp_path / "vc")
    calls: list[int] = []

    def once(req):
        calls.append(1)
        return VisualSummaryResult(status=STATUS_OK, summary="视觉类型: 柱状图", provider="api")

    first = _chunk(png)
    en.enrich_chunks("p1", [first], summarizer=once, cache=cache, cache_enabled=True, language="zh")
    second = _chunk(png)
    en.enrich_chunks(
        "p1", [second], summarizer=once, cache=cache, cache_enabled=True, language="zh"
    )
    assert len(calls) == 1
    assert second["metadata"]["visual_summary_status"] == STATUS_CACHED
    assert "视觉摘要: 视觉类型: 柱状图" in second["context_text"]


def test_cache_miss_across_languages(tmp_path, png):
    cache = VisionSummaryCache(tmp_path / "vc")
    calls: list[str | None] = []

    def spy(req):
        calls.append(req.language)
        return VisualSummaryResult(status=STATUS_OK, summary="s", provider="api")

    en.enrich_chunks(
        "p1", [_chunk(png)], summarizer=spy, cache=cache, cache_enabled=True, language="zh"
    )
    en.enrich_chunks(
        "p1", [_chunk(png)], summarizer=spy, cache=cache, cache_enabled=True, language="en"
    )
    assert calls == ["zh", "en"]  # 语言入键, 不发生跨语言脏命中


def test_parallel_completion_commits_chunks_and_cache_in_input_order(tmp_path, png, monkeypatch):
    """API 即使逆序返回, chunk 修改和缓存写入也只能由主线程原序提交。"""
    chunks = [_chunk(png, chunk_id=f"c{i}") for i in range(1, 4)]
    all_started = threading.Barrier(3)
    c3_done = threading.Event()
    c2_done = threading.Event()
    completion_order: list[str] = []
    main_thread = threading.get_ident()

    def deliberately_reversed(req):
        all_started.wait(timeout=2)
        if req.chunk_id == "c3":
            completion_order.append(req.chunk_id)
            c3_done.set()
        elif req.chunk_id == "c2":
            assert c3_done.wait(timeout=2)
            completion_order.append(req.chunk_id)
            c2_done.set()
        else:
            assert c2_done.wait(timeout=2)
            completion_order.append(req.chunk_id)
        return VisualSummaryResult(
            status=STATUS_OK,
            summary=req.chunk_id,
            provider="api",
        )

    mutation_order: list[str] = []
    mutation_threads: list[int] = []
    original_record = en._record

    def record_in_order(chunk, status, result, error=""):
        mutation_order.append(chunk["chunk_id"])
        mutation_threads.append(threading.get_ident())
        original_record(chunk, status, result, error)

    monkeypatch.setattr(en, "_record", record_in_order)

    class TrackingCache(VisionSummaryCache):
        def __init__(self, cache_dir):
            super().__init__(cache_dir)
            self.write_order: list[str] = []
            self.write_threads: list[int] = []

        def write(self, key, result):
            self.write_order.append(result.summary)
            self.write_threads.append(threading.get_ident())
            super().write(key, result)

    cache = TrackingCache(tmp_path / "vc")
    en.enrich_chunks(
        "p1",
        chunks,
        summarizer=deliberately_reversed,
        cache=cache,
        cache_enabled=True,
        max_concurrency=3,
    )

    assert completion_order == ["c3", "c2", "c1"]
    assert mutation_order == ["c1", "c2", "c3"]
    assert cache.write_order == ["c1", "c2", "c3"]
    assert mutation_threads == [main_thread] * 3
    assert cache.write_threads == [main_thread] * 3
    assert [c["metadata"]["visual_summary"] for c in chunks] == ["c1", "c2", "c3"]
