"""observability 指标与 trace id 的行为契约测试(纯逻辑, 无外部依赖)。

切片 0: counter(累加、labels 区分序列、无 labels)。
切片 1: histogram(observe 累积、time() 上下文计时)。
切片 2: snapshot(JSON 可序列化、分位数)与 reset。
切片 3: render(Prometheus 文本格式: counter/histogram bucket/sum/count)。
切片 4: new_trace_id(16 位 hex、唯一性)。
"""

from __future__ import annotations

import pytest

from paper_rag.observability import counter, histogram, new_trace_id, render, reset, snapshot


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset()
    yield
    reset()


# ---------- 切片 0: counter ----------


def test_counter_accumulates():
    counter("qa_total").inc()
    counter("qa_total").inc(2)
    snap = snapshot()
    assert snap["counters"] == [{"name": "qa_total", "labels": {}, "value": 3.0}]


def test_counter_labels_are_distinct_series():
    counter("qa_total", {"stop": "answered"}).inc()
    counter("qa_total", {"stop": "no_chunks"}).inc()
    values = {tuple(sorted(c["labels"].items())): c["value"] for c in snapshot()["counters"]}
    assert values[(("stop", "answered"),)] == 1.0
    assert values[(("stop", "no_chunks"),)] == 1.0


# ---------- 切片 1: histogram ----------


def test_histogram_observe_and_summary():
    h = histogram("latency_seconds")
    for v in (0.1, 0.2, 0.3):
        h.observe(v)
    [entry] = snapshot()["histograms"]
    assert entry["count"] == 3
    assert entry["sum"] == pytest.approx(0.6)


def test_histogram_time_context_manager():
    with histogram("latency_seconds").time():
        pass
    [entry] = snapshot()["histograms"]
    assert entry["count"] == 1
    assert entry["sum"] >= 0.0


# ---------- 切片 2: snapshot / reset ----------


def test_reset_clears_everything():
    counter("a").inc()
    histogram("b").observe(1.0)
    reset()
    snap = snapshot()
    assert snap["counters"] == [] and snap["histograms"] == []


def test_snapshot_quantiles_present():
    h = histogram("lat")
    for v in range(1, 101):
        h.observe(v / 100)
    [entry] = snapshot()["histograms"]
    assert entry["p50"] == pytest.approx(0.51, abs=0.02)
    assert entry["p95"] == pytest.approx(0.96, abs=0.02)


# ---------- 切片 3: render ----------


def test_render_prometheus_text_format():
    counter("qa_total", {"intent": "factual"}).inc(2)
    histogram("lat").observe(0.2)
    text = render()
    assert "# TYPE qa_total counter" in text
    assert 'qa_total{intent="factual"} 2.0' in text
    assert "# TYPE lat histogram" in text
    assert 'lat_bucket{le="0.25"} 1' in text
    assert "lat_sum 0.2" in text
    assert "lat_count 1" in text


# ---------- 切片 4: new_trace_id ----------


def test_trace_id_is_16_hex_and_unique():
    ids = {new_trace_id() for _ in range(50)}
    assert len(ids) == 50
    for tid in ids:
        assert len(tid) == 16
        int(tid, 16)  # 合法 hex
