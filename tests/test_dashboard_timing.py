"""Dashboard timing presentation contracts."""

from __future__ import annotations


def test_format_iteration_timing_shows_retrieval_reflect_and_total() -> None:
    from paper_rag.dashboard.pages.workbench import format_iteration_timing

    assert (
        format_iteration_timing(
            {
                "retrieval_latency_ms": 1234.56,
                "reflect_latency_ms": 456.7,
                "iteration_latency_ms": 1700.2,
            }
        )
        == "检索 1,235 ms · 反思 457 ms · 本轮 1,700 ms"
    )


def test_format_iteration_timing_supports_historical_traces() -> None:
    from paper_rag.dashboard.pages.workbench import format_iteration_timing

    assert format_iteration_timing({}) == "检索耗时 -"
