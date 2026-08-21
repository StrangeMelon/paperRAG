"""QA and pure retrieval evaluation panel."""

from __future__ import annotations

from pathlib import Path

from ..services.evaluation_service import EvaluationHistory, EvaluationService
from ..style import page_header


def _default_test_set(retrieval_mode: bool, backend: str) -> str:
    if retrieval_mode:
        return "tests/fixtures/evaluation/retrieval_golden.json"
    if backend == "ragas":
        return "tests/fixtures/evaluation/ragas_golden.json"
    return "tests/fixtures/evaluation/golden.json"


def render() -> None:
    import streamlit as st

    page_header("评测面板", "Quality regression")
    paths = st.session_state["dashboard_paths"]
    history = EvaluationHistory(paths.evaluation_history)
    service = EvaluationService(history)
    controls, results = st.columns([0.34, 0.66], gap="large")
    with controls:
        mode_label = st.segmented_control(
            "评测模式", ["QA", "纯检索"], default="QA", selection_mode="single"
        )
        retrieval_mode = mode_label == "纯检索"
        backend_label = st.segmented_control(
            "评测后端",
            ["Custom"] if retrieval_mode else ["Custom", "RAGAS", "Composite"],
            default="Custom",
            selection_mode="single",
        )
        backend = str(backend_label or "Custom").lower()
        test_set_mode = "retrieval" if retrieval_mode else backend
        if st.session_state.get("evaluation_test_set_mode") != test_set_mode:
            st.session_state["evaluation_test_set"] = _default_test_set(retrieval_mode, backend)
            st.session_state["evaluation_test_set_mode"] = test_set_mode
        test_set = st.text_input(
            "Golden Set",
            key="evaluation_test_set",
        )
        top_k = st.number_input("Top-K", min_value=1, max_value=100, value=8, step=1)
        query_rewrite = st.toggle(
            "Query Rewrite",
            value=True,
            disabled=not (retrieval_mode or backend == "ragas"),
        )
        ragas_max_concurrency = (
            st.number_input(
                "RAGAS 并发数",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
            )
            if not retrieval_mode and backend == "ragas"
            else None
        )
        if retrieval_mode:
            custom_metrics = st.multiselect(
                "检索指标",
                ["hit_rate", "mrr", "recall", "paper_hit_rate"],
                default=["hit_rate", "mrr", "recall", "paper_hit_rate"],
            )
            ragas_metrics = []
        elif backend == "ragas":
            custom_metrics = []
            ragas_metrics = st.multiselect(
                "RAGAS metrics",
                [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "answer_correctness",
                ],
                default=["faithfulness", "answer_relevancy", "context_precision"],
            )
        else:
            custom_metrics = st.multiselect(
                "Custom metrics",
                [
                    "hit_rate",
                    "mrr",
                    "recall",
                    "paper_hit_rate",
                    "citation_precision",
                    "citation_recall",
                    "abstain_accuracy",
                ],
                default=["hit_rate", "mrr", "recall"],
            )
            ragas_metrics = st.multiselect(
                "RAGAS metrics",
                [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "answer_correctness",
                ],
                default=["faithfulness", "answer_relevancy", "context_precision"],
            )
        path_exists = Path(test_set).exists()
        if not path_exists:
            st.warning("Golden Set 不存在。")
        run = st.button(
            "运行评测",
            icon=":material/play_arrow:",
            type="primary",
            use_container_width=True,
            disabled=not path_exists,
        )
        if run:
            with st.status("正在运行 Golden Set", expanded=True) as status:
                try:
                    record = service.run(
                        test_set=test_set,
                        backend=backend,
                        custom_metrics=custom_metrics,
                        ragas_metrics=ragas_metrics,
                        mode="retrieval" if retrieval_mode else "qa",
                        top_k=int(top_k),
                        query_rewrite=bool(query_rewrite),
                        max_concurrency=(
                            int(ragas_max_concurrency)
                            if ragas_max_concurrency is not None
                            else None
                        ),
                    )
                    status.update(label="评测完成", state="complete")
                    st.session_state["evaluation_result"] = record
                except Exception as exc:
                    status.update(label="评测失败", state="error")
                    st.error(str(exc))
                    with st.expander("诊断详情"):
                        st.exception(exc)

        runs = history.list(limit=20)
        st.markdown("#### 历史运行")
        if not runs:
            st.caption("尚未运行评测")
        else:
            run_labels = {
                f"{item['created_at'][:16]} · {item.get('evaluation', {}).get('mode', 'qa')} · "
                f"{item['backend']} · {item['run_id']}": item
                for item in runs
            }
            selected_history = st.selectbox(
                "历史记录", list(run_labels), label_visibility="collapsed"
            )
            if st.button("查看历史结果", use_container_width=True):
                st.session_state["evaluation_result"] = run_labels[selected_history]

    with results:
        record = st.session_state.get("evaluation_result")
        if not record:
            st.markdown(
                '<div class="empty-state">尚未运行评测。选择后端和 Golden Set 后开始。</div>',
                unsafe_allow_html=True,
            )
            return
        _render_record(record)


def _render_record(record: dict) -> None:
    import streamlit as st

    evaluation = record.get("evaluation") or {}
    if evaluation.get("mode") == "retrieval":
        st.info(
            "评测模式：纯检索\n\n"
            f"Top-K：{evaluation.get('top_k', '-')}\n\n"
            f"Query Rewrite：{'开启' if evaluation.get('query_rewrite') else '关闭'}\n\n"
            f"问题并发数：{evaluation.get('max_concurrency', '-')}\n\n"
            f"语料选择：{evaluation.get('corpus_selection', '-')}\n\n"
            f"实际论文数：{evaluation.get('corpus_paper_count', '-')}\n\n"
            f"Golden Set 题数：{evaluation.get('golden_case_count', '-')}\n\n"
            f"有效题数：{evaluation.get('valid_case_count', '-')}"
        )
    elif evaluation.get("mode") == "ragas":
        st.info(
            "评测模式：RAGAS\n\n"
            f"RAGAS：{evaluation.get('ragas_version', '-')}\n\n"
            f"Judge：{evaluation.get('judge_model', '-')}\n\n"
            f"Embedding：{evaluation.get('embedding_model', '-')}\n\n"
            f"Top-K：{evaluation.get('top_k', '-')}\n\n"
            f"Query Rewrite：{'开启' if evaluation.get('query_rewrite') else '关闭'}\n\n"
            f"QA/评分并发数：{evaluation.get('max_concurrency', '-')}\n\n"
            f"语料论文数：{evaluation.get('corpus_paper_count', '-')}"
        )
    st.caption(
        f"run_id `{record.get('run_id', '-')}` · backend `{record.get('backend', record.get('evaluator', '-'))}`"
    )
    metrics = record.get("aggregate_metrics") or {}
    if metrics:
        columns = st.columns(min(len(metrics), 4))
        for index, (name, value) in enumerate(metrics.items()):
            if evaluation.get("mode") == "ragas" and isinstance(value, dict):
                columns[index % len(columns)].metric(
                    name,
                    f"{float(value.get('mean', 0.0)):.3f}",
                    f"coverage {float(value.get('coverage', 0.0)):.0%}",
                    delta_color="off",
                )
            else:
                columns[index % len(columns)].metric(name, f"{float(value):.3f}")
    else:
        st.warning("本次运行没有产生指标。")
    rows = record.get("query_results") or []
    st.markdown("#### 逐题结果")
    if not rows:
        st.info("没有逐题结果。")
        return
    if evaluation.get("mode") == "retrieval":
        for item in rows:
            label = f"{item.get('id', '-')} · {item.get('query', '')}"
            with st.expander(label):
                st.write(
                    {
                        "search_scope": item.get("search_scope"),
                        "expected_chunk_ids": item.get("expected_chunk_ids", []),
                        "expected_paper_ids": item.get("expected_paper_ids", []),
                        "reference_answer": item.get("reference_answer", ""),
                        "retrieved_chunk_ids": item.get("retrieved_chunk_ids", []),
                        "retrieved_paper_ids": item.get("retrieved_paper_ids", []),
                        "metrics": item.get("metrics", {}),
                        "latency_ms": item.get("latency_ms"),
                        "status": item.get("status"),
                    }
                )
    else:
        if evaluation.get("mode") == "ragas":
            summary_rows = [
                {
                    "id": item.get("id"),
                    "query": item.get("query"),
                    "status": item.get("status"),
                    "qa_latency_ms": item.get("qa_latency_ms"),
                    "ragas_latency_ms": item.get("ragas_latency_ms"),
                    **(item.get("metrics") or {}),
                }
                for item in rows
            ]
        else:
            summary_rows = [
                {
                    "id": item.get("id"),
                    "query": item.get("query"),
                    "status": item.get("status"),
                    "latency_ms": item.get("latency_ms"),
                    **(item.get("metrics") or {}),
                }
                for item in rows
            ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    failures = [
        item
        for item in rows
        if item.get("status") in {"error", "partial", "qa_error"} or item.get("errors")
    ]
    if failures:
        with st.expander(f"失败样本 ({len(failures)})", expanded=True):
            for item in failures:
                st.error(f"{item.get('query')} · {item.get('errors')}")
    with st.expander("完整报告"):
        st.json(record)
