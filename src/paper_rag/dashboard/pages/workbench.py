"""Question answering workbench."""

from __future__ import annotations

from typing import Any

from ..services.data_service import DashboardDataService
from ..services.evaluation_service import EvaluationHistory
from ..services.pipeline_monitor import PipelineMonitorStore
from ..services.query_service import QueryService
from ..style import page_header

_EXAMPLES = (
    "这篇论文的核心贡献是什么？",
    "作者使用了哪些数据集和评测指标？",
    "比较这些论文的方法、实验结果与局限。",
)


def render() -> None:
    import streamlit as st

    paths = st.session_state["dashboard_paths"]
    data_service = DashboardDataService()
    monitor_store = PipelineMonitorStore(paths.pipeline_monitor)
    eval_history = EvaluationHistory(paths.evaluation_history)
    page_header("问答工作台", "Research workspace")

    try:
        summary = data_service.summary()
        recent_ingestion = monitor_store.list(pipeline="ingestion", limit=1)
        recent_retrieval = monitor_store.list(pipeline="retrieval", limit=1)
        recent_evals = eval_history.list(limit=1)
        cols = st.columns(6)
        cols[0].metric("论文", summary["papers"])
        cols[1].metric("Chunks", summary["chunks"])
        cols[2].metric("视觉内容", summary["visual_chunks"])
        cols[3].metric("最近入库", _latest_total(recent_ingestion, "total_seconds"))
        cols[4].metric("最近检索", _latest_total(recent_retrieval, "retrieval_total_ms"))
        cols[5].metric("监控运行", len(monitor_store.list(limit=100)))
        if recent_evals:
            metrics = recent_evals[0].get("aggregate_metrics") or {}
            st.caption(
                "最近评测  " + " · ".join(f"{key} {value:.3f}" for key, value in metrics.items())
            )
    except Exception as exc:
        st.warning(f"数据服务不可用：{exc}")
        summary = {"papers": 0}

    st.divider()
    control, result_col = st.columns([0.34, 0.66], gap="large")
    with control:
        st.markdown('<div class="section-label">QUERY</div>', unsafe_allow_html=True)
        for index, example in enumerate(_EXAMPLES):
            if not summary.get("papers") and st.button(
                example, key=f"example_{index}", use_container_width=True
            ):
                st.session_state["workbench_question"] = example
        question = st.text_area(
            "问题",
            key="workbench_question",
            height=150,
            placeholder="输入关于已入库论文的问题",
        )
        papers = data_service.list_papers() if summary.get("papers") else []
        paper_lookup = {
            f"{item['title']}  ·  {item['paper_id']}": item["paper_id"] for item in papers
        }
        selected = st.multiselect("论文范围", list(paper_lookup), placeholder="全部论文")
        mode_label = st.segmented_control(
            "QA 模式",
            options=["Agentic", "Simple", "Stream"],
            default="Agentic",
            selection_mode="single",
        )
        with st.expander("高级设置"):
            top_k = st.number_input("Top K", min_value=1, max_value=30, value=8)
        run = st.button(
            "运行问答",
            icon=":material/play_arrow:",
            type="primary",
            use_container_width=True,
            disabled=not question.strip(),
        )
        if run:
            mode = str(mode_label or "Agentic").lower()
            with st.status("正在执行 RAG 链路", expanded=True) as status:
                service = QueryService()
                output = service.run(
                    question,
                    mode=mode,
                    paper_ids=[paper_lookup[item] for item in selected] or None,
                    top_k=int(top_k),
                )
                status.update(
                    label="查询完成" if output["status"] == "ok" else "查询失败",
                    state="complete" if output["status"] == "ok" else "error",
                )
            st.session_state["workbench_result"] = output

    with result_col:
        output = st.session_state.get("workbench_result")
        if not output:
            st.markdown(
                '<div class="empty-state">尚无查询结果</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_result(output)


def _render_result(output: dict[str, Any]) -> None:
    import streamlit as st

    if output.get("status") == "error":
        st.error(output.get("error", "查询失败"))
        return
    meta = st.columns(4)
    meta[0].metric("Mode", output.get("mode", "-"))
    meta[1].metric("Intent", output.get("intent", "-"))
    meta[2].metric("Abstain", output.get("abstain", "-"))
    meta[3].metric("Latency", f"{output.get('latency_ms', 0):.0f} ms")
    st.markdown('<div class="section-label">ANSWER</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer">{output.get("answer") or "-"}</div>', unsafe_allow_html=True)
    citations = output.get("citations") or []
    if citations:
        st.caption("引用  " + " · ".join(f"`{item}`" for item in citations))
    st.divider()
    evidence, trace_tab = st.tabs(["证据", "完整链路"])
    with evidence:
        chunks = output.get("evidence_chunks") or output.get("chunks") or []
        if not chunks:
            st.info("没有可展示的证据")
        for index, chunk in enumerate(chunks, 1):
            title = (
                f"{index}. {chunk.get('section') or '未命名章节'} · {chunk.get('chunk_id', '-')}"
            )
            with st.expander(title, expanded=index == 1):
                st.caption(
                    f"paper_id `{chunk.get('paper_id', '-')}` · modality `{chunk.get('modality', 'text')}`"
                )
                text = str(chunk.get("text") or chunk.get("context_text") or "")
                st.write(text[:500] + ("..." if len(text) > 500 else ""))
    with trace_tab:
        _render_trace(output.get("trace") or {})


def _render_trace(trace: dict[str, Any]) -> None:
    import streamlit as st

    iterations = trace.get("iters") or trace.get("iterations") or []
    intent = trace.get("intent") or {}
    st.markdown(
        f'<div class="trace-step"><strong>Intent</strong><br>{intent}</div>',
        unsafe_allow_html=True,
    )
    for index, item in enumerate(iterations, 1):
        timing = format_iteration_timing(item)
        st.markdown(
            f'<div class="trace-step"><strong>Retrieval {index}</strong><br>'
            f"{item.get('query', '-')} · {item.get('n_retrieved', 0)} chunks<br>"
            f"{timing}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="trace-step"><strong>Abstain</strong><br>{trace.get("abstain") or {}}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="trace-step"><strong>Generation</strong><br>{trace.get("stopped_by", "done")}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Raw trace"):
        st.json(trace)


def _latest_total(records: list[dict], key: str) -> str:
    if not records:
        return "-"
    value = (records[0].get("timings_ms") or {}).get(key)
    return f"{float(value):,.0f} ms" if value is not None else "-"


def format_iteration_timing(iteration: dict[str, Any]) -> str:
    """Format new timing fields while remaining readable for historical traces."""
    retrieval = iteration.get("retrieval_latency_ms")
    if retrieval is None:
        return "检索耗时 -"
    parts = [f"检索 {float(retrieval):,.0f} ms"]
    reflection = iteration.get("reflect_latency_ms")
    if reflection is not None:
        parts.append(f"反思 {float(reflection):,.0f} ms")
    total = iteration.get("iteration_latency_ms")
    if total is not None:
        parts.append(f"本轮 {float(total):,.0f} ms")
    return " · ".join(parts)
