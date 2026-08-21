"""Ingestion and retrieval pipeline monitoring page."""

from __future__ import annotations

from ..services.pipeline_monitor import PipelineMonitorStore
from ..style import page_header


def render() -> None:
    import streamlit as st

    page_header("管道监控", "Pipeline observability")
    store = PipelineMonitorStore(st.session_state["dashboard_paths"].pipeline_monitor)
    ingestion, retrieval = st.tabs(["入库监控", "检索监控"])
    with ingestion:
        _render_pipeline_runs(store, "ingestion")
    with retrieval:
        _render_pipeline_runs(store, "retrieval")


def _render_pipeline_runs(store: PipelineMonitorStore, pipeline: str) -> None:
    import streamlit as st

    if pipeline == "retrieval":
        _render_retrieval_diagnostic()
        st.divider()
        _render_retrieval_history()
        return
    records = store.list(pipeline=pipeline, limit=100)
    if not records:
        label = "入库" if pipeline == "ingestion" else "检索"
        st.markdown(
            f'<div class="empty-state">暂无{label}监控记录。执行一次{label}流程后会在这里显示各模块耗时。</div>',
            unsafe_allow_html=True,
        )
        return
    labels = {_run_label(record, pipeline): record["run_id"] for record in records}
    selected_label = st.selectbox("运行记录", list(labels), key=f"monitor_run_{pipeline}")
    selected = store.get(labels[selected_label])
    if selected is None:
        return
    _render_summary(selected, pipeline)
    timings = selected.get("timings_ms") or {}
    if timings:
        st.markdown("#### 模块耗时")
        leaf_timings = {
            name: value for name, value in timings.items() if name not in _AGGREGATE_TIMINGS
        }
        rows = [
            {
                "模块": _module_label(name),
                "耗时 (ms)": round(float(value), 1),
                "占比": _share(value, leaf_timings),
            }
            for name, value in leaf_timings.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        chart_rows = [
            {"module": _module_label(name), "ms": float(value)}
            for name, value in leaf_timings.items()
        ]
        st.bar_chart(chart_rows, x="module", y="ms", horizontal=True)
    if pipeline == "retrieval":
        iterations = (selected.get("metadata") or {}).get("iterations") or []
        if iterations:
            st.markdown("#### 每轮检索")
            rows = []
            for index, iteration in enumerate(iterations, 1):
                row = {
                    "轮次": index,
                    "查询": iteration.get("query", ""),
                    "召回数": iteration.get("n_retrieved", 0),
                    "本轮 (ms)": iteration.get("iteration_latency_ms", "-"),
                }
                row.update(
                    {
                        _module_label(name): value
                        for name, value in (iteration.get("module_timings_ms") or {}).items()
                    }
                )
                rows.append(row)
            st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("运行元数据"):
        st.json(selected.get("metadata") or {})
    if st.button("删除该监控记录", key=f"delete_monitor_{pipeline}_{selected['run_id']}"):
        store.delete(selected["run_id"])
        st.rerun()


def _render_retrieval_diagnostic() -> None:
    import streamlit as st

    st.markdown("#### 当前检索诊断")
    st.caption("输入查询后查看检索链路中的中间结果、排序、分数和耗时。")
    query = st.text_area(
        "查询问题",
        key="retrieval_diagnostic_query",
        height=90,
        placeholder="例如：RAG 的检索阶段如何评估？",
    )
    try:
        from ..services.data_service import DashboardDataService

        papers = DashboardDataService().list_papers()
    except Exception:
        papers = []
    paper_lookup = {f"{item['title']}  ·  {item['paper_id']}": item["paper_id"] for item in papers}
    selected = st.multiselect("论文范围（可选）", list(paper_lookup), key="diagnostic_papers")
    options_col, button_col = st.columns([0.25, 0.75])
    with options_col:
        top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            key="diagnostic_top_k",
        )
    with button_col:
        st.write("")
        run = st.button(
            "运行检索诊断",
            icon=":material/search:",
            type="primary",
            disabled=not query.strip(),
            key="run_retrieval_diagnostic",
        )
    if run:
        with st.status(
            "正在执行 Query Rewrite、Dense、Sparse、RRF、Rerank、Diversify", expanded=True
        ):
            try:
                from ..services.retrieval_diagnostic import run_retrieval_diagnostic

                result = run_retrieval_diagnostic(
                    query,
                    paper_ids=[paper_lookup[item] for item in selected] or None,
                    top_k=int(top_k),
                )
                from ..services.retrieval_history import RetrievalHistoryStore

                result["run_id"] = RetrievalHistoryStore().save(
                    result,
                    paper_ids=[paper_lookup[item] for item in selected] or None,
                    top_k=int(top_k),
                )
                st.session_state["retrieval_diagnostic_result"] = result
            except Exception as exc:
                st.error(f"检索诊断失败：{exc}")
    result = st.session_state.get("retrieval_diagnostic_result")
    if result:
        _render_diagnostic_result(result)


def _render_retrieval_history() -> None:
    import streamlit as st

    from ..services.retrieval_history import RetrievalHistoryStore

    st.markdown("#### 历史检索运行")
    store = RetrievalHistoryStore()
    filter_col, page_col = st.columns([0.72, 0.28])
    with filter_col:
        keyword = st.text_input(
            "搜索历史查询",
            key="retrieval_history_keyword",
            placeholder="输入查询关键词",
        )
    page_size = 10
    total = store.count(query=keyword or None)
    max_page = max(1, (total + page_size - 1) // page_size)
    with page_col:
        page = int(
            st.number_input(
                "页码",
                min_value=1,
                max_value=max_page,
                value=min(int(st.session_state.get("retrieval_history_page", 1)), max_page),
                step=1,
                key="retrieval_history_page",
            )
        )
    records = store.list(
        query=keyword or None,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    st.caption(f"共 {total} 条记录 · 第 {page}/{max_page} 页")
    if not records:
        st.markdown(
            '<div class="empty-state">暂无检索诊断历史。运行一次检索诊断后会自动保存。</div>',
            unsafe_allow_html=True,
        )
        return
    labels = {
        f"{item['query'][:55]} · {item['total_ms']:.1f} ms · {item['created_at'][:16]}": item[
            "run_id"
        ]
        for item in records
    }
    selected_label = st.selectbox(
        "历史运行",
        list(labels),
        key="retrieval_history_selected",
    )
    selected = store.get(labels[selected_label])
    if selected is None:
        return
    action_col, delete_col = st.columns([0.8, 0.2])
    with action_col:
        st.caption(
            f"Top K `{selected['top_k']}` · 论文范围 "
            f"`{', '.join(selected['paper_ids']) if selected['paper_ids'] else '全部'}`"
        )
    with delete_col:
        if st.button(
            "删除记录",
            icon=":material/delete:",
            key=f"delete_diagnostic_{selected['run_id']}",
            use_container_width=True,
        ):
            store.delete(selected["run_id"])
            st.rerun()
    _render_diagnostic_result(selected)


def _render_diagnostic_result(result: dict) -> None:
    import streamlit as st

    timings = result.get("timings_ms") or {}
    st.markdown(f"**查询：** {result.get('query', '-')}")
    metric_cols = st.columns(7)
    for column, name in zip(
        metric_cols,
        (
            "query_rewrite_ms",
            "dense_ms",
            "sparse_ms",
            "rrf_ms",
            "rerank_ms",
            "diversify_ms",
            "retrieval_total_ms",
        ),
        strict=False,
    ):
        column.metric(_module_label(name), f"{float(timings.get(name, 0.0)):.1f} ms")
    stages = result.get("stages") or {}
    for name in ("Query Rewrite", "Dense", "Sparse", "RRF", "Rerank", "Diversify"):
        stage = stages.get(name) or {}
        with st.expander(
            f"{name} · {float(stage.get('timing_ms', 0.0)):.1f} ms",
            expanded=name == "Query Rewrite",
        ):
            if name == "Query Rewrite":
                queries = stage.get("rewritten_queries") or []
                st.markdown("**重写后的查询文本**")
                for index, item in enumerate(queries, 1):
                    st.write(f"{index}. {item}")
                st.markdown(f"**BM25 查询：** {stage.get('bm25_query') or '-'}")
            else:
                _render_ranked_items(stage.get("items") or [], name)


def _render_ranked_items(items: list[dict], stage_name: str) -> None:
    import streamlit as st

    if not items:
        st.info("该阶段没有召回块")
        return
    score_key = {
        "Dense": "score",
        "Sparse": "score_bm25",
        "RRF": "score_rrf",
        "Rerank": "score_rerank",
        "Diversify": "score_rerank",
    }.get(stage_name)
    rows = []
    for rank, item in enumerate(items, 1):
        score = item.get(score_key) if score_key else None
        rows.append(
            {
                "排名": rank,
                "chunk_id": item.get("chunk_id", "-"),
                "paper_id": item.get("paper_id", "-"),
                "分数": _format_score(score),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for rank, item in enumerate(items, 1):
        chunk_id = item.get("chunk_id", "-")
        with st.expander(f"#{rank} · {chunk_id}"):
            st.caption(
                f"paper_id `{item.get('paper_id', '-')}` · "
                f"{score_key or 'score'} `{_format_score(item.get(score_key))}`"
            )
            text = str(item.get("text") or item.get("context_text") or "")
            st.write(text[:800] + ("..." if len(text) > 800 else ""))


def _format_score(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "-"


def _render_summary(record: dict, pipeline: str) -> None:
    import streamlit as st

    timings = record.get("timings_ms") or {}
    total = _total_ms(timings, pipeline)
    cols = st.columns(4)
    cols[0].metric("Pipeline", "Ingestion" if pipeline == "ingestion" else "Retrieval")
    cols[1].metric("Total", f"{total:,.0f} ms")
    cols[2].metric("Modules", len(timings))
    cols[3].metric("Status", record.get("status", "-"))
    if pipeline == "ingestion":
        st.caption(f"paper_id `{record.get('paper_id', '-')}` · {record.get('created_at', '')}")
    else:
        st.caption(f"query `{record.get('query', '-')}` · {record.get('created_at', '')}")


def _run_label(record: dict, pipeline: str) -> str:
    subject = record.get("paper_id") if pipeline == "ingestion" else record.get("query")
    return f"{str(subject or '-')[:55]} · {record.get('created_at', '')[:16]}"


def _total_ms(timings: dict, pipeline: str) -> float:
    if pipeline == "ingestion" and timings.get("total_seconds") is not None:
        return float(timings["total_seconds"])
    if pipeline == "retrieval" and timings.get("retrieval_total_ms") is not None:
        return float(timings["retrieval_total_ms"])
    return sum(float(value) for value in timings.values())


def _share(value: float, timings: dict) -> str:
    total = sum(float(item) for item in timings.values())
    return f"{float(value) / total:.1%}" if total else "-"


def _module_label(name: str) -> str:
    labels = {
        "parse_seconds": "Parse",
        "chunk_seconds": "Chunk",
        "vision_seconds": "Vision",
        "sqlite_seconds": "SQLite write",
        "qdrant_snapshot_seconds": "Qdrant snapshot",
        "incremental_plan_seconds": "Incremental plan",
        "embedding_seconds": "Embedding",
        "index_seconds": "Index + FTS5",
        "incremental_update_seconds": "Incremental plan",
        "qdrant_write_seconds": "Qdrant write",
        "fts5_sync_seconds": "FTS5 sync",
        "wiki_enqueue_seconds": "Wiki enqueue",
        "query_rewrite_ms": "Query Rewrite",
        "dense_ms": "Dense Retrieval",
        "sparse_ms": "Sparse Retrieval",
        "rrf_ms": "RRF Fusion",
        "hybrid_ms": "Hybrid Search",
        "rerank_ms": "Rerank",
        "diversify_ms": "Diversify",
        "retrieval_total_ms": "Retrieval total",
        "total_seconds": "Ingestion total",
    }
    return labels.get(name, name)


_AGGREGATE_TIMINGS = {
    "total_seconds",
    "incremental_update_seconds",
    "index_seconds",
    "hybrid_ms",
    "retrieval_total_ms",
}
