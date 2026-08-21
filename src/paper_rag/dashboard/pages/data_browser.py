"""Paper, chunk, visual asset, and wiki browser."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..services.data_service import DashboardDataService
from ..style import page_header


def render() -> None:
    import streamlit as st

    page_header("数据浏览", "Corpus & index")
    service = DashboardDataService()
    try:
        summary = service.summary()
    except Exception as exc:
        st.error(f"数据服务不可用：{exc}")
        st.exception(exc)
        return
    metrics = st.columns(4)
    metrics[0].metric("论文", summary["papers"])
    metrics[1].metric("Chunks", summary["chunks"])
    metrics[2].metric("图 / 表 / 公式", summary["visual_chunks"])
    metrics[3].metric("Wiki 概念", summary["wiki_entries"])

    upload_tab, paper_tab, wiki_tab = st.tabs(["上传入库", "论文与 Chunks", "Wiki 概念"])
    with upload_tab:
        _render_upload()
    with paper_tab:
        _render_papers(service)
    with wiki_tab:
        _render_wiki(service)


def _render_upload() -> None:
    import streamlit as st

    uploaded = st.file_uploader("选择 PDF", type=["pdf"], accept_multiple_files=False)
    title = st.text_input("论文标题（可选）")
    force = st.checkbox("强制重新入库", value=False)
    if st.button(
        "开始入库",
        icon=":material/upload_file:",
        type="primary",
        disabled=uploaded is None,
    ):
        with st.status("正在执行解析、切块、向量化与索引", expanded=True) as status:
            try:
                with tempfile.TemporaryDirectory(prefix="paper-rag-upload-") as directory:
                    pdf_path = Path(directory) / str(uploaded.name)
                    pdf_path.write_bytes(uploaded.getvalue())
                    from ...ingest.local_source import LocalSource
                    from ...store.ingest_pipeline import ingest

                    fetched = LocalSource(title=title or None).fetch(str(pdf_path))
                    result = ingest(fetched, force=force)
                status.update(label="入库完成", state="complete")
                st.success(
                    f"`{result['paper_id']}` 已入库，生成 {result.get('chunks', 0)} 个 chunks。"
                )
                st.rerun()
            except Exception as exc:
                status.update(label="入库失败", state="error")
                st.error(str(exc))
                with st.expander("诊断详情"):
                    st.exception(exc)


def _render_papers(service: DashboardDataService) -> None:
    import streamlit as st

    keyword = st.text_input("筛选论文", placeholder="标题关键词")
    papers = service.list_papers(keyword=keyword or None)
    if not papers:
        st.markdown(
            '<div class="empty-state">暂无论文，请在“上传入库”中添加 PDF。</div>',
            unsafe_allow_html=True,
        )
        return
    labels = {
        f"{item['title']}  ·  {item['chunk_count']} chunks": item["paper_id"] for item in papers
    }
    selected_label = st.selectbox("论文", list(labels))
    detail = service.get_paper_detail(labels[selected_label])
    if detail is None:
        st.warning("论文记录不存在")
        return
    paper = detail["paper"]
    cols = st.columns(4)
    cols[0].metric("Status", paper["status"])
    cols[1].metric("Chunks", paper["chunk_count"])
    cols[2].metric("Year", paper.get("year") or "-")
    cols[3].metric("Parser", paper.get("parsed_with") or "-")
    st.caption(f"paper_id `{paper['paper_id']}` · {paper.get('venue') or '未填写 venue'}")

    visual_chunks = [
        item for item in detail["chunks"] if item.get("modality") in {"figure", "table", "formula"}
    ]
    chunks_tab, visual_tab, runs_tab, danger_tab = st.tabs(
        ["全文 Chunks", "视觉资源", "入库记录", "删除"]
    )
    with chunks_tab:
        for index, chunk in enumerate(detail["chunks"], 1):
            with st.expander(
                f"{index}. {chunk.get('section') or '未命名章节'} · {chunk['modality']} · {chunk['chunk_id']}",
                expanded=index == 1,
            ):
                st.caption(f"page {chunk.get('page') or '-'}")
                st.write(chunk.get("text") or "（空内容）")
                if chunk.get("metadata"):
                    with st.expander("Metadata"):
                        st.json(chunk["metadata"])
    with visual_tab:
        if not visual_chunks:
            st.info("该论文没有图、表或公式 chunks。")
        for chunk in visual_chunks:
            asset = Path(chunk["asset_path"]) if chunk.get("asset_path") else None
            st.markdown(f"**{chunk['modality'].upper()}** · `{chunk['chunk_id']}`")
            if asset and asset.exists() and chunk["modality"] in {"figure", "table"}:
                st.image(str(asset), width=520)
            st.write(chunk.get("text") or "")
    with runs_tab:
        if detail["ingest_runs"]:
            st.dataframe(detail["ingest_runs"], use_container_width=True, hide_index=True)
        else:
            st.info("暂无入库步骤记录。")
        if st.button("重新索引", icon=":material/sync:"):
            st.info("请在上传入库页选择原 PDF 并启用“强制重新入库”。")
    with danger_tab:
        _render_delete(service, paper["paper_id"])


def _render_delete(service: DashboardDataService, paper_id: str) -> None:
    import streamlit as st

    preview = service.preview_delete(paper_id)
    if preview is None:
        return
    st.warning("删除将级联清理论文、章节、Chunks、Qdrant、FTS5 与 Wiki 关联。")
    cols = st.columns(4)
    cols[0].metric("Sections", preview["sections"])
    cols[1].metric("Chunks", preview["chunks"])
    cols[2].metric("Assets", len(preview["assets"]))
    cols[3].metric("Wiki links", preview["wiki_links"])
    if preview["assets"]:
        with st.expander("将删除的资源文件"):
            st.code("\n".join(preview["assets"]))
    delete_assets = st.checkbox("同时删除资源文件", value=True)
    confirm = st.text_input("输入 DELETE 确认", key=f"delete_confirm_{paper_id}")
    if st.button(
        "删除论文",
        icon=":material/delete_forever:",
        type="primary",
        disabled=confirm != "DELETE",
    ):
        result = service.delete_paper(paper_id, delete_assets=delete_assets)
        if result["deleted"]:
            st.success("论文及关联数据已删除。")
            if result["errors"]:
                st.warning("；".join(result["errors"]))
            st.rerun()
        else:
            st.error("；".join(result["errors"]))


def _render_wiki(service: DashboardDataService) -> None:
    import streamlit as st

    entries = service.list_wiki_entries()
    if not entries:
        st.markdown(
            '<div class="empty-state">暂无 Wiki 概念，论文入库后的异步 worker 会生成概念关系。</div>',
            unsafe_allow_html=True,
        )
        return
    for entry in entries:
        with st.expander(f"{entry.get('name')} · {entry.get('category', 'concept')}"):
            st.write(entry.get("definition") or "暂无定义")
            st.caption(f"entry_id `{entry.get('entry_id')}` · version {entry.get('version', 1)}")
            st.write("关联论文", entry.get("key_papers") or [])
            st.write("证据 chunks", entry.get("evidence_chunks") or [])
