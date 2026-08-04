"""format_evidence 证据渲染的行为契约测试(纯函数)。

引用格式硬不变量的物理源头: 每块证据头部逐字携带
"Use this exact citation token when citing this chunk: [chunk:<id>]",
下游 LLM 据此产出 [chunk:<id>] 引用, citation_check 再按同格式校验闭环。
信封保持英文(LLM 协议文本, 非用户可见; 中文正文原样在 body, P6 收尾课确认)。
"""

from __future__ import annotations

from paper_rag.retrieve.format import format_evidence


def _chunk(**kw) -> dict:
    base = {
        "chunk_id": "abc123",
        "paper_id": "p1",
        "section": "Introduction",
        "modality": "text",
        "score": 0.815,
        "text": "Graph-Mamba improves long-range modeling.",
    }
    base.update(kw)
    return base


def test_citation_token_line_verbatim():
    out = format_evidence([_chunk()])
    assert "Use this exact citation token when citing this chunk: [chunk:abc123]" in out
    assert out.startswith("EVIDENCE CHUNK\n")


def test_header_fields_and_body():
    out = format_evidence([_chunk()])
    assert "paper_id=p1 section=Introduction modality=text score=0.815" in out
    assert out.endswith("Graph-Mamba improves long-range modeling.")


def test_multiple_chunks_joined_by_separator():
    out = format_evidence([_chunk(), _chunk(chunk_id="def456")])
    assert out.count("EVIDENCE CHUNK") == 2
    assert out.count("\n\n---\n\n") == 1
    assert "[chunk:def456]" in out


def test_empty_list_renders_empty_string():
    assert format_evidence([]) == ""


def test_missing_fields_tolerated():
    out = format_evidence([{"chunk_id": "x1"}])
    assert "[chunk:x1]" in out
    assert "score=0.000" in out  # score 缺失按 0 渲染


def test_chinese_body_kept_verbatim():
    out = format_evidence([_chunk(text="  区块链技术支撑综合能源服务。\n")])
    assert out.endswith("区块链技术支撑综合能源服务。")  # strip 后原样
