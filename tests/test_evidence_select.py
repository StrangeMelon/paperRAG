"""rag.evidence_select 确定性证据选择的行为契约测试(纯函数, 无 LLM/IO)。

切片 0: 空输入契约(返回空选集 + 完整 trace 骨架)。
切片 1: 确定性与排序(分数降序、同分按原始排名、同输入恒同输出)。
切片 2: 限额(max_chunks 截断、单篇 max_per_paper、无 paper_id 的块不占额)。
切片 3: 四层打分(模型分键位优先链、词面重叠平局裁决、章节提示微加分、
        排名兜底锚)。
切片 4: 中文扩展(CJK bigram 词面重叠——基准 [a-z0-9]+ 对中文问题恒 0;
        中英混排; 中文章节提示表)。
切片 5: trace 完整性(候选逐块四项得分记账、selected 标记、输入/选中 id 列表)。

接口约定(与基准一致):

    select_evidence(question, chunks, *, intent=None, max_chunks=4,
                    max_per_paper=2) -> tuple[list[dict], dict]
"""

from __future__ import annotations

from paper_rag.rag.evidence_select import select_evidence


def _chunk(cid: str, paper: str = "p1", score: float = 0.5, **over) -> dict:
    base = {
        "chunk_id": cid,
        "paper_id": paper,
        "text": f"generic body text for {cid}",
        "section": "Body",
        "title": "Some Paper",
        "score_rerank": score,
    }
    base.update(over)
    return base


# ---------- 切片 0: 空输入契约 ----------


def test_empty_input_returns_empty_selection_and_trace():
    selected, trace = select_evidence("q", [])
    assert selected == []
    assert trace["selected_chunk_ids"] == []
    assert trace["candidates"] == []
    assert trace["strategy"] == "deterministic_score_overlap"
    assert trace["max_chunks"] == 4
    assert trace["max_per_paper"] == 2


# ---------- 切片 1: 确定性与排序 ----------


def test_sorted_by_score_desc():
    chunks = [
        _chunk("low", score=0.1),
        _chunk("high", paper="p2", score=0.9),
        _chunk("mid", paper="p3", score=0.5),
    ]
    selected, _ = select_evidence("unrelated question tokens", chunks, max_chunks=3)
    assert [c["chunk_id"] for c in selected] == ["high", "mid", "low"]


def test_tie_broken_by_original_rank():
    """同分时先来者靠前——rank_bonus 1/rank 保证稳定, 不受 sort 稳定性摆布。"""
    chunks = [
        _chunk("first", paper="p1", score=0.5),
        _chunk("second", paper="p2", score=0.5),
    ]
    selected, _ = select_evidence("unrelated", chunks, max_chunks=2)
    assert [c["chunk_id"] for c in selected] == ["first", "second"]


def test_deterministic_same_input_same_output():
    chunks = [_chunk(f"c{i}", paper=f"p{i}", score=0.5 - i / 100) for i in range(6)]
    first = select_evidence("some question", chunks)
    second = select_evidence("some question", chunks)
    assert [c["chunk_id"] for c in first[0]] == [c["chunk_id"] for c in second[0]]
    assert first[1]["candidates"] == second[1]["candidates"]


def test_input_list_not_mutated():
    chunks = [_chunk("a", score=0.1), _chunk("b", paper="p2", score=0.9)]
    snapshot = [dict(c) for c in chunks]
    select_evidence("q", chunks)
    assert chunks == snapshot, "选择器不得改写调用方的列表或块"


# ---------- 切片 2: 限额 ----------


def test_max_chunks_caps_selection():
    chunks = [_chunk(f"c{i}", paper=f"p{i}", score=1.0 - i / 10) for i in range(8)]
    selected, _ = select_evidence("q", chunks, max_chunks=3)
    assert len(selected) == 3


def test_max_per_paper_caps_single_paper():
    chunks = [
        _chunk("a1", paper="pa", score=0.9),
        _chunk("a2", paper="pa", score=0.8),
        _chunk("a3", paper="pa", score=0.7),
        _chunk("b1", paper="pb", score=0.1),
    ]
    selected, _ = select_evidence("q", chunks, max_chunks=4)
    ids = [c["chunk_id"] for c in selected]
    assert ids == ["a1", "a2", "b1"], "单篇超过 max_per_paper 的块应让位给他篇"


def test_chunks_without_paper_id_do_not_consume_quota():
    chunks = [
        _chunk("n1", paper="", score=0.9),
        _chunk("n2", paper="", score=0.8),
        _chunk("n3", paper="", score=0.7),
    ]
    selected, _ = select_evidence("q", chunks, max_chunks=3)
    assert len(selected) == 3, "无 paper_id 的块不受单篇限额约束"


# ---------- 切片 3: 四层打分 ----------


def test_model_score_key_priority_chain():
    """score_rerank > score_rrf > score_dense > score, 取第一个存在的数值键。"""
    chunks = [
        _chunk("rrf_only", paper="p1", score=None, score_rerank=None, score_rrf=0.9),
        _chunk("rerank_wins", paper="p2", score_rerank=0.3, score_rrf=99.0),
    ]
    _, trace = select_evidence("unrelated", chunks)
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["rrf_only"]["model_score"] == 0.9
    assert by_id["rerank_wins"]["model_score"] == 0.3, "有 score_rerank 时不得看 score_rrf"


def test_lexical_overlap_breaks_score_tie():
    chunks = [
        _chunk("off_topic", paper="p1", score=0.5, text="completely unrelated content"),
        _chunk("on_topic", paper="p2", score=0.5, text="self-rag uses reflection tokens"),
    ]
    selected, trace = select_evidence("what are reflection tokens", chunks, max_chunks=1)
    assert selected[0]["chunk_id"] == "on_topic"
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["on_topic"]["lexical_overlap"] > by_id["off_topic"]["lexical_overlap"]


def test_section_hint_nudges_equal_candidates():
    chunks = [
        _chunk("plain", paper="p1", score=0.5, section="Acknowledgements"),
        _chunk("hinted", paper="p2", score=0.5, section="Experiments and Results"),
    ]
    selected, trace = select_evidence("unrelated", chunks, max_chunks=1)
    assert selected[0]["chunk_id"] == "hinted"
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["hinted"]["section_hint"] == 1
    assert by_id["plain"]["section_hint"] == 0


def test_model_score_dominates_overlap():
    """模型分占大头: 0.2 权重的满额重叠(0.2)不应翻越明显的分差(0.5)。"""
    chunks = [
        _chunk("high_score", paper="p1", score=0.9, text="nothing in common"),
        _chunk("high_overlap", paper="p2", score=0.4, text="reflection tokens retrieval"),
    ]
    selected, _ = select_evidence("reflection tokens retrieval", chunks, max_chunks=1)
    assert selected[0]["chunk_id"] == "high_score"


# ---------- 切片 4: 中文扩展 ----------


def test_chinese_question_overlap_not_blind():
    """基准 [a-z0-9]+ 对中文问题抽不出 token, overlap 恒 0; CJK bigram 修复后
    中文问题必须能区分相关与无关块。"""
    chunks = [
        _chunk("zh_off", paper="p1", score=0.5, text="图神经网络在分子建模中的应用"),
        _chunk("zh_on", paper="p2", score=0.5, text="反思令牌决定何时触发检索增强"),
    ]
    selected, trace = select_evidence("反思令牌是怎么工作的", chunks, max_chunks=1)
    assert selected[0]["chunk_id"] == "zh_on"
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["zh_on"]["lexical_overlap"] > 0.0, "中文重叠不得恒为 0"
    assert by_id["zh_on"]["lexical_overlap"] > by_id["zh_off"]["lexical_overlap"]


def test_mixed_question_uses_both_scripts():
    """混排问题 "RAG 的召回率" 同时抽拉丁词与 CJK bigram, 两种块都能得到重叠。"""
    chunks = [
        _chunk("latin_hit", paper="p1", score=0.5, text="RAG recall evaluation"),
        _chunk("cjk_hit", paper="p2", score=0.5, text="模型的召回率指标对比"),
        _chunk("no_hit", paper="p3", score=0.5, text="completely unrelated 天气很好"),
    ]
    _, trace = select_evidence("RAG 的召回率是多少", chunks)
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["latin_hit"]["lexical_overlap"] > 0.0
    assert by_id["cjk_hit"]["lexical_overlap"] > 0.0
    assert by_id["no_hit"]["lexical_overlap"] < by_id["cjk_hit"]["lexical_overlap"]


def test_chinese_section_hint():
    """中文章节名(实验/结论等)也应拿到 section 加分——基准提示表全英文。"""
    chunks = [
        _chunk("zh_ack", paper="p1", score=0.5, section="致谢", title="中文论文"),
        _chunk("zh_exp", paper="p2", score=0.5, section="实验与结果", title="中文论文"),
    ]
    selected, trace = select_evidence("无关问题词", chunks, max_chunks=1)
    assert selected[0]["chunk_id"] == "zh_exp"
    by_id = {c["chunk_id"]: c for c in trace["candidates"]}
    assert by_id["zh_exp"]["section_hint"] == 1
    assert by_id["zh_ack"]["section_hint"] == 0


def test_single_cjk_char_question_no_crash():
    """单字 CJK 问题不崩溃(无除零/空集异常), 仍能按分数返回选集。

    已记账边界: 单字问题的 token {图} 与文本侧 bigram 集合({图神, 神经, ...})
    无交集, overlap 为 0——与 P6 FTS5 "bigram 索引无 unigram" 同口径, 单字查询
    本就落在词面匹配的盲区, 由稠密检索分数兜底。
    """
    selected, trace = select_evidence("图", [_chunk("c1", text="图神经网络")], max_chunks=1)
    assert selected[0]["chunk_id"] == "c1"
    assert trace["candidates"][0]["lexical_overlap"] == 0.0


# ---------- 切片 5: trace 完整性 ----------


def test_trace_candidates_full_accounting():
    chunks = [
        _chunk("a", paper="p1", score=0.9),
        _chunk("b", paper="p2", score=0.5),
        _chunk("c", paper="p3", score=0.1),
    ]
    _selected, trace = select_evidence("q", chunks, intent="factual", max_chunks=2)

    assert trace["intent"] == "factual"
    assert trace["input_chunk_ids"] == ["a", "b", "c"]
    assert trace["selected_chunk_ids"] == ["a", "b"]
    assert len(trace["candidates"]) == 3, "落选块也要记账"
    for cand in trace["candidates"]:
        for key in ("selection_score", "model_score", "lexical_overlap", "section_hint", "rank"):
            assert key in cand, f"候选记账缺 {key}"
    flags = {c["chunk_id"]: c["selected"] for c in trace["candidates"]}
    assert flags == {"a": True, "b": True, "c": False}


def test_trace_intent_defaults_to_unknown():
    _, trace = select_evidence("q", [_chunk("a")])
    assert trace["intent"] == "unknown"
