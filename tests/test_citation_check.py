"""rag.citation_check 引用校验的行为契约测试(纯函数, 无 LLM/IO)。

硬不变量"答案只以 [chunk:<id>] 引用已检索块"的执行层:

切片 0: validate_citations(白名单校验、编造 id 删除、valid 保序去重、
        空检索集、缺 chunk_id 的块防御)。
切片 1: 数字可疑引用(半角 [1] 检测与报告格式; [chunk:...]/markdown 勾选框
        [x]/四位数 [1234] 不误报)。
切片 2: 拉丁作者-年份((Vaswani et al., 2017)/(Smith 2020)/带音标名/年份
        后缀 2020a; (see Figure 2)/裸 (2020) 不误报)。
切片 3: 中文扩展(全角【1】; CJK 作者-年份 (张三等, 2020) 半/全角括号逗号;
        归入既有 numeric/author_year 键, 消费方 schema 不变)。
切片 4: strip_suspicious_citation_forms(剥两类形态、ASCII 与全角标点前空格
        收拾、干净文本恒等、[chunk:] 不受伤)。
切片 5: 消费方调用顺序集成(validate -> detect -> strip 三段管道)。

接口约定(与基准一致):

    validate_citations(answer, retrieved) -> (cleaned, valid_ids)
    detect_suspicious_citations(answer) -> {"numeric": [...],
                                            "author_year": [...], "count": int}
    strip_suspicious_citation_forms(answer) -> str
"""

from __future__ import annotations

from paper_rag.rag.citation_check import (
    detect_suspicious_citations,
    strip_suspicious_citation_forms,
    validate_citations,
)

_ID_A = "a3f09b2c17d4e8f0a1b2"  # 与重建版 chunk_id 同形态: sha1 hex 前 20 位
_ID_B = "b4e18c3d28e5f9a0b2c3"


def _retrieved(*ids: str) -> list[dict]:
    return [{"chunk_id": cid, "text": "..."} for cid in ids]


# ---------- 切片 0: validate_citations ----------


def test_valid_citation_kept_and_collected():
    answer = f"Graph-Mamba uses selective state spaces [chunk:{_ID_A}]."
    cleaned, valid = validate_citations(answer, _retrieved(_ID_A))
    assert cleaned == answer
    assert valid == [_ID_A]


def test_fabricated_citation_removed_from_text_and_list():
    answer = f"True claim [chunk:{_ID_A}]. Fabricated claim [chunk:deadbeefdeadbeefdead]."
    cleaned, valid = validate_citations(answer, _retrieved(_ID_A))
    assert valid == [_ID_A]
    assert "deadbeef" not in cleaned
    assert f"[chunk:{_ID_A}]" in cleaned


def test_valid_ids_order_preserving_dedup():
    answer = f"A [chunk:{_ID_B}] B [chunk:{_ID_A}] C [chunk:{_ID_B}]."
    _, valid = validate_citations(answer, _retrieved(_ID_A, _ID_B))
    assert valid == [_ID_B, _ID_A]


def test_empty_retrieved_drops_everything():
    cleaned, valid = validate_citations(f"claim [chunk:{_ID_A}]", [])
    assert valid == []
    assert "chunk:" not in cleaned


def test_chunks_without_chunk_id_are_ignored():
    cleaned, valid = validate_citations(f"claim [chunk:{_ID_A}]", [{"text": "no id"}])
    assert valid == []
    assert "chunk:" not in cleaned


def test_no_citations_at_all_is_fine():
    cleaned, valid = validate_citations("no citations here.", _retrieved(_ID_A))
    assert cleaned == "no citations here."
    assert valid == []


# ---------- 切片 1: 数字可疑引用(半角) ----------


def test_numeric_citations_detected_with_bracket_report_format():
    report = detect_suspicious_citations("As shown in [1] and [12], ...")
    assert report["numeric"] == ["[1]", "[12]"]
    assert report["count"] == 2


def test_chunk_citations_not_flagged_as_numeric():
    report = detect_suspicious_citations(f"claim [chunk:{_ID_A}].")
    assert report["count"] == 0


def test_markdown_checkbox_and_long_numbers_not_flagged():
    report = detect_suspicious_citations("- [x] done; see [1234] (an id, not a citation)")
    assert report["numeric"] == []


# ---------- 切片 2: 拉丁作者-年份 ----------


def test_author_year_variants_detected():
    text = "Attention (Vaswani et al., 2017) and BERT (Devlin 2019a) and (Müller, 2020)."
    report = detect_suspicious_citations(text)
    assert "(Vaswani et al., 2017)" in report["author_year"]
    assert "(Devlin 2019a)" in report["author_year"]
    assert "(Müller, 2020)" in report["author_year"]
    assert report["count"] == 3


def test_ordinary_parentheticals_not_flagged():
    report = detect_suspicious_citations("see (see Figure 2) and a bare year (2020).")
    assert report["author_year"] == []
    assert report["count"] == 0


# ---------- 切片 3: 中文扩展 ----------


def test_fullwidth_numeric_citation_detected():
    report = detect_suspicious_citations("如文献【3】所示, 模型收敛更快。")
    assert report["numeric"] == ["【3】"]
    assert report["count"] == 1


def test_cjk_author_year_fullwidth_detected():
    report = detect_suspicious_citations("该方法最早由（张三等，2020）提出。")
    assert report["author_year"] == ["（张三等，2020）"]


def test_cjk_author_year_halfwidth_detected():
    report = detect_suspicious_citations("对比实验见 (李四 等, 2019)。")
    assert report["author_year"] == ["(李四 等, 2019)"]


def test_mixed_forms_counted_together():
    text = "结论见【2】与 (Vaswani et al., 2017), 另见（王五，2021）。"
    report = detect_suspicious_citations(text)
    assert report["count"] == 3


def test_cjk_plain_parenthetical_not_flagged():
    report = detect_suspicious_citations("(见图 2) 与（详见附录）不是引用。")
    assert report["author_year"] == []
    assert report["count"] == 0


# ---------- 切片 4: strip_suspicious_citation_forms ----------


def test_strip_removes_numeric_and_author_year():
    text = "Claim A [1]. Claim B (Vaswani et al., 2017)."
    cleaned = strip_suspicious_citation_forms(text)
    assert "[1]" not in cleaned
    assert "Vaswani" not in cleaned


def test_strip_tidies_ascii_punctuation_gaps():
    cleaned = strip_suspicious_citation_forms("Claim A [1] , then B (Smith 2020) .")
    assert cleaned == "Claim A, then B."


def test_strip_tidies_fullwidth_punctuation_gaps():
    cleaned = strip_suspicious_citation_forms(
        "如【3】所述, 收敛更快 【4】，且更稳 （张三等，2020）。"
    )
    assert "【" not in cleaned
    assert " ，" not in cleaned
    assert " 。" not in cleaned


def test_strip_is_noop_on_clean_text():
    text = f"Answer with only valid cites [chunk:{_ID_A}]."
    assert strip_suspicious_citation_forms(text) == text


def test_strip_never_touches_chunk_citations():
    text = f"A [chunk:{_ID_A}] and bad [2]."
    cleaned = strip_suspicious_citation_forms(text)
    assert f"[chunk:{_ID_A}]" in cleaned
    assert "[2]" not in cleaned


# ---------- 切片 5: 消费方三段管道 ----------


def test_pipeline_validate_then_detect_then_strip():
    """qa_simple/qa_agentic/qa_stream 的固定调用顺序在混合污染答案上端到端成立。"""
    answer = (
        f"Real claim [chunk:{_ID_A}]. Fabricated [chunk:ffffffffffffffffffff]. "
        "Habit citation [3], 中文惯性【5】, and (Vaswani et al., 2017)、（张三等，2020）。"
    )
    cleaned, valid = validate_citations(answer, _retrieved(_ID_A))
    report = detect_suspicious_citations(cleaned)
    assert valid == [_ID_A]
    assert report["count"] == 4
    final = strip_suspicious_citation_forms(cleaned)
    assert f"[chunk:{_ID_A}]" in final
    for bad in ("ffffffff", "[3]", "【5】", "Vaswani", "张三"):
        assert bad not in final
