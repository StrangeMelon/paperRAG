"""单轮 RAG: dense 检索 -> 组 prompt -> LLM 作答 -> 引用校验。

基准 phase-1 产物但现役: scripts/ask.py 的 CLI 入口、评测 ablation 的
dense-only 最小基线、citation_check 三段管道的最简单完整消费方。agentic
变体(rewrite + reflect + 迭代 + abstain)在 qa_agentic.py。

_SYSTEM 是硬不变量的 prompt 端(只用证据、逐句 [chunk:<id>]、禁数字/作者-
年份引用、不足要明说、不编造), citation_check 是执行端——prompt 端让模型
少犯, 执行端保证犯了也漏不出去。

相对基准的确认偏离:
a) 系统与用户模板 zh/en 双语路由(复用 _query_language; 中文版明确禁止全角
   引用形态 【1】/全角括号作者-年份, 与 citation_check 的中文扩展同步收紧);
b) 无证据短路文案按语言路由("(未检索到证据)" / 基准英文原文)。
照抄并记账: dense-only 检索不升级为 retrieve_round(设计决策——qa_simple 的
存在意义就是最小对照组, ablation 评测需要该基线; 完整管道是 qa_agentic 的
事); chat() 全默认参数; 输出四键 schema 无 trace。
"""

from __future__ import annotations

from .. import config as cfg
from ..retrieve.dense import retrieve
from ..retrieve.format import format_evidence
from ..retrieve.reference_policy import detect_reference_intent, filter_answer_evidence
from ..utils.logger import get_logger
from .citation_check import (
    detect_suspicious_citations,
    strip_suspicious_citation_forms,
    validate_citations,
)
from .llm import chat
from .query_rewrite import _query_language

log = get_logger("rag.qa_simple")

_SYSTEM = (
    "You are a careful academic research assistant. Answer ONLY using the "
    "evidence chunks provided. After each factual statement, cite the chunk "
    "with the format [chunk:<chunk_id>]. NEVER use [1], [2], or "
    "(Author 2020) style citations — they will be considered hallucinated. "
    "If the evidence is insufficient, say so explicitly. Do NOT fabricate "
    "paper titles or numbers."
)

_SYSTEM_ZH = (
    "你是一名严谨的学术研究助手。只能使用下面提供的证据块作答。每个事实陈述"
    "之后必须用 [chunk:<chunk_id>] 格式引用对应证据块。绝对不要使用 [1]、"
    "【1】、(作者 2020)、（张三等，2020）之类的数字或作者-年份引用——它们会被"  # noqa: RUF001
    "判为幻觉引用。证据不足以回答时必须明确说明。不要编造论文标题或数字。"
)

_NO_EVIDENCE = "(no evidence found)"
_NO_EVIDENCE_ZH = "(未检索到证据)"


def answer(question: str, *, top_k: int = 8, paper_ids: list[str] | None = None) -> dict:
    lang = _query_language(question)
    reference_cfg = cfg.load().retrieve.references
    raw_chunks = retrieve(question, top_k=top_k * 3, paper_ids=paper_ids)
    chunks = filter_answer_evidence(
        raw_chunks,
        reference_intent=detect_reference_intent(question),
        enabled=reference_cfg.enabled,
        exclude_from_evidence=reference_cfg.exclude_from_evidence,
        legacy_section_fallback=reference_cfg.legacy_section_fallback,
    )[:top_k]
    if not chunks:
        return {
            "answer": _NO_EVIDENCE_ZH if lang == "zh" else _NO_EVIDENCE,
            "citations": [],
            "chunks": [],
            "suspicious_citations": {"numeric": [], "author_year": [], "count": 0},
        }

    evidence = format_evidence(chunks)
    if lang == "zh":
        system = _SYSTEM_ZH
        user = f"问题: {question}\n\n证据:\n{evidence}\n\n回答(只允许 [chunk:<id>] 引用):"
    else:
        system = _SYSTEM
        user = f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer (with [chunk:<id>] citations only):"
    raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    cleaned, valid = validate_citations(raw, chunks)
    suspicious = detect_suspicious_citations(cleaned)
    if suspicious["count"]:
        log.warning(f"suspicious citations detected: {suspicious}")
        cleaned = strip_suspicious_citation_forms(cleaned)
    log.info(f"answer ok, citations valid={len(valid)} retrieved={len(chunks)}")
    return {
        "answer": cleaned,
        "citations": valid,
        "chunks": chunks,
        "suspicious_citations": suspicious,
    }


__all__ = [
    "answer",
]
