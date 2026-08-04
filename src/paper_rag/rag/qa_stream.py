"""qa_agentic 的流式变体。

管道每推进一个阶段就 yield 一个事件, 调用方(DeerFlow lead agent / chat UI)
增量渲染, 不必干等完整链路跑完。

事件协议:
    {"event": "intent",     "data": {"intent": "factual", "top_k": 5, ...}}
    {"event": "rewrite",    "data": {"queries": [...], "keywords": "..."}}
    {"event": "retrieved",  "data": {"iter": 0, "n_chunks": 7}}
    {"event": "reflect",    "data": {"sufficiency": "sufficient", ...}}
    {"event": "abstain",    "data": {"decision": "confident|weak|no_evidence|...", ...}}
    {"event": "answer_chunk","data": {"text": "..."}}
    {"event": "done",       "data": {"citations": [...], "suspicious": {...}, "abstain": {...}}}
    {"event": "error",      "data": {"message": "..."}}

硬上限与 qa_agentic 相同(max_inner_iters / max_inner_tokens)。

流式边界定夺(llm 课记账兑现): 流式只存在于本模块的 `_stream_chat` 一处,
不下沉进 llm.py 公共 API——主链路保持全同步, 与基准同构。

相对基准的确认偏离(第十课):
a) 系统/用户 prompt、no-chunks 文案、拒答文案 zh/en 路由(qa_agentic 同款
   先例; weak hint 调 abstain 课的 weak_evidence_hint(language));
b) `_stream_chat` 补 `llm.extra_body` 透传——基准漏配: llm 课的透传只覆盖
   非流式 chat(), 思考型模型流式默认 enable_thinking=true 会先吐
   reasoning_content delta, 基准只认 delta.content, 思考 token 被静默丢弃但
   照常计费; 透传后配置可显式关闭。空表缺省时调用形参与基准逐键一致。
照抄并记账: max_tokens=600(基准与 qa_agentic 的 1024 不一致, 流式面向前端
短答案); 无 trace_id/metrics/cache/wiki/history(轻装分工, 非偷工)。
"""

from __future__ import annotations

from collections.abc import Generator

from .. import config as cfg
from ..retrieve.format import format_evidence
from ..retrieve.pipeline import retrieve_round_with_rewrite
from ..utils.logger import get_logger
from . import abstain as abstain_mod
from .citation_check import (
    detect_suspicious_citations,
    strip_suspicious_citation_forms,
    validate_citations,
)
from .evidence_select import select_evidence
from .intent_classifier import classify
from .query_rewrite import (
    _query_language,
    rewrite,  # re-export: 测试可 monkeypatch qa_stream.rewrite
)
from .reflect import reflect

log = get_logger("rag.qa_stream")

_SYSTEM = (
    "You are a careful academic research assistant. Answer ONLY using the "
    "evidence chunks provided. After each factual statement, cite the chunk "
    "with [chunk:<chunk_id>]. NEVER use [1], [2], or (Author 2020) style "
    "citations. Keep the answer concise (≤200 words). If insufficient "
    "evidence, say so explicitly."
)

_SYSTEM_ZH = (
    "你是一名严谨的学术研究助手。只能使用下面提供的证据块作答。每个事实陈述"
    "之后必须用 [chunk:<chunk_id>] 引用对应证据块。绝对不要使用 [1]、【1】、"
    "(作者 2020)、（张三等，2020）之类的数字或作者-年份引用。回答保持精炼"  # noqa: RUF001
    "(不超过 300 字)。证据不足以回答时必须明确说明。"
)

_NO_CHUNKS_MSG = "(no evidence found)"
_NO_CHUNKS_MSG_ZH = "(未检索到证据)"


def _retrieve_round(query: str, paper_ids, top_k: int) -> tuple[list[dict], dict]:
    # 传本模块的 `rewrite` 引用, monkeypatch qa_stream.rewrite 的测试仍然生效。
    return retrieve_round_with_rewrite(query, paper_ids, top_k, rewrite_fn=rewrite)


def _no_evidence_message(abstain_cfg, language: str) -> str:
    # 与 qa_agentic 同款语言路由(小函数就地复制, 两模块保持独立——基准姿态)
    if language == "zh":
        return abstain_cfg.no_evidence_message
    return getattr(abstain_cfg, "no_evidence_message_en", None) or abstain_cfg.no_evidence_message


def stream_answer(
    question: str, *, paper_ids: list[str] | None = None
) -> Generator[dict, None, None]:
    """随 agentic 管道推进逐事件 yield。"""
    c = cfg.load().rag
    language = _query_language(question)
    intent_cfg = classify(question)
    yield {"event": "intent", "data": intent_cfg}

    max_iter = min(intent_cfg["max_iter"], c.max_inner_iters)
    top_k = intent_cfg["top_k"]
    all_chunks: dict[str, dict] = {}
    current_query = question

    for it in range(max_iter):
        try:
            chunks, rw = _retrieve_round(current_query, paper_ids, top_k)
        except Exception as e:
            yield {"event": "error", "data": {"message": f"retrieve failed: {e}"}}
            return
        if it == 0:
            yield {
                "event": "rewrite",
                "data": {
                    "queries": rw.get("dense_queries", []),
                    "keywords": rw.get("bm25_query", ""),
                },
            }
        for ch in chunks:
            cid = ch.get("chunk_id")
            if cid and cid not in all_chunks:
                all_chunks[cid] = ch
        yield {"event": "retrieved", "data": {"iter": it, "n_chunks": len(chunks)}}

        if not chunks:
            break
        if c.enable_reflect and it < max_iter - 1:
            r = reflect(question, format_evidence(chunks))
            yield {"event": "reflect", "data": r}
            if r["sufficiency"] == "sufficient":
                break
            if r["follow_up"]:
                current_query = r["follow_up"]
                continue
            break
        else:
            break

    final_chunks = list(all_chunks.values())[: top_k * 2]
    if not final_chunks:
        yield {
            "event": "done",
            "data": {
                "answer": _NO_CHUNKS_MSG_ZH if language == "zh" else _NO_CHUNKS_MSG,
                "citations": [],
                "suspicious": {"count": 0},
                "degraded": "no_chunks",
                "abstain": {"decision": abstain_mod.DECISION_NO_CHUNKS},
            },
        }
        return

    # === ADR-0014 abstain 裁决 ===
    abstain_cfg = c.abstain
    abstain_result = abstain_mod.decide(
        final_chunks,
        enabled=abstain_cfg.enabled,
        threshold_low=abstain_cfg.threshold_low,
        threshold_high=abstain_cfg.threshold_high,
        min_chunks=abstain_cfg.min_chunks,
    )
    yield {"event": "abstain", "data": abstain_result}

    if abstain_result["decision"] == abstain_mod.DECISION_NO_EVIDENCE:
        # 整体跳过 LLM 流; 拒答文案也走 answer_chunk, 前端渲染路径统一。
        message = _no_evidence_message(abstain_cfg, language)
        yield {"event": "answer_chunk", "data": {"text": message}}
        yield {
            "event": "done",
            "data": {
                "answer": message,
                "citations": [],
                "suspicious": {"count": 0},
                "abstain": abstain_result,
                "n_chunks": len(final_chunks),
            },
        }
        return

    evidence_chunks, evidence_selection = select_evidence(
        question,
        final_chunks,
        intent=intent_cfg.get("intent"),
    )

    # 逐 token 流式作答。
    allowed_citations = " ".join(
        f"[chunk:{ch['chunk_id']}]" for ch in evidence_chunks if ch.get("chunk_id")
    )
    if language == "zh":
        system = _SYSTEM_ZH
        user = (
            f"问题: {question}\n\n证据:\n{format_evidence(evidence_chunks)}\n\n"
            f"允许的引用令牌: {allowed_citations}\n\n"
            "最多使用 2 个引用。选择最直接支撑答案的证据块; 不要因为背景块可用"
            "就顺手引用。\n\n"
            "回答(引用令牌必须从允许列表逐字拷贝; 绝不发明 [chunk:1]、"
            "[chunk:2]、[1] 或 (作者 2020) 式引用; 不超过 300 字):"
        )
    else:
        system = _SYSTEM
        user = (
            f"Question: {question}\n\nEvidence:\n{format_evidence(evidence_chunks)}\n\n"
            f"Allowed citation tokens: {allowed_citations}\n\n"
            "Use at most 2 citations. Choose the chunks that most directly support "
            "the answer; do not cite background chunks just because they are available.\n\n"
            "Answer (copy citation tokens EXACTLY from the allowed list; never invent "
            "[chunk:1], [chunk:2], [1], or (Author 2020) citations; ≤200 words):"
        )
    if abstain_result["decision"] == abstain_mod.DECISION_WEAK:
        user += abstain_mod.weak_evidence_hint(language)
    full = ""
    try:
        for tok in _stream_chat(system, user):
            full += tok
            yield {"event": "answer_chunk", "data": {"text": tok}}
    except Exception as e:
        yield {"event": "error", "data": {"message": f"chat stream failed: {e}"}}
        return

    cleaned, valid = validate_citations(full, evidence_chunks)
    suspicious = detect_suspicious_citations(cleaned)
    if suspicious["count"]:
        cleaned = strip_suspicious_citation_forms(cleaned)
    paper_ids_used = sorted({c.get("paper_id") for c in final_chunks if c.get("paper_id")})
    yield {
        "event": "done",
        "data": {
            "answer": cleaned,
            "citations": valid,
            "suspicious": suspicious,
            "abstain": abstain_result,
            "n_chunks": len(final_chunks),
            "evidence_chunks": evidence_chunks,
            "evidence_selection": evidence_selection,
            "paper_ids": paper_ids_used,
        },
    }


def _stream_chat(system: str, user: str):
    """从 OpenAI 兼容流式端点逐 token yield。"""
    c = cfg.load().llm
    chosen = c.chat_model
    if not chosen:
        raise RuntimeError("CHAT_MODEL not set")
    from .llm import get_client

    vendor_kwargs: dict = {}
    if c.extra_body:
        # 确认偏离 b: 与 llm.chat() 同款透传(思考型模型流式也受配置控制)
        vendor_kwargs["extra_body"] = dict(c.extra_body)
    resp = get_client().chat.completions.create(
        model=chosen,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=c.temperatures.stream,
        max_tokens=600,
        stream=True,
        **vendor_kwargs,
    )
    for chunk in resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content
