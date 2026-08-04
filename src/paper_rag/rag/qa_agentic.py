"""Agentic paper_qa: intent -> rewrite -> hybrid retrieve -> rerank -> reflect -> iterate.

闭环: 上层 Agent 只看到一次工具调用, 所有内部跳转都发生在这里。硬上限
max_inner_iters / max_inner_tokens 来自配置。

Output:
    {
      "answer": str,
      "citations": [chunk_id, ...],
      "chunks": [...],          # final chunks used for the answer
      "trace": {                # for debugging/inspection
        "intent": ...,
        "iters": [{"query":..., "n_retrieved":..., "reflect":...}, ...],
        "stopped_by": "answered" | "max_iters" | "no_evidence",
      }
    }

相对基准的确认偏离(第九课):
a) 前置依赖 observability(counter/histogram/new_trace_id)已随本课重建;
b) 系统/用户 prompt 与 no_chunks/chat 失败文案 zh/en 路由(复用
   _query_language; weak hint 改调 abstain 课的 weak_evidence_hint(language));
c) 拒答文案按语言路由: 配置新增 rag.abstain.no_evidence_message_en,
   关闭 abstain 课"英文问题收中文拒答文案"的记账项。
钩子照抄: qa_cache/history/research_memory/wiki 均为 try/except 非致命
lazy import, 模块未重建时打 warning 降级(诚实信号, vision/wiki 先例同款)。
"""

from __future__ import annotations

from time import perf_counter

from .. import config as cfg
from ..retrieve.format import format_evidence
from ..retrieve.pipeline import retrieve_round as _retrieve_round
from ..utils.logger import get_logger
from . import abstain as abstain_mod
from .citation_check import (
    detect_suspicious_citations,
    strip_suspicious_citation_forms,
    validate_citations,
)
from .evidence_select import select_evidence
from .intent_classifier import classify
from .llm import chat
from .query_rewrite import _query_language
from .reflect import reflect

log = get_logger("rag.qa_agentic")

_SYSTEM = (
    "You are a careful academic research assistant. Answer ONLY using the "
    "evidence chunks provided. After each factual statement, cite the chunk "
    "with [chunk:<chunk_id>]. NEVER use [1], [2], or (Author 2020) style "
    "citations — they will be considered hallucinated. Keep the answer "
    "concise: at most 200 words, dense and informative, no padding. If the "
    "evidence is insufficient, say so explicitly. Do NOT fabricate paper "
    "titles, numbers, authors, or years."
)

_SYSTEM_ZH = (
    "你是一名严谨的学术研究助手。只能使用下面提供的证据块作答。每个事实陈述"
    "之后必须用 [chunk:<chunk_id>] 引用对应证据块。绝对不要使用 [1]、【1】、"
    "(作者 2020)、（张三等，2020）之类的数字或作者-年份引用——它们会被判为"  # noqa: RUF001
    "幻觉引用。回答保持精炼: 不超过 300 字, 信息密集, 不注水。证据不足以回答"
    "时必须明确说明。不要编造论文标题、数字、作者或年份。"
)

_NO_CHUNKS_MSG = "(no evidence found in the indexed papers)"
_NO_CHUNKS_MSG_ZH = "(未在已索引文献中检索到证据)"
_CHAT_FAILED_MSG = "(LLM unavailable; see chunks for evidence)"
_CHAT_FAILED_MSG_ZH = "(LLM 不可用; 证据块见 chunks 字段)"

_EMPTY_SUSPICIOUS: dict = {"numeric": [], "author_year": [], "count": 0}


# ---------------------------------------------------------------------------
# Stage helpers — 每个 stage 独立可单测。
# ---------------------------------------------------------------------------


def _maybe_rewrite_with_history(question: str, conversation_id: str | None) -> str:
    """多轮会话时把历史折叠成自包含问题。失败非致命。"""
    if not conversation_id:
        return question
    try:
        from . import history

        rewritten = history.rewrite_with_history(question, conversation_id)
        if rewritten != question:
            log.info(f"history rewrite: {question!r} -> {rewritten!r}")
        return rewritten
    except Exception as e:
        log.warning(f"history rewrite failed (non-fatal): {e}")
        return question


def _resolve_wiki_context_safe(question: str, paper_ids: list[str] | None) -> dict:
    """解析 wiki 背景。失败非致命; wiki 只是背景, 永不作证据。"""
    try:
        from ..wiki.context import resolve_wiki_context

        return resolve_wiki_context(question, paper_ids=paper_ids)
    except Exception as e:
        log.warning(f"wiki context resolve failed (non-fatal): {e}")
        return {"role": "background_not_evidence", "fingerprint": "", "entries": []}


def _cache_question(question: str, wiki_context: dict | None) -> str:
    fingerprint = (wiki_context or {}).get("fingerprint") or ""
    if not fingerprint:
        return question
    return f"{question}\n\nwiki_context_fingerprint:{fingerprint}"


def _record_wiki_consumption_safe(
    *,
    question: str,
    paper_ids: list[str] | None,
    wiki_context: dict,
    trace_id: str,
) -> None:
    if not (wiki_context or {}).get("entries"):
        return
    try:
        from ..wiki.usage import record_consumption

        record_consumption(
            question=question,
            paper_ids=paper_ids,
            wiki_context=wiki_context,
            trace_id=trace_id,
        )
    except Exception as e:
        log.warning(f"wiki consumption record failed (non-fatal): {e}")


def _check_cache(question: str, paper_ids: list[str] | None, trace_id: str) -> dict | None:
    """qa_cache 短路。命中返回已按公开契约整形的响应, 否则 None。"""
    try:
        from ..observability import counter
        from . import qa_cache

        cached = qa_cache.get(question, paper_ids)
    except Exception as e:
        log.warning(f"qa_cache get failed (non-fatal): {e}")
        return None
    if cached is None:
        return None
    counter("paper_rag_qa_total", {"stop": "cache_hit"}).inc()
    return {
        "answer": cached.get("answer", ""),
        "citations": cached.get("citations", []),
        "chunks": [],  # 不回捞; chunk_ids 留在 trace 里
        "suspicious_citations": cached.get("suspicious_citations", _EMPTY_SUSPICIOUS),
        "trace": {
            **(cached.get("trace") or {}),
            "from_cache": True,
            "trace_id": trace_id,
            "cached_chunk_ids": cached.get("chunk_ids", []),
        },
    }


def _retrieve_loop(
    question: str,
    paper_ids: list[str] | None,
    top_k: int,
    max_iter: int,
    enable_reflect: bool,
    wiki_context: dict | None = None,
) -> tuple[dict[str, dict], list[dict], str]:
    """最多 max_iter 轮 retrieve+reflect。Returns (all_chunks, trace, stopped_by)。"""
    all_chunks: dict[str, dict] = {}
    trace: list[dict] = []
    current_query = question
    stopped = "max_iters"

    for it in range(max_iter):
        try:
            chunks = _retrieve_round(current_query, paper_ids, top_k, wiki_context=wiki_context)
        except TypeError as e:
            if "wiki_context" not in str(e):
                raise
            chunks = _retrieve_round(current_query, paper_ids, top_k)
        for ch in chunks:
            cid = ch.get("chunk_id")
            if cid and cid not in all_chunks:
                all_chunks[cid] = ch

        if not chunks:
            trace.append({"query": current_query, "n_retrieved": 0, "reflect": None})
            stopped = "no_evidence"
            break

        if enable_reflect and it < max_iter - 1:
            r = reflect(question, format_evidence(chunks))
            trace.append({"query": current_query, "n_retrieved": len(chunks), "reflect": r})
            if r["sufficiency"] == "sufficient":
                stopped = "answered"
                break
            if r["follow_up"]:
                current_query = r["follow_up"]
                continue
            stopped = "answered"
            break

        trace.append({"query": current_query, "n_retrieved": len(chunks), "reflect": None})
        stopped = "answered"
        break

    return all_chunks, trace, stopped


def _no_chunks_response(
    intent_cfg: dict,
    trace: list[dict],
    stopped: str,
    trace_id: str,
    wiki_context: dict | None = None,
    language: str = "en",
) -> dict:
    """检索零块时的最终响应。"""
    from ..observability import counter

    counter("paper_rag_qa_total", {"intent": intent_cfg["intent"], "stop": "no_chunks"}).inc()
    counter("paper_rag_qa_degraded_total", {"reason": "no_chunks"}).inc()
    counter("paper_rag_qa_abstain_total", {"decision": abstain_mod.DECISION_NO_CHUNKS}).inc()
    return {
        "answer": _NO_CHUNKS_MSG_ZH if language == "zh" else _NO_CHUNKS_MSG,
        "citations": [],
        "chunks": [],
        "suspicious_citations": _EMPTY_SUSPICIOUS,
        "trace": {
            "intent": intent_cfg,
            "iters": trace,
            "stopped_by": stopped,
            "degraded": "no_chunks",
            "abstain": {
                "decision": abstain_mod.DECISION_NO_CHUNKS,
                "evidence_score": 0.0,
                "n_chunks": 0,
            },
            "wiki_context": wiki_context
            or {"role": "background_not_evidence", "fingerprint": "", "entries": []},
            "trace_id": trace_id,
        },
    }


def _decide_abstain(final_chunks: list[dict], abstain_cfg) -> dict:
    """跑 abstain.decide 并打对应计数/日志。"""
    from ..observability import counter

    result = abstain_mod.decide(
        final_chunks,
        enabled=abstain_cfg.enabled,
        threshold_low=abstain_cfg.threshold_low,
        threshold_high=abstain_cfg.threshold_high,
        min_chunks=abstain_cfg.min_chunks,
    )
    counter("paper_rag_qa_abstain_total", {"decision": result["decision"]}).inc()
    if result.get("signal_quality") == "low_degraded":
        counter("paper_rag_qa_degraded_total", {"reason": "abstain_low_quality_signal"}).inc()
    log.info(
        f"abstain decision: {result['decision']} "
        f"score={result['evidence_score']:.3f} "
        f"top={result['top_chunk_score']:.3f} "
        f"field={result['score_field']} "
        f"quality={result.get('signal_quality')} "
        f"n={result['n_chunks']}"
    )
    return result


def _no_evidence_message(abstain_cfg, language: str) -> str:
    if language == "zh":
        return abstain_cfg.no_evidence_message
    return getattr(abstain_cfg, "no_evidence_message_en", None) or abstain_cfg.no_evidence_message


def _no_evidence_response(
    intent_cfg: dict,
    trace: list[dict],
    abstain_result: dict,
    abstain_cfg,
    final_chunks: list[dict],
    trace_id: str,
    wiki_context: dict | None = None,
    language: str = "en",
) -> dict:
    """abstain 判 no_evidence 时整体跳过 LLM。"""
    from ..observability import counter

    counter(
        "paper_rag_qa_total",
        {"intent": intent_cfg["intent"], "stop": "no_evidence_abstain"},
    ).inc()
    return {
        "answer": _no_evidence_message(abstain_cfg, language),
        "citations": [],
        "chunks": final_chunks,  # 仍返回块供检查/调试
        "suspicious_citations": _EMPTY_SUSPICIOUS,
        "trace": {
            "intent": intent_cfg,
            "iters": trace,
            "stopped_by": "no_evidence_abstain",
            "abstain": abstain_result,
            "wiki_context": wiki_context
            or {"role": "background_not_evidence", "fingerprint": "", "entries": []},
            "trace_id": trace_id,
        },
    }


def _build_user_prompt(
    question: str,
    final_chunks: list[dict],
    abstain_result: dict,
    wiki_context: dict | None = None,
    language: str = "en",
) -> str:
    evidence = format_evidence(final_chunks)
    wiki_block = ""
    try:
        from ..wiki.context import format_wiki_background

        wiki_block = format_wiki_background(wiki_context or {})
    except Exception as e:
        log.warning(f"wiki background formatting failed (non-fatal): {e}")
    allowed_citations = " ".join(
        f"[chunk:{ch['chunk_id']}]" for ch in final_chunks if ch.get("chunk_id")
    )
    wiki_section = f"\n\n{wiki_block}\n" if wiki_block else ""
    if language == "zh":
        user = (
            f"问题: {question}{wiki_section}\n证据:\n{evidence}\n\n"
            f"允许的引用令牌: {allowed_citations}\n\n"
            "最多使用 2 个引用。选择最直接支撑答案的证据块; 不要因为背景块可用"
            "就顺手引用。\n\n"
            "回答(引用令牌必须从允许列表逐字拷贝; 绝不发明 [chunk:1]、"
            "[chunk:2]、[1] 或 (作者 2020) 式引用):"
        )
    else:
        user = (
            f"Question: {question}{wiki_section}\nEvidence:\n{evidence}\n\n"
            f"Allowed citation tokens: {allowed_citations}\n\n"
            "Use at most 2 citations. Choose the chunks that most directly support "
            "the answer; do not cite background chunks just because they are available.\n\n"
            "Answer (copy citation tokens EXACTLY from the allowed list; never invent "
            "[chunk:1], [chunk:2], [1], or (Author 2020) citations):"
        )
    if abstain_result["decision"] == abstain_mod.DECISION_WEAK:
        # 注入显式不足提示——LLM 仍可作答, 但被要求标注不确定而不是幻觉引用。
        user += abstain_mod.weak_evidence_hint(language)
    return user


def _chat_failed_response(
    intent_cfg: dict,
    trace: list[dict],
    stopped: str,
    final_chunks: list[dict],
    evidence_chunks: list[dict],
    evidence_selection: dict,
    trace_id: str,
    err: Exception,
    wiki_context: dict | None = None,
    language: str = "en",
) -> dict:
    from ..observability import counter

    counter(
        "paper_rag_qa_total",
        {"intent": intent_cfg["intent"], "stop": "chat_error"},
    ).inc()
    counter("paper_rag_qa_degraded_total", {"reason": "chat_error"}).inc()
    return {
        "answer": _CHAT_FAILED_MSG_ZH if language == "zh" else _CHAT_FAILED_MSG,
        "citations": [],
        "chunks": final_chunks,
        "evidence_chunks": evidence_chunks,
        "suspicious_citations": _EMPTY_SUSPICIOUS,
        "trace": {
            "intent": intent_cfg,
            "iters": trace,
            "stopped_by": stopped,
            "degraded": f"chat_error:{type(err).__name__}",
            "evidence_selection": evidence_selection,
            "wiki_context": wiki_context
            or {"role": "background_not_evidence", "fingerprint": "", "entries": []},
            "trace_id": trace_id,
        },
    }


def _store_in_cache(question: str, paper_ids: list[str] | None, out: dict) -> None:
    try:
        from . import qa_cache

        qa_cache.put(question, paper_ids, out)
    except Exception as e:
        log.warning(f"qa_cache put failed (non-fatal): {e}")


def _first_wiki_concept(wiki_context: dict | None) -> str | None:
    entries = (wiki_context or {}).get("entries") or []
    if not entries:
        return None
    name = entries[0].get("name")
    return str(name) if name else None


def _first_paper_id(paper_ids: list[str] | None, chunks: list[dict] | None) -> str | None:
    if paper_ids:
        return paper_ids[0]
    for chunk in chunks or []:
        paper_id = chunk.get("paper_id")
        if paper_id:
            return str(paper_id)
    return None


def _enqueue_wiki_review_event(
    event_type: str,
    *,
    question: str,
    paper_ids: list[str] | None = None,
    chunks: list[dict] | None = None,
    wiki_context: dict | None = None,
    reason: str = "",
    trace_id: str | None = None,
    payload: dict | None = None,
    concept: str | None = None,
    paper_id: str | None = None,
) -> None:
    try:
        from ..wiki import review_queue

        review_queue.enqueue(
            event_type,
            concept=concept or _first_wiki_concept(wiki_context),
            paper_id=paper_id or _first_paper_id(paper_ids, chunks),
            question=question,
            reason=reason,
            payload={
                "trace_id": trace_id,
                "wiki_context_fingerprint": (wiki_context or {}).get("fingerprint", ""),
                **(payload or {}),
            },
        )
    except Exception as e:
        log.warning(f"wiki review enqueue failed (non-fatal): {e}")


def _persist_history(conversation_id: str | None, question: str, out: dict) -> None:
    if not conversation_id:
        return
    try:
        from . import history

        history.append(
            conversation_id,
            question,
            out.get("answer", ""),
            out.get("citations", []),
        )
    except Exception as e:
        log.warning(f"history.append failed (non-fatal): {e}")


def _maybe_rewrite_with_research_memory(
    question: str,
    conversation_id: str | None,
) -> tuple[str, dict]:
    """压缩研究记忆只作查询上下文, 永不作证据。"""
    if not conversation_id:
        return question, {
            "conversation_id": conversation_id,
            "memory_role": "query_context_only_not_evidence",
            "has_compressed_memory": False,
        }
    try:
        from . import research_memory

        rewritten, memory = research_memory.rewrite_with_memory(question, conversation_id)
        if rewritten != question:
            log.info(f"research memory rewrite: {question!r} -> {rewritten!r}")
        return rewritten, memory
    except Exception as e:
        log.warning(f"research memory rewrite failed (non-fatal): {e}")
        return question, {
            "conversation_id": conversation_id,
            "memory_role": "query_context_only_not_evidence",
            "has_compressed_memory": False,
            "error": type(e).__name__,
        }


def _attach_loop_trace(out: dict, *, latency_ms: int) -> None:
    """把调试 trace 归一成产品可读的 loop trace。"""
    trace = out.setdefault("trace", {})
    intent_cfg = trace.get("intent") or {}
    intent_name = intent_cfg.get("intent") if isinstance(intent_cfg, dict) else str(intent_cfg)
    iterations = trace.get("iters") or []
    chunks = out.get("chunks") or []
    evidence_chunks = out.get("evidence_chunks") or chunks
    trace["loop"] = {
        "intent": intent_name or "unknown",
        "intent_config": intent_cfg,
        "iterations": iterations,
        "stopped_by": trace.get("stopped_by", "unknown"),
        "abstain": trace.get("abstain") or {},
        "citations": out.get("citations", []),
        "n_chunks": len(chunks),
        "n_evidence_chunks": len(evidence_chunks),
        "latency_ms": latency_ms,
        "cost": {
            "llm_calls": None,
            "tokens": None,
            "note": "placeholder; provider usage accounting is not wired in v1",
        },
    }


def _persist_research_memory(
    conversation_id: str | None,
    question: str,
    out: dict,
) -> dict:
    if not conversation_id:
        return {
            "conversation_id": conversation_id,
            "memory_role": "query_context_only_not_evidence",
            "has_compressed_memory": False,
        }
    try:
        from . import research_memory

        trace_for_memory = {
            **(out.get("trace") or {}),
            "chunks": out.get("chunks") or [],
        }
        return research_memory.append(
            conversation_id,
            question,
            out.get("answer", ""),
            out.get("citations", []),
            trace=trace_for_memory,
        )
    except Exception as e:
        log.warning(f"research_memory.append failed (non-fatal): {e}")
        return {
            "conversation_id": conversation_id,
            "memory_role": "query_context_only_not_evidence",
            "has_compressed_memory": False,
            "error": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def answer(
    question: str,
    *,
    paper_ids: list[str] | None = None,
    conversation_id: str | None = None,
) -> dict:
    from ..observability import histogram, new_trace_id

    trace_id = new_trace_id()
    original_question = question
    question, memory_before = _maybe_rewrite_with_research_memory(question, conversation_id)
    timer = histogram("paper_rag_qa_latency_seconds")
    started = perf_counter()
    with timer.time():
        out = _answer_impl(
            question,
            paper_ids=paper_ids,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    latency_ms = int((perf_counter() - started) * 1000)
    _attach_loop_trace(out, latency_ms=latency_ms)
    memory_after = _persist_research_memory(conversation_id, original_question, out)
    out.setdefault("trace", {})["memory_before"] = memory_before
    out.setdefault("trace", {})["memory"] = memory_after
    _persist_history(conversation_id, original_question, out)
    return out


def _answer_impl(
    question: str,
    *,
    paper_ids: list[str] | None,
    trace_id: str,
    conversation_id: str | None = None,
) -> dict:
    from ..observability import counter

    # Stage 1 — 把会话历史折叠成自包含问题。
    question = _maybe_rewrite_with_history(question, conversation_id)
    language = _query_language(question)

    # Stage 2 — 解析 wiki 背景。wiki 只是查询/prompt 上下文, 永不作最终证据。
    wiki_context = _resolve_wiki_context_safe(question, paper_ids)
    _record_wiki_consumption_safe(
        question=question,
        paper_ids=paper_ids,
        wiki_context=wiki_context,
        trace_id=trace_id,
    )

    # Stage 3 — qa_cache 短路。有效键含 wiki 条目版本, 背景笔记打补丁后
    # 不会复用旧答案。
    question_for_cache = _cache_question(question, wiki_context)
    cached = _check_cache(question_for_cache, paper_ids, trace_id)
    if cached is not None:
        return cached

    # Stage 4 — 选 intent + 检索循环。
    c = cfg.load().rag
    intent_cfg = classify(question)
    max_iter = min(intent_cfg["max_iter"], c.max_inner_iters)
    top_k = intent_cfg["top_k"]
    all_chunks, trace, stopped = _retrieve_loop(
        question,
        paper_ids,
        top_k,
        max_iter,
        enable_reflect=c.enable_reflect,
        wiki_context=wiki_context,
    )

    # Stage 5 — 检索一无所获时短路。
    final_chunks = list(all_chunks.values())[: top_k * 2]
    if not final_chunks:
        _enqueue_wiki_review_event(
            "qa_no_chunks",
            question=question,
            paper_ids=paper_ids,
            wiki_context=wiki_context,
            reason="no_chunks",
            trace_id=trace_id,
            concept=_first_wiki_concept(wiki_context),
            paper_id=_first_paper_id(paper_ids, []),
        )
        return _no_chunks_response(intent_cfg, trace, stopped, trace_id, wiki_context, language)

    # Stage 6 — abstain 裁决(检索后、LLM 前, 见 ADR-0014)。
    abstain_cfg = c.abstain
    abstain_result = _decide_abstain(final_chunks, abstain_cfg)
    if abstain_result["decision"] == abstain_mod.DECISION_NO_EVIDENCE:
        _enqueue_wiki_review_event(
            "qa_no_evidence",
            question=question,
            paper_ids=paper_ids,
            chunks=final_chunks,
            wiki_context=wiki_context,
            reason="no_evidence",
            trace_id=trace_id,
            payload={"abstain": abstain_result},
            concept=_first_wiki_concept(wiki_context),
            paper_id=_first_paper_id(paper_ids, final_chunks),
        )
        return _no_evidence_response(
            intent_cfg,
            trace,
            abstain_result,
            abstain_cfg,
            final_chunks,
            trace_id,
            wiki_context,
            language,
        )
    if abstain_result["decision"] == abstain_mod.DECISION_WEAK:
        _enqueue_wiki_review_event(
            "qa_weak_evidence",
            question=question,
            paper_ids=paper_ids,
            chunks=final_chunks,
            wiki_context=wiki_context,
            reason="weak_evidence",
            trace_id=trace_id,
            payload={"abstain": abstain_result},
            concept=_first_wiki_concept(wiki_context),
            paper_id=_first_paper_id(paper_ids, final_chunks),
        )

    # Stage 7 — 确定性证据选择 + LLM 调用 + 引用清理。
    evidence_chunks, evidence_selection = select_evidence(
        question,
        final_chunks,
        intent=intent_cfg.get("intent"),
    )
    user = _build_user_prompt(question, evidence_chunks, abstain_result, wiki_context, language)
    system = _SYSTEM_ZH if language == "zh" else _SYSTEM
    try:
        raw = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=cfg.load().llm.temperatures.answer,
            max_tokens=1024,
        )
    except Exception as e:
        log.warning(f"chat failed, returning evidence-only: {e}")
        return _chat_failed_response(
            intent_cfg,
            trace,
            stopped,
            final_chunks,
            evidence_chunks,
            evidence_selection,
            trace_id,
            e,
            wiki_context,
            language,
        )

    cleaned, valid = validate_citations(raw, evidence_chunks)
    suspicious = detect_suspicious_citations(cleaned)
    if suspicious["count"]:
        log.warning(f"suspicious citations detected: {suspicious}")
        cleaned = strip_suspicious_citation_forms(cleaned)
    log.info(
        f"qa_agentic done: trace_id={trace_id} iters={len(trace)} stop={stopped} cites={len(valid)}"
    )
    counter("paper_rag_qa_total", {"intent": intent_cfg["intent"], "stop": stopped}).inc()
    counter("paper_rag_qa_citations_total").inc(len(valid))
    if suspicious["count"]:
        counter("paper_rag_qa_suspicious_total").inc(suspicious["count"])

    out = {
        "answer": cleaned,
        "citations": valid,
        "chunks": final_chunks,
        "evidence_chunks": evidence_chunks,
        "suspicious_citations": suspicious,
        "trace": {
            "intent": intent_cfg,
            "iters": trace,
            "stopped_by": stopped,
            "abstain": abstain_result,
            "evidence_selection": evidence_selection,
            "wiki_context": wiki_context,
            "trace_id": trace_id,
        },
    }
    _store_in_cache(question_for_cache, paper_ids, out)
    return out
