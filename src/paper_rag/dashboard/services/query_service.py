"""Adapter from the three QA modes to one dashboard result contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from ...observability import new_trace_id
from .trace_store import QueryTraceStore


class QueryService:
    def __init__(
        self,
        *,
        store: QueryTraceStore | None = None,
        agentic_answer: Callable[..., dict[str, Any]] | None = None,
        simple_answer: Callable[..., dict[str, Any]] | None = None,
        stream_answer: Callable[..., Iterable[dict[str, Any]]] | None = None,
    ) -> None:
        self.store = store
        self._agentic_answer = agentic_answer
        self._simple_answer = simple_answer
        self._stream_answer = stream_answer

    def run(
        self,
        question: str,
        *,
        mode: str = "agentic",
        paper_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question cannot be empty")
        started = time.perf_counter()
        try:
            if mode == "agentic":
                raw = self._agentic(question, paper_ids)
            elif mode == "simple":
                raw = self._simple(question, paper_ids, top_k)
            elif mode == "stream":
                raw = self._stream(question, paper_ids)
            else:
                raise ValueError(f"unsupported QA mode: {mode}")
            result = self._normalise(question, mode, raw, started, paper_ids)
        except Exception as exc:
            result = {
                "trace_id": new_trace_id(),
                "query": question,
                "mode": mode,
                "status": "error",
                "created_at": datetime.now(UTC).isoformat(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "answer": "",
                "citations": [],
                "chunks": [],
                "evidence_chunks": [],
                "intent": "unknown",
                "abstain": "unknown",
                "paper_ids": paper_ids or [],
                "trace": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        if self.store is not None:
            self.store.append(result)
        return result

    def _agentic(self, question: str, paper_ids: list[str] | None) -> dict[str, Any]:
        if self._agentic_answer is None:
            from ...rag.qa_agentic import answer

            self._agentic_answer = answer
        return self._agentic_answer(question, paper_ids=paper_ids)

    def _simple(self, question: str, paper_ids: list[str] | None, top_k: int) -> dict[str, Any]:
        if self._simple_answer is None:
            from ...rag.qa_simple import answer

            self._simple_answer = answer
        return self._simple_answer(question, paper_ids=paper_ids, top_k=top_k)

    def _stream(self, question: str, paper_ids: list[str] | None) -> dict[str, Any]:
        if self._stream_answer is None:
            from ...rag.qa_stream import stream_answer

            self._stream_answer = stream_answer
        answer_parts: list[str] = []
        events: list[dict[str, Any]] = []
        intent: dict[str, Any] = {}
        done: dict[str, Any] = {}
        for event in self._stream_answer(question, paper_ids=paper_ids):
            events.append(event)
            event_type = event.get("event")
            data = event.get("data") or {}
            if event_type == "intent":
                intent = data
            elif event_type == "answer_chunk":
                answer_parts.append(str(data.get("text", "")))
            elif event_type == "done":
                done = data
            elif event_type == "error":
                raise RuntimeError(str(data.get("message", "stream failed")))
        return {
            "answer": "".join(answer_parts),
            "citations": done.get("citations", []),
            "chunks": done.get("chunks", []),
            "trace": {
                "trace_id": new_trace_id(),
                "intent": intent,
                "abstain": done.get("abstain", {}),
                "events": events,
            },
        }

    @staticmethod
    def _normalise(
        question: str,
        mode: str,
        raw: dict[str, Any],
        started: float,
        paper_ids: list[str] | None,
    ) -> dict[str, Any]:
        trace = raw.get("trace") or {}
        intent = trace.get("intent") or {}
        abstain = trace.get("abstain") or {}
        return {
            "trace_id": str(trace.get("trace_id") or new_trace_id()),
            "query": question,
            "mode": mode,
            "status": "ok",
            "created_at": datetime.now(UTC).isoformat(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "answer": str(raw.get("answer", "")),
            "citations": [str(item) for item in raw.get("citations", [])],
            "chunks": raw.get("chunks") or [],
            "evidence_chunks": raw.get("evidence_chunks") or raw.get("chunks") or [],
            "intent": str(intent.get("intent", intent or "unknown")),
            "abstain": str(abstain.get("decision", abstain or "unknown")),
            "paper_ids": paper_ids or [],
            "trace": trace,
            "suspicious_citations": raw.get("suspicious_citations") or {},
        }
