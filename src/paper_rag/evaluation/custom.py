"""Deterministic retrieval and citation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import BaseEvaluator, normalise_ids


class CustomEvaluator(BaseEvaluator):
    """Fast metrics suitable for every local regression run."""

    SUPPORTED_METRICS = {
        "hit_rate", # 是否命中了任意一个预期的chunk
        "mrr",      # 第一个命中的chunk的排名的倒数
        "recall",   # 命中的chunk数量 / 预期的chunk数量
        "paper_hit_rate",   # 是否命中了任意一个预期的paper
        "citation_precision",
        "citation_recall",
        "abstain_accuracy",
    }

    def __init__(self, metrics: Sequence[str] | None = None) -> None:
        names = [str(item).strip().lower() for item in (metrics or ("hit_rate", "mrr"))]
        unsupported = sorted(set(names) - self.SUPPORTED_METRICS)
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )
        self.metrics = names

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        generated_answer: str | None = None,
        ground_truth: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_chunks(retrieved_chunks)
        truth = ground_truth or {}
        retrieved_ids = [
            str(chunk.get("chunk_id", chunk.get("id", ""))) for chunk in retrieved_chunks
        ]
        expected_chunks = normalise_ids(truth.get("chunk_ids", truth.get("expected_chunk_ids")))
        retrieved_papers = {
            str(chunk.get("paper_id")) for chunk in retrieved_chunks if chunk.get("paper_id")
        }
        expected_papers = set(normalise_ids(truth.get("paper_ids", truth.get("expected_sources"))))
        citations = normalise_ids(kwargs.get("citations", truth.get("citations")))
        decision = str(kwargs.get("abstain_decision", ""))
        expected_abstain = truth.get("expected_abstain")

        result: dict[str, float] = {}
        hits = [item for item in retrieved_ids if item in set(expected_chunks)]
        if "hit_rate" in self.metrics:
            result["hit_rate"] = float(bool(hits)) if expected_chunks else 0.0
        if "mrr" in self.metrics:
            result["mrr"] = next(
                (
                    1.0 / index
                    for index, item in enumerate(retrieved_ids, 1)
                    if item in expected_chunks
                ),
                0.0,
            )
        if "recall" in self.metrics:
            result["recall"] = (
                len(set(hits)) / len(set(expected_chunks)) if expected_chunks else 0.0
            )
        if "paper_hit_rate" in self.metrics:
            result["paper_hit_rate"] = (
                float(bool(retrieved_papers & expected_papers)) if expected_papers else 0.0
            )
        if "citation_precision" in self.metrics:
            result["citation_precision"] = (
                len(set(citations) & set(retrieved_ids)) / len(citations) if citations else 0.0
            )
        if "citation_recall" in self.metrics:
            result["citation_recall"] = (
                len(set(citations) & set(expected_chunks)) / len(set(expected_chunks))
                if expected_chunks
                else 0.0
            )
        if "abstain_accuracy" in self.metrics:
            actual_abstain = decision in {"no_evidence", "abstain", "weak"}
            result["abstain_accuracy"] = (
                float(actual_abstain == bool(expected_abstain))
                if expected_abstain is not None
                else 0.0
            )
        return result
