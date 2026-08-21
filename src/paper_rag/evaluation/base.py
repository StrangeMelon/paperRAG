"""Contracts shared by evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """A backend that scores one question/answer/retrieval sample."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        generated_answer: str | None = None,
        ground_truth: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return metric names mapped to values in the inclusive 0..1 range."""

    @staticmethod
    def validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

    @staticmethod
    def validate_chunks(retrieved_chunks: list[dict]) -> None:
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")


def normalise_ids(value: Any) -> list[str]:
    """Read IDs from the supported Golden Set shapes."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("chunk_id", "id", "paper_id"):
            if key in value:
                return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        return [
            str(item.get("chunk_id", item.get("id", item)) if isinstance(item, dict) else item)
            for item in value
        ]
    raise ValueError(f"unsupported ID value: {type(value).__name__}")
