#!/usr/bin/env python3
"""Real offline acceptance test for the evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from paper_rag.evaluation.custom import CustomEvaluator
from paper_rag.evaluation.runner import EvalRunner


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="paper-rag-eval-") as directory:
        golden_path = Path(directory) / "golden.json"
        golden_path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "test_cases": [
                        {
                            "id": "accept-001",
                            "query": "What is RAG?",
                            "expected_chunk_ids": ["c1"],
                            "expected_sources": ["p1"],
                            "expected_abstain": False,
                        },
                        {
                            "id": "accept-002",
                            "query": "Unknown topic",
                            "expected_chunk_ids": ["missing"],
                            "expected_sources": ["missing-paper"],
                            "expected_abstain": True,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        def fake_qa(query: str, *, paper_ids=None) -> dict:
            if query == "What is RAG?":
                return {
                    "answer": "RAG retrieves evidence before generation.",
                    "citations": ["c1"],
                    "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "RAG context"}],
                    "trace": {"abstain": {"decision": "confident"}},
                }
            return {
                "answer": "No relevant evidence.",
                "citations": [],
                "chunks": [],
                "trace": {"abstain": {"decision": "no_evidence"}},
            }

        metrics = ["hit_rate", "mrr", "recall", "paper_hit_rate", "abstain_accuracy"]
        report = EvalRunner(answer_fn=fake_qa, evaluator=CustomEvaluator(metrics=metrics)).run(golden_path)
        assert report.query_count == 2
        assert report.aggregate_metrics["hit_rate"] == 0.5
        assert report.aggregate_metrics["mrr"] == 0.5
        assert report.aggregate_metrics["paper_hit_rate"] == 0.5
        assert report.aggregate_metrics["abstain_accuracy"] == 1.0
        assert all(item.status == "ok" for item in report.query_results)

    print("evaluation acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
