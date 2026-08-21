#!/usr/bin/env python3
"""Real opt-in acceptance for the isolated RAGAS evaluation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_rag import config as cfg
from paper_rag.evaluation.ragas import RagasEvaluator
from paper_rag.evaluation.ragas_schema import RagasSample

_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _adapter_sample() -> RagasSample:
    return RagasSample(
        id="ragas-adapter-smoke",
        query="What does retrieval augmented generation do?",
        response="It retrieves relevant evidence before generating an answer.",
        retrieved_contexts=[
            "Retrieval augmented generation retrieves relevant passages and conditions "
            "the generator on that evidence."
        ],
        retrieved_chunk_ids=["adapter:c1"],
        citations=["adapter:c1"],
        reference="It retrieves evidence before answer generation.",
        reference_chunk_ids=["adapter:c1"],
        expected_abstain=False,
        actual_abstain="confident",
        tags=["adapter-smoke", "en"],
    )


def _run_adapter() -> dict:
    settings = cfg.load().evaluation.ragas
    evaluation = RagasEvaluator(_METRICS, settings=settings).evaluate_batch([_adapter_sample()])[0]
    if evaluation.status != "ok":
        errors = {
            name: observation.to_dict()
            for name, observation in evaluation.observations.items()
            if observation.status != "ok"
        }
        raise RuntimeError(f"RAGAS adapter acceptance failed: {errors}")
    return {
        "status": "accepted",
        "mode": "adapter-only",
        "metrics": evaluation.values,
        "judge_model": settings.judge_model,
        "embedding_model": settings.embedding_model,
    }


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    cfg.load.cache_clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--test-set", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        if args.adapter_only:
            payload = _run_adapter()
        else:
            from paper_rag.evaluation.ragas_runner import RagasEvalRunner
            from paper_rag.rag.qa_agentic import answer

            settings = cfg.load().evaluation.ragas
            test_set = args.test_set or Path(settings.golden_set)
            payload = (
                RagasEvalRunner(
                    answer,
                    RagasEvaluator(settings.metrics, settings=settings),
                    max_concurrency=settings.max_concurrency,
                )
                .run(test_set)
                .to_dict()
            )
    except Exception as exc:
        payload = {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
