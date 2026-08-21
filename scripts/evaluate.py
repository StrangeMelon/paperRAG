#!/usr/bin/env python3
"""Run the paper-rag Golden Set evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from paper_rag.evaluation.composite import CompositeEvaluator
from paper_rag.evaluation.custom import CustomEvaluator
from paper_rag.evaluation.ragas import RagasEvaluator
from paper_rag.evaluation.retrieval import RETRIEVAL_METRICS, RetrievalEvalRunner
from paper_rag.evaluation.runner import EvalRunner


def _build_evaluator(backend: str, metrics: list[str]):
    if backend == "custom":
        return CustomEvaluator(metrics=metrics)
    if backend == "ragas":
        return RagasEvaluator(metrics=metrics)
    if backend == "composite":
        custom_metrics = [name for name in metrics if name in CustomEvaluator.SUPPORTED_METRICS]
        ragas_metrics = [name for name in metrics if name in RagasEvaluator.SUPPORTED_METRICS]
        evaluators = []
        if custom_metrics:
            evaluators.append(("custom", CustomEvaluator(metrics=custom_metrics)))
        if ragas_metrics:
            evaluators.append(("ragas", RagasEvaluator(metrics=ragas_metrics)))
        return CompositeEvaluator(evaluators)
    raise ValueError(f"Unsupported backend: {backend}")


def _default_test_set(mode: str, backend: str) -> str:
    if mode == "retrieval":
        return "tests/fixtures/evaluation/retrieval_golden.json"
    if backend == "ragas":
        return "tests/fixtures/evaluation/ragas_golden.json"
    return "tests/fixtures/evaluation/golden.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 16:
        raise argparse.ArgumentTypeError("value must be a positive integer between 1 and 16")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set")
    parser.add_argument("--backend", choices=("custom", "ragas", "composite"), default="custom")
    parser.add_argument("--mode", choices=("qa", "retrieval"), default="qa")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--query-rewrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--max-concurrency", type=_positive_int)
    parser.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-under", action="append", default=[])
    parser.add_argument("--min-coverage", action="append", default=[])
    parser.add_argument("--compare-baseline", type=Path)
    parser.add_argument("--max-regression", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ragas_only_options_used = any(
        (
            args.max_concurrency is not None,
            args.fail_on_error is not None,
            bool(args.fail_under),
            bool(args.min_coverage),
            args.compare_baseline is not None,
            bool(args.max_regression),
        )
    )
    if args.backend != "ragas" and ragas_only_options_used:
        parser.error("RAGAS quality and concurrency options require --backend ragas")
    if args.test_set is None:
        args.test_set = _default_test_set(args.mode, args.backend)

    if args.mode == "retrieval":
        if args.backend != "custom":
            parser.error("retrieval mode only supports the custom backend")
        from paper_rag.retrieve.pipeline import retrieve_round

        metrics = tuple(args.metrics or RETRIEVAL_METRICS)
        report = RetrievalEvalRunner(
            retrieve_round,
            top_k=args.top_k,
            rewrite_enabled=args.query_rewrite,
            metrics=metrics,
        ).run(args.test_set)
        payload = report
        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            args.output = Path("data/evaluation") / f"retrieval-{stamp}.json"
    elif args.backend == "ragas":
        from dotenv import load_dotenv

        load_dotenv(override=False)
        from paper_rag import config as cfg
        from paper_rag.evaluation.ragas_gates import (
            evaluate_ragas_gates,
            load_ragas_baseline,
            parse_metric_thresholds,
        )
        from paper_rag.evaluation.ragas_runner import RagasEvalRunner
        from paper_rag.rag.qa_agentic import answer

        cfg.load.cache_clear()
        settings = cfg.load().evaluation.ragas
        if args.max_concurrency is not None:
            settings = settings.model_copy(update={"max_concurrency": args.max_concurrency})
        metrics = args.metrics or settings.metrics
        supported_metrics = set(metrics)
        try:
            fail_under = parse_metric_thresholds(args.fail_under, supported_metrics)
            min_coverage = parse_metric_thresholds(args.min_coverage, supported_metrics)
            max_regression = parse_metric_thresholds(args.max_regression, supported_metrics)
            baseline = load_ragas_baseline(args.compare_baseline) if args.compare_baseline else None
            if max_regression and baseline is None:
                raise ValueError("--max-regression requires --compare-baseline")
        except ValueError as exc:
            parser.error(str(exc))
        report = RagasEvalRunner(
            answer,
            RagasEvaluator(metrics=metrics, settings=settings),
            top_k=args.top_k,
            query_rewrite=args.query_rewrite,
            max_concurrency=settings.max_concurrency,
        ).run(args.test_set)
        payload = report.to_dict()
        try:
            gate_violations = evaluate_ragas_gates(
                payload,
                fail_under=fail_under,
                min_coverage=min_coverage,
                baseline=baseline,
                max_regression=max_regression,
            )
        except ValueError as exc:
            parser.error(str(exc))
        payload["quality_gate"] = {
            "passed": not gate_violations,
            "violations": gate_violations,
        }
        ragas_fail_on_error = (
            cfg.load().evaluation.fail_on_error
            if args.fail_on_error is None
            else args.fail_on_error
        )
        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            args.output = Path("data/evaluation") / f"ragas-{stamp}.json"
    else:
        from paper_rag.rag.qa_agentic import answer

        metrics = args.metrics or ["hit_rate", "mrr", "recall"]
        report = EvalRunner(
            answer_fn=answer, evaluator=_build_evaluator(args.backend, metrics)
        ).run(args.test_set)
        payload = report.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed_statuses = {"error", "qa_error"}
    if args.backend == "ragas" and args.mode == "qa":
        status_failed = ragas_fail_on_error and any(
            item["status"] in failed_statuses | {"partial"} for item in payload["query_results"]
        )
        return 1 if gate_violations or status_failed else 0
    return 1 if any(item["status"] in failed_statuses for item in payload["query_results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
