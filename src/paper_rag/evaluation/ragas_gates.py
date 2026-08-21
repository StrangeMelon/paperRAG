"""Quality gates for ragas-report.v1 without touching Custom report semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ragas_schema import RAGAS_REPORT_SCHEMA_VERSION

_IDENTITY_FIELDS = (
    "ragas_version",
    "judge_model",
    "embedding_model",
    "golden_set_sha256",
    "corpus_manifest_sha256",
)


def parse_metric_thresholds(
    values: list[str] | None, supported_metrics: set[str]
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in values or []:
        name, separator, raw_value = raw.partition("=")
        name = name.strip()
        if not separator or not name or not raw_value.strip():
            raise ValueError(f"invalid metric threshold: {raw!r}; expected metric=value")
        if name not in supported_metrics:
            raise ValueError(f"unknown RAGAS metric in threshold: {name}")
        if name in thresholds:
            raise ValueError(f"duplicate RAGAS metric threshold: {name}")
        try:
            threshold = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"invalid threshold value for {name}: {raw_value!r}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold for {name} must be between 0 and 1")
        thresholds[name] = threshold
    return thresholds


def load_ragas_baseline(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read RAGAS baseline: {source}") from exc
    _validate_report(payload, label="baseline")
    return payload


def evaluate_ragas_gates(
    report: dict[str, Any],
    *,
    fail_under: dict[str, float],
    min_coverage: dict[str, float],
    baseline: dict[str, Any] | None,
    max_regression: dict[str, float],
) -> list[dict[str, Any]]:
    _validate_report(report, label="current report")
    if max_regression and baseline is None:
        raise ValueError("--max-regression requires --compare-baseline")
    if baseline is not None:
        _validate_report(baseline, label="baseline")
        _validate_identity(report, baseline)

    aggregate = report["aggregate_metrics"]
    requested = set(fail_under) | set(min_coverage) | set(max_regression)
    missing = sorted(requested - set(aggregate))
    if missing:
        raise ValueError(f"RAGAS report does not contain metrics: {', '.join(missing)}")

    violations: list[dict[str, Any]] = []
    for metric, threshold in fail_under.items():
        actual = float(aggregate[metric]["mean"])
        if actual < threshold:
            violations.append(
                {
                    "gate": "fail_under",
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                }
            )
    for metric, threshold in min_coverage.items():
        actual = float(aggregate[metric]["coverage"])
        if actual < threshold:
            violations.append(
                {
                    "gate": "min_coverage",
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                }
            )
    if baseline is not None:
        baseline_metrics = baseline["aggregate_metrics"]
        for metric, tolerance in max_regression.items():
            if metric not in baseline_metrics:
                raise ValueError(f"RAGAS baseline does not contain metric: {metric}")
            baseline_mean = float(baseline_metrics[metric]["mean"])
            actual = float(aggregate[metric]["mean"])
            regression = baseline_mean - actual
            if regression > tolerance:
                violations.append(
                    {
                        "gate": "max_regression",
                        "metric": metric,
                        "actual": actual,
                        "baseline": baseline_mean,
                        "regression": regression,
                        "threshold": tolerance,
                    }
                )
    return violations


def _validate_report(report: Any, *, label: str) -> None:
    if not isinstance(report, dict) or report.get("schema_version") != RAGAS_REPORT_SCHEMA_VERSION:
        raise ValueError(f"{label} must use schema {RAGAS_REPORT_SCHEMA_VERSION}")
    if not isinstance(report.get("evaluation"), dict):
        raise ValueError(f"{label} is missing evaluation metadata")
    if not isinstance(report.get("aggregate_metrics"), dict):
        raise ValueError(f"{label} is missing aggregate_metrics")


def _validate_identity(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    current_metadata = current["evaluation"]
    baseline_metadata = baseline["evaluation"]
    mismatched = [
        field
        for field in _IDENTITY_FIELDS
        if current_metadata.get(field) != baseline_metadata.get(field)
    ]
    if mismatched:
        raise ValueError(f"RAGAS baseline is not comparable; mismatched: {', '.join(mismatched)}")
