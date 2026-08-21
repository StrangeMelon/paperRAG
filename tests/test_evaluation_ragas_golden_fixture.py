"""Regression checks for the checked-in RAGAS Golden Set composition."""

from __future__ import annotations

import json
from pathlib import Path


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "ragas_golden.json"


def test_ragas_golden_fixture_has_the_v1_coverage_matrix() -> None:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = data["test_cases"]

    assert data["schema_version"] == "ragas-eval.v1"
    assert len(cases) == 40
    assert len({case["id"] for case in cases}) == 40
    assert sum("direct" in case["tags"] for case in cases) == 14
    assert sum("cross-section-reasoning" in case["tags"] for case in cases) == 8
    assert sum("multi-paper" in case["tags"] for case in cases) == 8
    assert sum("definition" in case["tags"] for case in cases) == 4
    assert sum(case["expected_abstain"] for case in cases) == 4
    assert sum("cross-lingual" in case["tags"] for case in cases) == 2


def test_ragas_golden_fixture_keeps_abstain_cases_without_references() -> None:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    for case in data["test_cases"]:
        if case["expected_abstain"]:
            assert case["reference_answer"] is None
            assert case["reference_chunk_ids"] == []
        else:
            assert case["reference_answer"]
            assert case["paper_ids"]
            assert case["reference_chunk_ids"]
