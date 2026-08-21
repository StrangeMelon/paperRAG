"""CLI routing contracts specific to the RAGAS backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_ragas_uses_an_independent_default_golden_set() -> None:
    assert _MODULE._default_test_set("qa", "ragas") == (
        "tests/fixtures/evaluation/ragas_golden.json"
    )
    assert _MODULE._default_test_set("qa", "custom") == ("tests/fixtures/evaluation/golden.json")
    assert _MODULE._default_test_set("retrieval", "custom") == (
        "tests/fixtures/evaluation/retrieval_golden.json"
    )


def test_positive_int_rejects_invalid_ragas_concurrency() -> None:
    assert _MODULE._positive_int("3") == 3
    with pytest.raises(Exception, match="positive integer"):
        _MODULE._positive_int("0")
    with pytest.raises(Exception, match="between 1 and 16"):
        _MODULE._positive_int("17")
