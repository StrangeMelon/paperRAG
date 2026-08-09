"""Acceptance assertions for the real evidence retrieval script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "accept_evidence_retrieval.py"
_SPEC = importlib.util.spec_from_file_location("accept_evidence_retrieval", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_execution = _MODULE.validate_execution


def _execution(**overrides):
    chunk = {
        "chunk_id": "c1",
        "paper_id": "paper:one",
        "score_dense": 0.8,
        "score_bm25": 4.2,
        "score_rrf": 0.03,
        "score_rerank": 0.9,
    }
    values = {
        "retrieval_id": "r_real",
        "public_decision": "confident",
        "candidate_chunks": [chunk],
        "evidence_chunks": [chunk],
        "allowed_chunk_ids": ["c1"],
        "trace": {
            "rewrites": [{"dense_queries": ["q", "rewrite"], "raw": {"variants": ["x"]}}],
            "iters": [{"query": "q", "n_retrieved": 1, "reflect": None}],
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_execution_accepts_full_real_component_signals() -> None:
    summary = validate_execution(_execution())

    assert summary["status"] == "accepted"
    assert summary["components"] == {
        "dense": True,
        "sparse": True,
        "rrf": True,
        "reranker": True,
        "llm_rewrite": True,
    }


@pytest.mark.parametrize(
    "missing_field",
    ["score_dense", "score_bm25", "score_rrf", "score_rerank"],
)
def test_validate_execution_rejects_missing_retrieval_component(missing_field: str) -> None:
    execution = _execution()
    execution.candidate_chunks[0].pop(missing_field)

    with pytest.raises(AssertionError, match="component"):
        validate_execution(execution)


def test_validate_execution_rejects_unselected_evidence() -> None:
    with pytest.raises(AssertionError, match="evidence"):
        validate_execution(_execution(evidence_chunks=[], allowed_chunk_ids=[]))
