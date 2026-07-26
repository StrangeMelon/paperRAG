"""Tests for safe Hugging Face cache resolution."""

from __future__ import annotations

from paper_rag.utils.hf_cache import resolve_cached_snapshot


def test_resolver_ignores_partial_snapshot_without_model_weights(tmp_path):
    repo = tmp_path / "models--BAAI--bge-m3"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("revision", encoding="utf-8")

    assert resolve_cached_snapshot("BAAI/bge-m3", tmp_path) == "BAAI/bge-m3"
