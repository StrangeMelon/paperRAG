"""Evidence-to-Wiki reverse association contracts."""

from __future__ import annotations

from types import SimpleNamespace

from paper_rag.wiki import context
from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _entry(
    entry_id: str,
    name: str,
    *,
    evidence: list[str] | None = None,
    papers: list[str] | None = None,
    definition: str = "A compact definition [chunk:wiki-cite].",
    labels: list[WikiLabel] | None = None,
    merged_into: str | None = None,
) -> WikiEntry:
    return WikiEntry(
        entry_id=entry_id,
        name=name,
        category="method",
        definition=definition,
        definition_language="en",
        labels=labels or [WikiLabel(text=name, language="en", kind="primary")],
        key_papers=papers or [],
        evidence_chunks=evidence or [],
        merged_into=merged_into,
    )


def test_evidence_chunk_association_has_priority_over_paper_and_label(monkeypatch) -> None:
    direct = _entry("concept:direct", "Direct", evidence=["c1"], papers=["p1"])
    paper = _entry("concept:paper", "Paper", papers=["p1"])
    label = _entry(
        "concept:label",
        "Graph Neural Network",
        labels=[WikiLabel(text="GNN", language="en", kind="acronym")],
    )
    monkeypatch.setattr(context.wstore, "list_entries", lambda **kwargs: [paper, label, direct])

    result = context.resolve_evidence_wiki_context(
        "What is GNN?",
        [{"chunk_id": "c1", "paper_id": "p1"}],
        max_entries=3,
    )

    assert [item["name"] for item in result["entries"]] == [
        "Direct",
        "Paper",
        "Graph Neural Network",
    ]


def test_evidence_wiki_deduplicates_redirect_targets_and_strips_citations(monkeypatch) -> None:
    target = _entry("concept:target", "Target", evidence=["c1"])
    monkeypatch.setattr(context.wstore, "list_entries", lambda **kwargs: [target])
    monkeypatch.setattr(context.wstore, "search_qdrant", lambda vector, top_k=5: [])

    result = context.resolve_evidence_wiki_context(
        "Target",
        [{"chunk_id": "c1", "paper_id": "p1"}],
        max_entries=3,
    )

    assert [item["name"] for item in result["entries"]] == ["Target"]
    assert "[chunk:" not in result["entries"][0]["definition"]
    assert set(result["entries"][0]) == {"name", "definition"}


def test_evidence_wiki_limits_entries_and_definition_budget(monkeypatch) -> None:
    entries = [
        _entry(
            f"concept:{index}", f"Concept {index}", evidence=[f"c{index}"], definition="x" * 5000
        )
        for index in range(4)
    ]
    monkeypatch.setattr(context.wstore, "list_entries", lambda **kwargs: entries)

    result = context.resolve_evidence_wiki_context(
        "question",
        [{"chunk_id": f"c{index}", "paper_id": "p1"} for index in range(4)],
        max_entries=2,
    )

    assert len(result["entries"]) == 2
    assert all(
        len(item["definition"]) <= context._EVIDENCE_WIKI_DEFINITION_MAX_CHARS
        for item in result["entries"]
    )


def test_evidence_wiki_uses_semantic_fallback_when_relations_miss(monkeypatch) -> None:
    entry = _entry("concept:semantic", "Semantic")
    monkeypatch.setattr(context.wstore, "list_entries", lambda **kwargs: [entry])
    monkeypatch.setattr(context, "_embed", lambda text: [0.0] * 4)
    monkeypatch.setattr(
        context.wstore,
        "search_qdrant",
        lambda vector, top_k=5: [{"entry_id": "concept:semantic", "score": 0.91}],
    )
    monkeypatch.setattr(context.wstore, "get_entry", lambda entry_id, **kwargs: entry)

    result = context.resolve_evidence_wiki_context(
        "unrelated question",
        [{"chunk_id": "c1", "paper_id": "p1"}],
        max_entries=1,
    )

    assert [item["name"] for item in result["entries"]] == ["Semantic"]


def test_evidence_wiki_failure_is_non_fatal(monkeypatch) -> None:
    monkeypatch.setattr(
        context.wstore, "list_entries", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down"))
    )

    assert context.resolve_evidence_wiki_context("q", [{"chunk_id": "c1"}]) == {
        "role": "background_not_evidence",
        "fingerprint": "",
        "entries": [],
    }
