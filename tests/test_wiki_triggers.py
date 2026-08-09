"""wiki/triggers.py 编排契约(全依赖打桩)。

钉死的关键行为:
- kill switch: wiki.enabled=false 且非 force -> skipped;
- 质量门槛: parsed_with 命中黑名单或文本块数低于 min_chunks -> skipped 带原因
  (征文通知类文档不产词条, 如实记账);
- novel -> flow.create_entry -> store.upsert + Qdrant 镜像;
- match -> flow.patch_entry; **即使 patch 被 self_eval 拦下, 论文/证据/标签的
  机械关联仍直接落库**(关系新增不受限的另一半保障);
- review -> 进复核队列, 不建不并;
- 镜像失败非致命(脏标存续, worker 补偿), 单概念异常不中断整篇报告。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _mod():
    return importlib.import_module("paper_rag.wiki.triggers")


def _fake_config(*, enabled: bool = True):
    return SimpleNamespace(
        wiki=SimpleNamespace(
            enabled=enabled,
            quality_gate=SimpleNamespace(
                min_chunks=3,
                skip_parsed_with=["mineru+broken", "pymupdf+broken"],
            ),
        ),
    )


def _paper():
    return {"paper_id": "arxiv:zh1", "title": "某中文论文", "parsed_with": "mineru"}


def _chunks(n=5):
    return [{"chunk_id": f"c{i}", "section": "方法", "text": f"第 {i} 段内容"} for i in range(n)]


def _concept():
    return {
        "name": "Reinforcement Learning",
        "surface_name": "强化学习",
        "category": "method",
        "labels_zh": ["强化学习"],
        "labels_en": ["RL"],
        "evidence_chunk_ids": ["c1", "c2"],
        "confidence": 0.9,
    }


def _entry() -> WikiEntry:
    return WikiEntry(
        entry_id="concept:reinforcementlearning",
        name="Reinforcement Learning",
        category="method",
        definition="def",
        labels=[WikiLabel(text="Reinforcement Learning", kind="primary")],
    )


def _setup(
    monkeypatch,
    *,
    enabled=True,
    paper=-1,
    chunks=None,
    concepts=None,
    decision="novel",
    created=None,
    patched=None,
):
    mod = _mod()
    calls = {
        "upserts": [],
        "papers": [],
        "evidence": [],
        "labels": [],
        "mirror": [],
        "review": [],
    }
    monkeypatch.setattr(mod.cfg, "load", lambda: _fake_config(enabled=enabled))
    monkeypatch.setattr(mod, "_load_paper", lambda pid: _paper() if paper == -1 else paper)
    monkeypatch.setattr(
        mod, "_load_text_chunks", lambda pid: chunks if chunks is not None else _chunks()
    )
    monkeypatch.setattr(
        mod.concept_extractor,
        "extract_concepts",
        lambda **kw: concepts if concepts is not None else [_concept()],
    )
    monkeypatch.setattr(
        mod.normalize,
        "resolve_concept",
        lambda name, **kw: {
            "decision": decision,
            "entry_id": "concept:reinforcementlearning" if decision == "match" else None,
            "candidates": [{"entry_id": "concept:x"}] if decision == "review" else [],
            "reason": "test",
        },
    )
    monkeypatch.setattr(mod.flow, "create_entry", lambda **kw: created)
    monkeypatch.setattr(mod.flow, "patch_entry", lambda **kw: patched)
    monkeypatch.setattr(mod.wstore, "get_entry", lambda eid, **kw: _entry())
    monkeypatch.setattr(
        mod.wstore, "upsert_entry", lambda e, **kw: calls["upserts"].append(e.entry_id) or e
    )
    monkeypatch.setattr(
        mod.wstore, "add_key_papers", lambda eid, pids: calls["papers"].append((eid, pids)) or 1
    )
    monkeypatch.setattr(
        mod.wstore, "add_evidence", lambda eid, items: calls["evidence"].append((eid, items)) or 1
    )
    monkeypatch.setattr(
        mod.wstore, "add_labels", lambda eid, labels: calls["labels"].append((eid, labels)) or 1
    )
    monkeypatch.setattr(mod, "_mirror", lambda e: calls["mirror"].append(e.entry_id))
    monkeypatch.setattr(mod, "_enqueue_review", lambda **kw: calls["review"].append(kw))
    return mod, calls


def test_kill_switch_and_force(monkeypatch):
    mod, _ = _setup(monkeypatch, enabled=False)
    assert mod.on_paper_indexed("arxiv:zh1")["skipped"] == "wiki_disabled"
    # force 绕过 kill switch(供 backfill 使用)
    report = mod.on_paper_indexed("arxiv:zh1", force=True)
    assert "skipped" not in report


def test_paper_not_found_is_error(monkeypatch):
    mod, _ = _setup(monkeypatch, paper=None)
    assert "error" in mod.on_paper_indexed("arxiv:missing")


def test_quality_gate_skips_broken_and_tiny_docs(monkeypatch):
    broken = {"paper_id": "sha1:n", "title": "征文通知", "parsed_with": "mineru+broken"}
    mod, _ = _setup(monkeypatch, paper=broken)
    report = mod.on_paper_indexed("sha1:n")
    assert report["skipped"] == "parsed_with=mineru+broken"

    mod2, _ = _setup(monkeypatch, chunks=_chunks(2))  # min_chunks=3
    report2 = mod2.on_paper_indexed("arxiv:zh1")
    assert "chunks" in report2["skipped"]


def test_novel_creates_and_mirrors(monkeypatch):
    mod, calls = _setup(monkeypatch, decision="novel", created=_entry())
    report = mod.on_paper_indexed("arxiv:zh1", language="zh")
    assert report["created"] == 1
    assert calls["upserts"] == ["concept:reinforcementlearning"]
    assert calls["mirror"] == ["concept:reinforcementlearning"]


def test_novel_create_dropped_by_gate(monkeypatch):
    mod, calls = _setup(monkeypatch, decision="novel", created=None)
    report = mod.on_paper_indexed("arxiv:zh1", language="zh")
    assert report["created"] == 0 and report["dropped"] == 1
    assert calls["upserts"] == []


def test_match_records_relations_even_if_patch_dropped(monkeypatch):
    # patch 被 self_eval 拦下(None): 论文/证据/标签的机械关联仍要落库
    mod, calls = _setup(monkeypatch, decision="match", patched=None)
    report = mod.on_paper_indexed("arxiv:zh1", language="zh")
    assert report["patched"] == 0
    assert calls["papers"] == [("concept:reinforcementlearning", ["arxiv:zh1"])]
    (eid, items) = calls["evidence"][0]
    assert eid == "concept:reinforcementlearning"
    assert {i["chunk_id"] for i in items} == {"c1", "c2"}
    assert all(i["paper_id"] == "arxiv:zh1" for i in items)
    # 抽取器给出的双语标签也机械并入
    (_eid_l, labels) = calls["labels"][0]
    assert {lb.text for lb in labels} == {"强化学习", "RL"}


def test_match_with_successful_patch_upserts(monkeypatch):
    mod, calls = _setup(monkeypatch, decision="match", patched=_entry())
    report = mod.on_paper_indexed("arxiv:zh1", language="zh")
    assert report["patched"] == 1
    assert calls["upserts"] == ["concept:reinforcementlearning"]
    assert calls["mirror"] == ["concept:reinforcementlearning"]


def test_review_enqueues_without_create_or_merge(monkeypatch):
    mod, calls = _setup(monkeypatch, decision="review")
    report = mod.on_paper_indexed("arxiv:zh1", language="zh")
    assert report["review"] == 1
    assert calls["upserts"] == []
    assert calls["review"][0]["concept"]["name"] == "Reinforcement Learning"


def test_no_concepts_returns_zero_report(monkeypatch):
    mod, _calls = _setup(monkeypatch, concepts=[])
    report = mod.on_paper_indexed("arxiv:zh1")
    assert report == {
        "paper_id": "arxiv:zh1",
        "created": 0,
        "patched": 0,
        "review": 0,
        "dropped": 0,
    }
