"""wiki/review_queue.py 人工复核队列契约(真实临时 SQLite)。

钉死的关键行为:
- enqueue 24h 内按 (event_type, concept_norm, paper_id, reason) 去重, 返回原 id;
- resolve_merge 是复核闭环的执行动作: 调 store.merge_entries 完成词条合并
  (关系吸收 + tombstone 重定向), 并把复核行标记 resolved —— 没有这条路径,
  review 队列只是只能看不能动的列表;
- dismiss 标记"确实是不同概念", 不动词条。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _isolate(monkeypatch, tmp_path):
    sqlite_store = importlib.import_module("paper_rag.store.sqlite_store")
    config = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "wiki.sqlite")),
        qdrant=SimpleNamespace(collection_wiki="wiki_test"),
    )
    monkeypatch.setattr(sqlite_store.cfg, "load", lambda: config)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    rq = importlib.import_module("paper_rag.wiki.review_queue")
    wstore = importlib.import_module("paper_rag.wiki.store")
    return rq, wstore


def test_enqueue_dedupes_within_window(monkeypatch, tmp_path):
    rq, _ = _isolate(monkeypatch, tmp_path)

    rid = rq.enqueue("resolve_review", concept="强化学习", paper_id="arxiv:1", reason="llm_unsure")
    assert rid is not None
    dup = rq.enqueue("resolve_review", concept="强化 学习", paper_id="arxiv:1", reason="llm_unsure")
    assert dup == rid  # 规范化等价概念 + 同论文同原因 -> 去重
    other = rq.enqueue(
        "resolve_review", concept="强化学习", paper_id="arxiv:1", reason="exact_label_ambiguous"
    )
    assert other != rid
    assert rq.count_pending() == 2


def test_recent_returns_payload(monkeypatch, tmp_path):
    rq, _ = _isolate(monkeypatch, tmp_path)
    rq.enqueue(
        "resolve_review",
        concept="REINFORCE",
        paper_id="arxiv:2",
        reason="vector_recall+llm_unsure",
        payload={"candidates": [{"entry_id": "concept:reinforcementlearning"}]},
    )
    rows = rq.recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["concept"] == "REINFORCE"
    assert rows[0]["status"] == "pending"
    assert rows[0]["payload"]["candidates"][0]["entry_id"] == "concept:reinforcementlearning"


def _mk_entry(wstore, name, label_texts, papers):
    entry = WikiEntry(
        entry_id=f"concept:{name.lower().replace(' ', '')}",
        name=name,
        definition=f"definition of {name} long enough",
        labels=[WikiLabel(text=t, kind="variant") for t in label_texts],
        key_papers=papers,
    )
    wstore.upsert_entry(entry, reason="create")
    return entry


def test_resolve_merge_executes_store_merge(monkeypatch, tmp_path):
    rq, wstore = _isolate(monkeypatch, tmp_path)
    target = _mk_entry(wstore, "Reinforcement Learning", ["Reinforcement Learning"], ["arxiv:1"])
    dup = _mk_entry(wstore, "增强学习", ["增强学习"], ["arxiv:2"])
    rid = rq.enqueue("resolve_review", concept="增强学习", paper_id="arxiv:2", reason="unsure")

    rq.resolve_merge(rid, source_id=dup.entry_id, target_id=target.entry_id)

    # 词条已合并: 旧 ID 重定向, 关系吸收
    merged = wstore.get_entry(dup.entry_id)
    assert merged.entry_id == target.entry_id
    assert "arxiv:2" in merged.key_papers
    # 复核行闭环
    row = rq.recent(limit=1)[0]
    assert row["status"] == "resolved"
    assert rq.count_pending() == 0


def test_dismiss_marks_without_touching_entries(monkeypatch, tmp_path):
    rq, wstore = _isolate(monkeypatch, tmp_path)
    entry = _mk_entry(wstore, "REINFORCE", ["REINFORCE"], ["arxiv:3"])
    rid = rq.enqueue("resolve_review", concept="REINFORCE", paper_id="arxiv:3", reason="unsure")

    rq.dismiss(rid, note="distinct algorithm, keep separate")

    assert rq.recent(limit=1)[0]["status"] == "dismissed"
    assert wstore.get_entry(entry.entry_id).merged_into is None
    assert rq.count_pending() == 0
