"""wiki/store.py 行为契约: SQLite 为真相源, 关系表可索引, Qdrant 只是可重建镜像。

钉死的关键行为:
- 词条快照 + 标签/论文/证据关系表 + 追加式版本史的写读往返;
- 名字查找一律走 wiki_labels 的 text_norm 索引(中英文都命中), 不做全表扫描;
- merged_into 重定向: 旧 ID 与旧标签在合并后仍能命中目标词条, 关系被吸收,
  tombstone 不出现在默认列表;
- Qdrant 镜像失败不回滚 SQLite: 用 qdrant_dirty 标脏, 补偿同步后清除。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

from paper_rag.wiki.schema import WikiEntry, WikiLabel, make_entry_id


def _isolated_store(monkeypatch, tmp_path: Path) -> ModuleType:
    sqlite_store = importlib.import_module("paper_rag.store.sqlite_store")
    wstore = importlib.import_module("paper_rag.wiki.store")
    config = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(tmp_path / "wiki.sqlite")),
        qdrant=SimpleNamespace(collection_wiki="wiki_entries_test"),
    )
    # sqlite_store 与 wiki.store 共享同一 paper_rag.config 模块对象, 一次 patch 全覆盖
    monkeypatch.setattr(sqlite_store.cfg, "load", lambda: config)
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)
    return wstore


def _rl_entry() -> WikiEntry:
    return WikiEntry(
        entry_id=make_entry_id("Reinforcement Learning"),
        name="Reinforcement Learning",
        category="method",
        definition="A learning paradigm driven by reward signals.",
        definition_language="en",
        labels=[
            WikiLabel(text="Reinforcement Learning", language="en", kind="primary"),
            WikiLabel(text="强化学习", language="zh", kind="translation"),
            WikiLabel(text="RL", language="en", kind="acronym"),
        ],
        key_papers=["arxiv:1811.12560"],
        evidence_chunks=["arxiv:1811.12560:intro:0", "arxiv:1811.12560:intro:1"],
    )


def test_upsert_and_get_entry_roundtrip(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    entry = _rl_entry()

    saved = wstore.upsert_entry(entry, reason="created from arxiv:1811.12560")
    assert saved.version == 1

    got = wstore.get_entry(entry.entry_id)
    assert got is not None
    assert got.name == "Reinforcement Learning"
    assert got.definition_language == "en"
    assert {lb.text for lb in got.labels} == {"Reinforcement Learning", "强化学习", "RL"}
    assert got.key_papers == ["arxiv:1811.12560"]
    assert set(got.evidence_chunks) == set(entry.evidence_chunks)
    assert wstore.version_count(entry.entry_id) == 1


def test_upsert_existing_bumps_version_and_never_deletes(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    entry = _rl_entry()
    wstore.upsert_entry(entry, reason="create")

    entry2 = entry.model_copy(deep=True)
    entry2.definition = "通过奖励信号驱动策略优化的学习范式。"
    entry2.definition_language = "zh"
    entry2.labels = [WikiLabel(text="增强学习", language="zh", kind="variant")]
    entry2.key_papers = ["arxiv:2005.01643"]

    saved = wstore.upsert_entry(entry2, reason="patched from arxiv:2005.01643")
    assert saved.version == 2

    got = wstore.get_entry(entry.entry_id)
    # 标签与论文只追加不删除: 旧三个标签仍在, 新标签并入
    assert {lb.text for lb in got.labels} == {
        "Reinforcement Learning",
        "强化学习",
        "RL",
        "增强学习",
    }
    assert set(got.key_papers) == {"arxiv:1811.12560", "arxiv:2005.01643"}
    assert got.definition == "通过奖励信号驱动策略优化的学习范式。"
    assert wstore.version_count(entry.entry_id) == 2


def test_find_by_label_uses_normalized_index(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    wstore.upsert_entry(_rl_entry(), reason="create")

    rl_id = make_entry_id("Reinforcement Learning")
    # 英文: 大小写/连字符差异折叠
    assert wstore.find_by_label("reinforcement-learning") == [rl_id]
    # 中文: 词内空格折叠
    assert wstore.find_by_label("强化 学习") == [rl_id]
    # 缩写标签同样可查(短标签的合并限制在解析层, 不在存储层)
    assert wstore.find_by_label("rl") == [rl_id]
    assert wstore.find_by_label("不存在的概念") == []


def test_labels_and_relations_are_idempotent(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    entry = _rl_entry()
    wstore.upsert_entry(entry, reason="create")

    # 规范化等价的标签不重复入库
    added = wstore.add_labels(
        entry.entry_id,
        [WikiLabel(text="reinforcement learning", language="en", kind="variant")],
    )
    assert added == 0
    added = wstore.add_labels(
        entry.entry_id,
        [WikiLabel(text="逆强化学习", language="zh", kind="variant")],
    )
    assert added == 1

    assert wstore.add_key_papers(entry.entry_id, ["arxiv:1811.12560"]) == 0
    assert wstore.add_key_papers(entry.entry_id, ["arxiv:2203.02155"]) == 1
    assert (
        wstore.add_evidence(
            entry.entry_id,
            [{"chunk_id": "arxiv:1811.12560:intro:0", "paper_id": "arxiv:1811.12560"}],
        )
        == 0
    )
    assert (
        wstore.add_evidence(
            entry.entry_id,
            [{"chunk_id": "arxiv:2203.02155:m:1", "paper_id": "arxiv:2203.02155"}],
        )
        == 1
    )


def test_merge_redirect_and_absorb(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    target = _rl_entry()
    wstore.upsert_entry(target, reason="create")

    dup = WikiEntry(
        entry_id=make_entry_id("增强学习"),
        name="增强学习",
        category="method",
        definition="奖励驱动的学习。",
        definition_language="zh",
        labels=[WikiLabel(text="增强学习", language="zh", kind="primary")],
        key_papers=["arxiv:2005.01643"],
        evidence_chunks=["arxiv:2005.01643:m:0"],
    )
    wstore.upsert_entry(dup, reason="create duplicate")

    wstore.merge_entries(dup.entry_id, target.entry_id, reason="review: same concept")

    # 旧 ID 跟随重定向命中目标词条
    got = wstore.get_entry(dup.entry_id)
    assert got is not None
    assert got.entry_id == target.entry_id
    # tombstone 本体仍可显式取出(不删除旧事实)
    tomb = wstore.get_entry(dup.entry_id, follow_redirect=False)
    assert tomb.merged_into == target.entry_id
    # 旧标签解析到目标
    assert wstore.find_by_label("增强学习") == [target.entry_id]
    # 关系被吸收
    merged = wstore.get_entry(target.entry_id)
    assert "arxiv:2005.01643" in merged.key_papers
    assert "arxiv:2005.01643:m:0" in merged.evidence_chunks
    # 默认列表排除 tombstone
    ids = [e.entry_id for e in wstore.list_entries()]
    assert ids == [target.entry_id]


def test_qdrant_dirty_flag_lifecycle(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    entry = _rl_entry()
    wstore.upsert_entry(entry, reason="create")

    pending = wstore.pending_qdrant_entries()
    assert [e.entry_id for e in pending] == [entry.entry_id]

    wstore.mark_qdrant_synced(entry.entry_id)
    assert wstore.pending_qdrant_entries() == []

    # 再次 upsert 重新标脏 —— 镜像失败/滞后不回滚 SQLite, 只表现为脏标记存续
    wstore.upsert_entry(entry, reason="patch")
    assert [e.entry_id for e in wstore.pending_qdrant_entries()] == [entry.entry_id]


def test_mirror_entry_upserts_to_qdrant_and_clears_dirty(monkeypatch, tmp_path):
    wstore = _isolated_store(monkeypatch, tmp_path)
    entry = _rl_entry()
    wstore.upsert_entry(entry, reason="create")

    captured: dict = {}

    class _FakeClient:
        def upsert(self, collection_name, points, wait=True):
            captured["collection"] = collection_name
            captured["points"] = points

    qdrant_store = importlib.import_module("paper_rag.store.qdrant_store")
    monkeypatch.setattr(qdrant_store, "get_client", lambda: _FakeClient())

    wstore.mirror_entry(wstore.get_entry(entry.entry_id), [0.1, 0.2, 0.3])

    assert captured["collection"] == "wiki_entries_test"
    payload = captured["points"][0].payload
    assert payload["entry_id"] == entry.entry_id
    assert payload["name"] == "Reinforcement Learning"
    assert wstore.pending_qdrant_entries() == []
