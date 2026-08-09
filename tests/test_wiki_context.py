"""wiki/context.py QA 消费端契约。

钉死的关键行为:
- 语义: wiki 只是背景(role=background_not_evidence), 证据边界不破;
- 词面召回对中文问题有效(规范化包含判断, CJK 一等公民);
- 词面无命中时走语义兜底(Qdrant 词条库), 失败非致命返回空上下文;
- fingerprint = entry_id:version 有序拼接, 供 QA cache 失效;
- 改写提示的中文扩展: 中文定义能产出 CJK 关键短语(基准 [A-Za-z] 正则对
  中文定义颗粒无收)且过滤中文停用词; 英文行为与基准对齐。
"""

from __future__ import annotations

import importlib

from paper_rag.wiki.schema import WikiEntry, WikiLabel


def _mod():
    return importlib.import_module("paper_rag.wiki.context")


def _rl_entry(version: int = 3) -> WikiEntry:
    return WikiEntry(
        entry_id="concept:reinforcementlearning",
        name="Reinforcement Learning",
        category="method",
        definition="通过奖励信号驱动策略优化的学习范式。",
        definition_language="zh",
        labels=[
            WikiLabel(text="Reinforcement Learning", language="en", kind="primary"),
            WikiLabel(text="强化学习", language="zh", kind="translation"),
            WikiLabel(text="RL", language="en", kind="acronym"),
        ],
        key_papers=["arxiv:1811.12560", "arxiv:2005.01643"],
        evidence_chunks=["c1", "c2"],
        version=version,
    )


def _other_entry() -> WikiEntry:
    return WikiEntry(
        entry_id="concept:contrastivelearning",
        name="Contrastive Learning",
        category="method",
        definition="A method that learns representations by contrasting positives.",
        definition_language="en",
        labels=[WikiLabel(text="Contrastive Learning", language="en", kind="primary")],
        key_papers=["arxiv:2002.05709"],
        version=1,
    )


def test_zh_question_hits_via_label_and_fingerprint(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [_rl_entry(), _other_entry()])

    ctx = mod.resolve_wiki_context("什么是强化学习?")
    assert ctx["role"] == "background_not_evidence"
    assert [e["name"] for e in ctx["entries"]] == ["Reinforcement Learning"]
    assert ctx["fingerprint"] == "concept:reinforcementlearning:3"
    entry = ctx["entries"][0]
    assert "强化学习" in entry["aliases"]
    assert entry["key_papers"] == ["arxiv:1811.12560", "arxiv:2005.01643"]


def test_paper_overlap_scores_entry_without_name_hit(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [_rl_entry(), _other_entry()])

    ctx = mod.resolve_wiki_context("这篇论文的主要贡献?", paper_ids=["arxiv:2002.05709"])
    assert [e["entry_id"] for e in ctx["entries"]] == ["concept:contrastivelearning"]


def test_semantic_fallback_when_no_lexical_hit(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [_rl_entry()])
    monkeypatch.setattr(mod, "_embed", lambda text: [0.0] * 4)
    monkeypatch.setattr(
        mod.wstore,
        "search_qdrant",
        lambda vec, top_k=3: [{"entry_id": "concept:reinforcementlearning", "score": 0.8}],
    )
    monkeypatch.setattr(mod.wstore, "get_entry", lambda eid, **kw: _rl_entry())

    ctx = mod.resolve_wiki_context("agent 如何从环境反馈中改进决策?")
    assert [e["entry_id"] for e in ctx["entries"]] == ["concept:reinforcementlearning"]


def test_store_failure_returns_empty_context(monkeypatch):
    mod = _mod()

    def _boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(mod.wstore, "list_entries", _boom)
    ctx = mod.resolve_wiki_context("任何问题")
    assert ctx == {"role": "background_not_evidence", "fingerprint": "", "entries": []}


def test_format_background_carries_not_evidence_instruction(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [_rl_entry()])
    ctx = mod.resolve_wiki_context("强化学习")

    text = mod.format_wiki_background(ctx)
    assert "not evidence" in text or "不得引用" in text
    assert "Reinforcement Learning" in text
    assert "奖励信号" in text
    assert mod.format_wiki_background({"entries": []}) == ""


def test_format_background_strips_chunk_pseudo_citations(monkeypatch):
    """建条 prompt 要求定义内引用 [chunk:xx] 做 grounding, 但这些 id 属于词条
    自己的证据, 不是本轮检索结果。原样进背景块会诱导模型伪引用未检索到的
    chunk(真实 Demo 实证), 故消费端必须剥离。"""
    mod = _mod()
    entry = _rl_entry()
    entry.definition = "通过奖励信号驱动策略优化的学习范式 [chunk:c1]。参见 [chunk:c2]"
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [entry])
    ctx = mod.resolve_wiki_context("强化学习")

    text = mod.format_wiki_background(ctx)
    assert "[chunk:" not in text  # 伪引用已剥离
    assert "奖励信号驱动策略优化的学习范式" in text  # 定义正文保留
    # 剥离发生在进上下文时, entries 里也不应残留
    assert "[chunk:" not in ctx["entries"][0]["definition"]


def test_rewrite_hints_extract_cjk_and_en_phrases(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.wstore, "list_entries", lambda **kw: [_rl_entry(), _other_entry()])
    ctx = mod.resolve_wiki_context("强化学习 contrastive learning 对比")

    hints = mod.wiki_rewrite_hints(ctx)
    dense = hints["dense_queries"]
    # 名字与别名进 dense 扩展
    assert "Reinforcement Learning" in dense
    assert "强化学习" in dense
    # 中文定义产出 CJK 关键短语(剥离"通过"等停用词, 基准英文正则对此无收)
    assert any(p.startswith("奖励信号") for p in dense)
    # 英文定义仍按词提取(learns/method 属基准停用词, 剩余实词成短语)
    assert any("representations" in p and "positives" in p for p in dense)
    # bm25 查询非空, key_papers 覆盖两词条且去重
    assert hints["bm25_query"]
    assert {"arxiv:1811.12560", "arxiv:2002.05709"} <= set(hints["key_papers"])
    assert len(hints["key_papers"]) == len(set(hints["key_papers"]))
