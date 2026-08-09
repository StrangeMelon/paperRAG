"""QA 消费端: 只读 wiki 背景上下文, 永不作为答案证据。

三个消费口(与 qa_agentic / query_rewrite 既有插座签名一致):
- resolve_wiki_context(question, paper_ids) -> {"role", "fingerprint", "entries"}
- format_wiki_background(context) -> 进回答 prompt 的"背景不得引用"文本块
- wiki_rewrite_hints(context) -> 查询改写的 dense/bm25 扩展与 key_papers 路由

中文扩展(基准的确认缺陷):
- 词面召回用 normalize_label 包含判断, 中文问题命中中文标签;
- _definition_phrases 增加 CJK 分支: 基准 [A-Za-z] 正则对中文定义一个短语
  都抽不出来, 改写提示对中文词条完全失效; 这里抽 CJK 连续片段、剥离中文
  停用词后截前 8 字(BM25 侧交给检索层既有 bigram, 与 ADR-0001/0002 一致)。

规模注记: 词条扫描是每问一次的内存遍历, 词条量(千级)下可接受; 词面无命中
时走 Qdrant 语义兜底。失败一律非致命——wiki 坏了 QA 照常工作。
"""

from __future__ import annotations

import re
from typing import Any

from ..utils.logger import get_logger
from . import store as wstore
from .schema import WikiEntry, normalize_label

log = get_logger(__name__)

_ROLE = "background_not_evidence"
_EVIDENCE_WIKI_DEFINITION_MAX_CHARS = 1200
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_CJK_RUN_RE = re.compile(r"[一-鿿]{2,}")
# 建条 prompt 让 LLM 在定义里引用 [chunk:xx] 以确保有据可依, 但那些 id 是词条
# 自己的证据, 不属于本轮检索结果。进 QA 上下文前必须剥离, 否则模型会照抄成
# 伪引用(真实 Demo 实证)。证据边界: wiki 只做背景, 引用只能来自检索到的 chunk。
_CHUNK_CITE_RE = re.compile(r"\s*[\[(]\s*chunk:[^\])]*[\])]")

_STOPWORDS_EN = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "learn",
    "learns",
    "learning",
    "method",
    "paper",
    "that",
    "the",
    "this",
    "with",
}
# 中文定义里的组织词/套话: 单独成短语没有检索价值
_STOPWORDS_ZH = {
    "本文",
    "研究",
    "提出",
    "一种",
    "通过",
    "进行",
    "基于",
    "相关",
    "问题",
    "以及",
    "可以",
    "用于",
    "任务",
    "范式",
}


def _empty_context() -> dict[str, Any]:
    return {"role": _ROLE, "fingerprint": "", "entries": []}


def _embed(text: str) -> list[float]:
    from ..embed import bge_m3

    return bge_m3.encode_one(text)


def _strip_chunk_citations(text: str) -> str:
    return _CHUNK_CITE_RE.sub("", text or "").strip()


def _entry_to_context(entry: WikiEntry) -> dict[str, Any]:
    aliases = [lb.text for lb in entry.labels if lb.text != entry.name]
    return {
        "entry_id": entry.entry_id,
        "name": entry.name,
        "category": entry.category,
        "definition": _strip_chunk_citations(entry.definition),
        "definition_language": entry.definition_language,
        "aliases": aliases[:5],
        "key_papers": list(entry.key_papers or [])[:8],
        "evidence_chunks": list(entry.evidence_chunks or [])[:8],
        "version": int(entry.version or 1),
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }


def _fingerprint(entries: list[dict[str, Any]]) -> str:
    pairs = [f"{e.get('entry_id')}:{int(e.get('version') or 1)}" for e in entries]
    return "|".join(sorted(pairs))


def _score_entry(entry: WikiEntry, question_norm: str, paper_set: set[str]) -> int:
    score = 0
    if (n := normalize_label(entry.name)) and n in question_norm:
        score += 100
    else:
        for lb in entry.labels:
            norm = normalize_label(lb.text)
            # 短标签(rl 等两字符)词面包含误报率高, 不参与问题包含判断
            if norm and len(norm) >= 3 and norm in question_norm:
                score += 90
                break
    overlap = paper_set.intersection(entry.key_papers or [])
    if overlap:
        score += 50 + min(len(overlap), 3)
    return score


def resolve_wiki_context(
    question: str,
    paper_ids: list[str] | None = None,
    max_entries: int = 3,
) -> dict[str, Any]:
    """解析问题相关的 wiki 背景。任何失败都返回空上下文, 不影响 QA。"""
    try:
        entries = wstore.list_entries()
    except Exception as e:
        log.warning(f"wiki context skipped: {e}")
        return _empty_context()
    if not entries:
        return _empty_context()

    question_norm = normalize_label(question or "")
    paper_set = {str(pid) for pid in (paper_ids or []) if pid}
    scored: list[tuple[int, WikiEntry]] = []
    for entry in entries:
        score = _score_entry(entry, question_norm, paper_set)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        try:
            hits = wstore.search_qdrant(_embed(question), top_k=3)
            for h in hits:
                entry = wstore.get_entry(str(h.get("entry_id") or ""))
                if entry is not None:
                    scored.append((25, entry))
                    break
        except Exception as e:
            log.debug(f"wiki semantic context skipped: {e}")

    scored.sort(key=lambda item: (-item[0], item[1].entry_id))
    compact = [_entry_to_context(entry) for _, entry in scored[:max_entries]]
    return {"role": _ROLE, "fingerprint": _fingerprint(compact), "entries": compact}


def format_wiki_background(context: dict[str, Any]) -> str:
    entries = list((context or {}).get("entries") or [])
    if not entries:
        return ""
    blocks = [
        # 表头刻意不写 chunk 引用的字面格式: 背景块内任何位置出现该字面都会被
        # 模型当成可照抄的引用样例, 故整块保持零出现(测试钉死)。
        "Wiki background (not evidence). Do not cite anything from this background; "
        "cite only chunk ids that appear in the retrieved evidence section. "
        "If it conflicts with evidence, evidence wins.\n"
        "Wiki 背景(不得引用): 仅用于理解术语、别名与相关论文; 与检索证据冲突时以证据为准。"
    ]
    for entry in entries:
        aliases = ", ".join(entry.get("aliases") or [])
        key_papers = ", ".join(entry.get("key_papers") or [])
        line = (
            f"- {entry.get('name')} (v{entry.get('version', 1)}): {entry.get('definition', '')}"
        ).strip()
        if aliases:
            line += f"\n  aliases: {aliases}"
        if key_papers:
            line += f"\n  key_papers: {key_papers}"
        blocks.append(line)
    return "\n".join(blocks)


def _clip_definition(text: str, max_chars: int = _EVIDENCE_WIKI_DEFINITION_MAX_CHARS) -> str:
    text = _strip_chunk_citations(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def resolve_evidence_wiki_context(
    question: str,
    evidence_chunks: list[dict],
    *,
    max_entries: int = 3,
) -> dict[str, Any]:
    """Resolve compact Wiki background associated with selected evidence."""
    if max_entries <= 0 or not evidence_chunks:
        return _empty_context()
    try:
        raw_entries = wstore.list_entries()
        chunk_ids = {str(c.get("chunk_id")) for c in evidence_chunks if c.get("chunk_id")}
        paper_ids = {str(c.get("paper_id")) for c in evidence_chunks if c.get("paper_id")}
        question_norm = normalize_label(question or "")
        scored: dict[str, tuple[int, WikiEntry]] = {}
        for raw_entry in raw_entries:
            entry = raw_entry
            if raw_entry.merged_into:
                entry = wstore.get_entry(raw_entry.merged_into)
                if entry is None:
                    continue
            direct = len(chunk_ids.intersection(entry.evidence_chunks or []))
            paper = len(paper_ids.intersection(entry.key_papers or []))
            labels = [entry.name, *(label.text for label in entry.labels)]
            label_hit = any(
                (norm := normalize_label(label)) and norm in question_norm for label in labels
            )
            tier = 300 if direct else 200 if paper else 100 if label_hit else 0
            if tier:
                score = tier + min(direct, 10) + min(paper, 5)
                current = scored.get(entry.entry_id)
                if current is None or score > current[0]:
                    scored[entry.entry_id] = (score, entry)

        if len(scored) < max_entries:
            try:
                hits = wstore.search_qdrant(_embed(question), top_k=max_entries)
                for hit in hits:
                    entry_id = str(hit.get("entry_id") or "")
                    if not entry_id or entry_id in scored:
                        continue
                    entry = wstore.get_entry(entry_id)
                    if entry is not None:
                        scored[entry.entry_id] = (25, entry)
                    if len(scored) >= max_entries:
                        break
            except Exception as exc:
                log.debug(f"evidence wiki semantic context skipped: {exc}")

        ranked = sorted(scored.values(), key=lambda item: (-item[0], item[1].entry_id))
        selected = [entry for _, entry in ranked[:max_entries]]
        entries = [
            {"name": entry.name, "definition": _clip_definition(entry.definition)}
            for entry in selected
        ]
        fingerprint = _fingerprint(
            [{"entry_id": entry.entry_id, "version": entry.version} for entry in selected]
        )
        return {"role": _ROLE, "fingerprint": fingerprint, "entries": entries}
    except Exception as exc:
        log.warning(f"evidence wiki context skipped: {exc}")
        return _empty_context()


def _strip_zh_stopwords(run: str) -> str:
    changed = True
    while changed and run:
        changed = False
        for sw in _STOPWORDS_ZH:
            if run.startswith(sw):
                run = run[len(sw) :]
                changed = True
            if run.endswith(sw):
                run = run[: -len(sw)]
                changed = True
    return run


def _definition_phrases(definition: str) -> list[str]:
    """定义 -> 检索关键短语。英文按词提取(基准逻辑), 中文抽 CJK 片段并剥离
    停用词后截前 8 字; 两个分支独立, 中英混排定义两路都有产出。"""
    phrases: list[str] = []
    words = [
        w.lower() for w in _WORD_RE.findall(definition or "") if w.lower() not in _STOPWORDS_EN
    ]
    if words:
        phrases.append(" ".join(words[:6]))
        if len(words) >= 3:
            phrases.append(" ".join(words[-3:]))
    for run in _CJK_RUN_RE.findall(definition or ""):
        core = _strip_zh_stopwords(run)
        if len(core) >= 2 and core not in _STOPWORDS_ZH:
            phrases.append(core[:8])
    return [p for p in dict.fromkeys(phrases) if p]


def wiki_rewrite_hints(context: dict[str, Any]) -> dict[str, Any]:
    dense: list[str] = []
    bm25_terms: list[str] = []
    key_papers: list[str] = []
    for entry in (context or {}).get("entries") or []:
        name = (entry.get("name") or "").strip()
        if name:
            dense.append(name)
            bm25_terms.append(name.lower())
        for alias in entry.get("aliases") or []:
            if alias:
                dense.append(str(alias))
        for phrase in _definition_phrases(entry.get("definition") or ""):
            dense.append(phrase)
            bm25_terms.append(phrase)
        for paper_id in entry.get("key_papers") or []:
            if paper_id:
                key_papers.append(str(paper_id))

    dense = list(dict.fromkeys(dense))[:12]
    key_papers = list(dict.fromkeys(key_papers))[:8]
    return {
        "dense_queries": dense,
        "bm25_query": " ".join(dict.fromkeys(bm25_terms)),
        "key_papers": key_papers,
    }


__all__ = [
    "format_wiki_background",
    "resolve_evidence_wiki_context",
    "resolve_wiki_context",
    "wiki_rewrite_hints",
]
