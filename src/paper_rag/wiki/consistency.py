"""Wiki 词条的轻量一致性检查(纯启发式, 无 LLM)。

只标记待人工复核的问题, 永不自动删除。定义长度门槛按语言分档:
中文信息密度高于英文, 12 个汉字已能构成完整定义, 照抄英文 20 字符
门槛会把合格的中文定义误报为过短。
"""

from __future__ import annotations

from .. import config as cfg
from .schema import WikiEntry, normalize_label


def check_entry(entry: WikiEntry) -> list[str]:
    c = cfg.load().wiki.consistency
    issues: list[str] = []

    # 语言未知时按较宽松的中文档判, 避免对中文定义误报
    min_chars = (
        c.min_definition_chars_en
        if entry.definition_language == "en"
        else c.min_definition_chars_zh
    )
    if not entry.definition or len(entry.definition.strip()) < min_chars:
        issues.append("definition_too_short")

    if not entry.key_papers:
        issues.append("no_key_papers")
    if entry.version > 10 and not entry.evidence_chunks:
        issues.append("high_version_no_evidence")
    if any(not normalize_label(lb.text) for lb in entry.labels):
        issues.append("empty_normalized_label")
    if entry.entry_id in entry.related:
        issues.append("self_related")
    return issues


def find_problematic_entries(entries: list[WikiEntry]) -> list[dict]:
    return [
        {"entry_id": e.entry_id, "name": e.name, "issues": issues}
        for e in entries
        if (issues := check_entry(e))
    ]
