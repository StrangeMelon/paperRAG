"""把 chunk 渲染成 LLM 可引用的证据文本块。

引用格式硬不变量([chunk:<id>], 绝不允许 [1] 或作者-年份)的物理源头:
每块证据头部逐字携带自己的引用令牌, 下游 LLM 据此产出合规引用,
citation_check 按同格式校验, 一头一尾锁死闭环。

信封文本保持英文(与基准一致, P6 收尾课确认): 这是给 LLM 看的协议文本,
不是用户可见内容; 中文论文正文原样放在 body。
"""

from __future__ import annotations


def format_evidence(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        cid = c.get("chunk_id")
        head = (
            "EVIDENCE CHUNK\n"
            f"Use this exact citation token when citing this chunk: [chunk:{cid}]\n"
            f"paper_id={c.get('paper_id')} section={c.get('section')} "
            f"modality={c.get('modality')} score={c.get('score', 0):.3f}"
        )
        body = (c.get("text") or "").strip()
        parts.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(parts)
