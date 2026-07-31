"""文本切块器真实验收: 真实解析产物经 split_sections 后逐节跑 chunk_text。

数据来自已验收的解析层真实输出(只读, 不产生任何新文件):
- demo-mineru-data/  MinerU 双语 GPU OCR 产物(1 篇中文期刊 + 2 篇英文会议论文)
- demo-pymupdf-data/ PyMuPDF 兜底产物(密排纯文本, 整篇无空行)

验收点(对应本课修复的英文隐式假设):
- 偏移即切片: 每个 chunk 满足 body[char_start:char_end] == text, 可回切原文;
- 上界保证: 超长段落(中文期刊单段 3545 token、PyMuPDF 密排段 1469 token)经句子
  切分/硬切后, 所有 chunk token <= target * 1.2(硬切等分的近似余量);
- 语言路由按 language.json 生效; overlap 防重守卫下不产生重复 chunk。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from paper_rag.chunk.section_splitter import split_sections
from paper_rag.chunk.text_chunker import _count_tokens, chunk_text
from paper_rag.config import load

REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    (
        "中文期刊(MinerU, zh)",
        "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
    ),
    (
        "Graph-Mamba(MinerU, en)",
        "demo-mineru-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
    ),
    (
        "LocAgent(MinerU, en)",
        "demo-mineru-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
    ),
    (
        "Graph-Mamba(PyMuPDF, None)",
        "demo-pymupdf-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
    ),
)


def _load_case(parsed_dir: Path) -> tuple[str, str | None]:
    md = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    language_file = parsed_dir / "language.json"
    language = None
    if language_file.is_file():
        language = json.loads(language_file.read_text(encoding="utf-8"))["document_language"]
    return md, language


def _max_paragraph_tokens(md: str) -> int:
    paras = [p.strip() for p in md.split("\n\n") if p.strip()]
    return max(_count_tokens(p) for p in paras)


def _check_case(idx: int, label: str, parsed_dir: Path) -> None:
    md, language = _load_case(parsed_dir)
    print(f"[{idx}/{len(CASES)}] {label}")

    target = load().chunk.text.target_tokens
    bound = math.ceil(target * 1.2)
    sections = split_sections(md, language=language)
    assert sections, "切分结果为空"

    total_chunks = 0
    max_chunk_tokens = 0
    for sec in sections:
        chunks = chunk_text(sec.body, language=language)
        assert chunks or not sec.body.strip(), f"非空章节切出空结果: {sec.name!r}"
        seen_texts = set()
        for c in chunks:
            assert sec.body[c.char_start : c.char_end] == c.text, (
                f"偏移不可回切原文: {sec.name!r} @ {c.char_start}"
            )
            assert c.text == c.text.strip() and c.text, f"chunk 文本异常: {sec.name!r}"
            tokens = _count_tokens(c.text)
            max_chunk_tokens = max(max_chunk_tokens, tokens)
            assert tokens <= bound, f"chunk 超过上界 {bound}: {sec.name!r} 实测 {tokens}"
            assert c.text not in seen_texts, f"重复 chunk(防重守卫失效): {sec.name!r}"
            seen_texts.add(c.text)
        total_chunks += len(chunks)

    max_para = _max_paragraph_tokens(md)
    print(
        f"      sections={len(sections)} chunks={total_chunks} "
        f"最大段落 {max_para}tok -> 最大 chunk {max_chunk_tokens}tok (上界 {bound})"
    )
    if max_para > target:
        assert max_chunk_tokens < max_para, "超长段落未被切分"
    print("      验收通过\n")


def main() -> None:
    for _, rel in CASES:
        if not (REPO_ROOT / rel / "paper.md").is_file():
            print(f"缺少真实解析产物: {rel}", file=sys.stderr)
            raise SystemExit(1)

    for i, (label, rel) in enumerate(CASES, start=1):
        _check_case(i, label, REPO_ROOT / rel)
    print("文本切块器真实验收通过: 2 解析器 x 中英双语, 共 4 份真实产物。")


if __name__ == "__main__":
    main()
