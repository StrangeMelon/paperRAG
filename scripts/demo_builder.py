"""切块组装器真实验收: 真实解析产物走完整 build_chunks 链路。

输入组装(demo-builder-data/parsed/<id>/, 每轮重建, .gitignore 已忽略):
- 3 篇 MinerU: 注标 paper.md 取自 demo-page-markers-data(需先跑
  scripts/demo_page_markers.py), language.json 取自 demo-mineru-data;
- 1 篇 PyMuPDF: demo-pymupdf-data 原样(标记天然存在, 无 language.json -> None)。
产出 sections/chunks 落盘 chunks.json 供人工查看。

验收点:
- 全局不变量: 每个 chunk 满足 md[char_start:char_end] == text(偏移精确化);
- 页码归属: 4 篇论文所有文本 chunk page 非空且随文档顺序单调不减
  (基准对 MinerU 论文全是 page=None, 本链路的核心修复);
- 语言贯通: zh 论文 context_text 用中文模板, en/None 用英文模板;
- 章节数与 splitter 真实验收一致(16/25/26/9)。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from paper_rag.chunk.builder import build_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "demo-builder-data/parsed"

CASES = (
    # (标签, 注标 md 来源, language.json 来源, 期望章节数, 语言, 标题)
    (
        "中文期刊(MinerU, zh)",
        "demo-page-markers-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
        "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
        16,
        "zh",
        "综合能源服务区块链的网络架构、交互模型与信用评价",
    ),
    (
        "Graph-Mamba(MinerU, en)",
        "demo-page-markers-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        "demo-mineru-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        25,
        "en",
        "Graph-Mamba: Towards Long-Range Graph Sequence Modeling",
    ),
    (
        "LocAgent(MinerU, en)",
        "demo-page-markers-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
        "demo-mineru-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
        26,
        "en",
        "LocAgent: Graph-Guided LLM Agents for Code Localization",
    ),
    (
        "Graph-Mamba(PyMuPDF, None)",
        "demo-pymupdf-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        None,
        9,
        None,
        "Graph-Mamba: Towards Long-Range Graph Sequence Modeling",
    ),
)


def _prepare_case(md_src: str, lang_src: str | None) -> Path:
    src_dir = REPO_ROOT / md_src
    # 同一论文可能有 MinerU/PyMuPDF 两种解析产物, 目录名带来源后缀避免互相覆盖
    flavor = "pymupdf" if "pymupdf" in md_src else "mineru"
    out_dir = OUT_ROOT / f"{src_dir.name}--{flavor}"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # 只清理本 Demo 自己的上一轮产物
    out_dir.mkdir(parents=True)
    shutil.copy2(src_dir / "paper.md", out_dir / "paper.md")
    if lang_src is not None:
        lang_file = REPO_ROOT / lang_src / "language.json"
        if lang_file.is_file():
            shutil.copy2(lang_file, out_dir / "language.json")
    return out_dir


def _check_case(
    idx: int, label: str, parsed_dir: Path, expect_sections: int, language: str | None, title: str
) -> None:
    print(f"[{idx}/{len(CASES)}] {label}")
    md = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    sections, chunks = build_chunks(parsed_dir.name, parsed_dir, title=title)

    assert len(sections) == expect_sections, f"章节数 {len(sections)} != {expect_sections}"
    assert chunks, "chunks 为空"

    prefix = "[标题: " if language == "zh" else "[Title: "
    pages = []
    for c in chunks:
        assert md[c["char_start"] : c["char_end"]] == c["text"], (
            f"全局偏移不变量被破坏: {c['section']!r} @ {c['char_start']}"
        )
        assert c["page"] is not None, f"page=None: {c['section']!r} @ {c['char_start']}"
        assert c["context_text"].startswith(prefix), f"语言模板不符: {c['context_text'][:30]!r}"
        pages.append(c["page"])
    assert pages == sorted(pages), "页码未随文档顺序单调不减"

    payload = {"sections": sections, "chunks": chunks}
    (parsed_dir / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"      sections={len(sections)} chunks={len(chunks)} "
        f"页码范围={min(pages)}-{max(pages)} 全部有页码、偏移可回切"
    )
    print(f"      产物已保存: {parsed_dir.relative_to(REPO_ROOT)}/chunks.json")


def main() -> None:
    for _, md_src, _, _, _, _ in CASES:
        if not (REPO_ROOT / md_src / "paper.md").is_file():
            print(
                f"缺少输入: {md_src}(MinerU 注标产物需先跑 scripts/demo_page_markers.py)",
                file=sys.stderr,
            )
            raise SystemExit(1)

    for i, (label, md_src, lang_src, n_sec, language, title) in enumerate(CASES, start=1):
        parsed_dir = _prepare_case(md_src, lang_src)
        _check_case(i, label, parsed_dir, n_sec, language, title)
        print()
    print("切块组装器真实验收通过: 2 解析器 x 中英双语, 页码/偏移/语言模板全链一致。")


if __name__ == "__main__":
    main()
