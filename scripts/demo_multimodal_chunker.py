"""多模态切块真实验收: 真实解析产物走完整 build_chunks 链路验图/表/公式块。

输入组装(demo-multimodal-data/parsed/<id>--<来源>/, 每轮重建, .gitignore 已忽略):
- 3 篇 MinerU: 注标 paper.md 取自 demo-page-markers-data(需先跑
  scripts/demo_page_markers.py), language.json + layout.json 取自
  demo-mineru-data, figures/ 用符号链接引用真实图片;
- 1 篇 PyMuPDF: demo-pymupdf-data 的 paper.md 原样(无 layout, 三类元素实测为零,
  基准 docstring 承认的 reduced recall 在此如实记账为零召回)。

验收点:
- 多模态块数量与 layout 侦查一致: 中文期刊 19(图14/表5)、Graph-Mamba
  10(图3/表7)、LocAgent 16(图5/表11)——OCR 模式下 MinerU 把表格渲染成图片,
  重定型让它们以正确身份进索引;
- 每个多模态块: md[char_start:char_end] == raw_snippet(偏移不变量)、page 非空
  (图块自身 page_idx, 纯图表页也有页码)、asset_path 存在(真实图片文件);
- 语言路由: zh 论文所有多模态块嵌入文本用 图:/表: 前缀;
- 图注注入: 中文期刊与 Graph-Mamba 图注覆盖 100%, LocAgent 9/16(7 个 layout
  块本身无图注, 如实记账);
- vision enrich 钩子: vision 模块未重建, 每篇 build 会打一行
  "visual enrichment skipped" warning——这是预期的诚实信号, 非错误。
产出 mm_chunks.json 落盘供人工查看。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from paper_rag.chunk.builder import build_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "demo-multimodal-data/parsed"

CASES = (
    # (标签, 注标 md 来源, layout/language 来源, 期望多模态块 {modality: 数量}, 期望图注覆盖数, zh?)
    (
        "中文期刊(MinerU, zh)",
        "demo-page-markers-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
        "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
        {"figure": 14, "table": 5},
        19,
        True,
    ),
    (
        "Graph-Mamba(MinerU, en)",
        "demo-page-markers-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        "demo-mineru-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        {"figure": 3, "table": 7},
        10,
        False,
    ),
    (
        "LocAgent(MinerU, en)",
        "demo-page-markers-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
        "demo-mineru-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
        {"figure": 5, "table": 11},
        9,
        False,
    ),
    (
        "Graph-Mamba(PyMuPDF, None)",
        "demo-pymupdf-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800",
        None,
        {},
        0,
        False,
    ),
)

TITLES = {
    "sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566": "综合能源服务区块链的网络架构、交互模型与信用评价",
    "sha1_28acb520c921be7a1968207519dfa95d6af88800": "Graph-Mamba: Towards Long-Range Graph Sequence Modeling",
    "sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932": "LocAgent: Graph-Guided LLM Agents for Code Localization",
}


def _prepare_case(md_src: str, aux_src: str | None) -> Path:
    src_dir = REPO_ROOT / md_src
    flavor = "pymupdf" if "pymupdf" in md_src else "mineru"
    out_dir = OUT_ROOT / f"{src_dir.name}--{flavor}"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # 只清理本 Demo 自己的上一轮产物
    out_dir.mkdir(parents=True)
    shutil.copy2(src_dir / "paper.md", out_dir / "paper.md")
    if aux_src is not None:
        aux_dir = REPO_ROOT / aux_src
        for name in ("language.json", "layout.json"):
            if (aux_dir / name).is_file():
                shutil.copy2(aux_dir / name, out_dir / name)
        if (aux_dir / "figures").is_dir():
            (out_dir / "figures").symlink_to(aux_dir / "figures")  # 真实图片, 不复制
    return out_dir


def _captioned(c: dict) -> bool:
    """嵌入文本的语义行(前缀后第一段内容)是否非空。"""
    first, _, rest = c["text"].partition("\n")
    content = rest.partition("\n")[0] if first.rstrip(":") in ("表", "Table") else first
    return bool(content.split(":", 1)[-1].strip())


def _check_case(
    idx: int, label: str, parsed_dir: Path, expect: dict, expect_captioned: int, is_zh: bool
) -> None:
    print(f"[{idx}/{len(CASES)}] {label}")
    md = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    paper_id = parsed_dir.name.split("--")[0]
    _, chunks = build_chunks(paper_id, parsed_dir, title=TITLES[paper_id])

    mm = [c for c in chunks if c["modality"] != "text"]
    got = {}
    for c in mm:
        got[c["modality"]] = got.get(c["modality"], 0) + 1
    assert got == expect, f"多模态块分布 {got} != {expect}"

    for c in mm:
        assert md[c["char_start"] : c["char_end"]] == c["raw_snippet"], (
            f"raw_snippet 不可回切: {c['section']!r} @ {c['char_start']}"
        )
        assert c["page"] is not None and 1 <= c["page"] <= 30, f"页码异常: {c['page']}"
        assert c["asset_path"] and Path(c["asset_path"]).is_file(), (
            f"asset_path 不存在: {c['asset_path']}"
        )
        assert c["metadata"]["element_type"] == c["modality"]
        if is_zh:
            assert c["text"].startswith(("图: ", "表:\n")), f"zh 前缀不符: {c['text'][:20]!r}"

    n_captioned = sum(1 for c in mm if _captioned(c))
    assert n_captioned == expect_captioned, f"图注覆盖 {n_captioned} != {expect_captioned}"

    if mm:
        pages = [c["page"] for c in sorted(mm, key=lambda c: c["char_start"])]
        (parsed_dir / "mm_chunks.json").write_text(
            json.dumps(mm, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        n_text = len(chunks) - len(mm)
        print(
            f"      text={n_text} mm={len(mm)}({got}) 页码范围={min(pages)}-{max(pages)} "
            f"图注覆盖={n_captioned}/{len(mm)} 偏移可回切、图片文件存在"
        )
        print(f"      产物已保存: {parsed_dir.relative_to(REPO_ROOT)}/mm_chunks.json")
    else:
        print("      mm=0(PyMuPDF 密排版面三类元素实测为零, 如实记账的零召回)")


def main() -> None:
    for _, md_src, *_ in CASES:
        if not (REPO_ROOT / md_src / "paper.md").is_file():
            print(
                f"缺少输入: {md_src}(MinerU 注标产物需先跑 scripts/demo_page_markers.py)",
                file=sys.stderr,
            )
            raise SystemExit(1)

    for i, (label, md_src, aux_src, expect, expect_captioned, is_zh) in enumerate(CASES, start=1):
        parsed_dir = _prepare_case(md_src, aux_src)
        _check_case(i, label, parsed_dir, expect, expect_captioned, is_zh)
        print()
    print("多模态切块真实验收通过: 图/表重定型、自身页码、图注注入、双语前缀全链一致。")


if __name__ == "__main__":
    main()
