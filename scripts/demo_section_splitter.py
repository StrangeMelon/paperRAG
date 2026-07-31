"""章节切分器真实验收: 用解析层真实产物跑 split_sections 并断言章节骨架。

数据来自已验收的解析层真实输出(只读, 不产生任何新文件):
- demo-mineru-data/  MinerU 双语 GPU OCR 产物(1 篇中文期刊 + 2 篇英文会议论文)
- demo-pymupdf-data/ PyMuPDF 兜底产物(密排纯文本, 整篇无空行)

验收点: 语言路由按 language.json 生效; 中文 markdown 标题的编号直贴形态
("# 1综合…" / "# 2.1综合…" / "# 4.2Dataset…")清洗干净; 行内摘要冒号切分;
参考文献尾部过滤; 英文密排版面下编号标题(无段落边界)仍可识别。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from paper_rag.chunk.section_splitter import RawSection, split_sections

REPO_ROOT = Path(__file__).resolve().parents[1]

ZH_JOURNAL = REPO_ROOT / "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566"
GRAPH_MAMBA = REPO_ROOT / "demo-mineru-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800"
LOCAGENT = REPO_ROOT / "demo-mineru-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932"
GRAPH_MAMBA_PYMUPDF = (
    REPO_ROOT / "demo-pymupdf-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800"
)


def _load_case(parsed_dir: Path) -> tuple[str, str | None]:
    md = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    language_file = parsed_dir / "language.json"
    language = None
    if language_file.is_file():
        language = json.loads(language_file.read_text(encoding="utf-8"))["document_language"]
    return md, language


def _print_sections(sections: list[RawSection]) -> None:
    for s in sections:
        print(f"      [{s.idx:2d}] L{s.level} {s.name[:52]!r} body_chars={len(s.body)}")


def _common_invariants(md: str, sections: list[RawSection]) -> None:
    assert sections, "切分结果为空"
    assert [s.idx for s in sections] == list(range(len(sections)))
    for s in sections:
        assert s.name.strip(), f"空标题: {s!r}"
        assert md[s.start : s.end].strip() == s.body, f"offset 不变量被破坏: {s.name!r}"


def _check_zh_journal() -> None:
    print("[1/4] 中文期刊(MinerU, language=zh): 综合能源服务区块链")
    md, language = _load_case(ZH_JOURNAL)
    assert language == "zh"
    sections = split_sections(md, language=language)
    _print_sections(sections)
    _common_invariants(md, sections)

    names = [s.name for s in sections]
    # 行内摘要冒号切分 + 中文规范标题骨架
    for expected in ("摘要", "引言", "结论", "参考文献"):
        assert expected in names, f"缺少章节: {expected}"
    abstract = sections[names.index("摘要")]
    assert abstract.body.startswith("综合能源服务是一种全新的能源服务模式")
    # 编号直贴标题清洗: "# 1综合…" / "# 2.1综合…" 都不得残留数字前缀
    assert "综合能源服务系统物理架构" in names
    assert "综合能源服务系统的主从多链结构模型" in names
    assert not any(name[0].isdigit() for name in names), "存在未清洗的数字前缀标题"
    # 参考文献尾部过滤: 文末重复的英文题名与条目噪声都被并入参考文献
    assert names[-1] == "参考文献"
    assert "Network Architecture" in sections[-1].body
    print(f"      sections={len(names)} 验收通过\n")


def _check_graph_mamba_mineru() -> None:
    print("[2/4] 英文论文(MinerU, language=en): Graph-Mamba")
    md, language = _load_case(GRAPH_MAMBA)
    assert language == "en"
    sections = split_sections(md, language=language)
    _print_sections(sections)
    _common_invariants(md, sections)

    names = [s.name for s in sections]
    for expected in ("Abstract", "Introduction", "Related Work", "Experiments", "Conclusion"):
        assert expected in names, f"缺少章节: {expected}"
    assert names[-1] == "References"
    assert len(names) >= 20, f"markdown 标题识别数量异常: {len(names)}"
    print(f"      sections={len(names)} 验收通过\n")


def _check_locagent_mineru() -> None:
    print("[3/4] 英文论文(MinerU, language=en): LocAgent")
    md, language = _load_case(LOCAGENT)
    assert language == "en"
    sections = split_sections(md, language=language)
    _print_sections(sections)
    _common_invariants(md, sections)

    names = [s.name for s in sections]
    for expected in ("Abstract", "Introduction", "Experiments", "Conclusion", "Limitations"):
        assert expected in names, f"缺少章节: {expected}"
    # "# 4.2Dataset Construction" 的编号直贴形态由点分前缀分支清洗(语言中立)
    assert "Dataset Construction" in names
    assert "Fine-tuned Open-source Models" in names
    assert names[-1] == "References"
    print(f"      sections={len(names)} 验收通过\n")


def _check_graph_mamba_pymupdf() -> None:
    print("[4/4] 英文密排纯文本(PyMuPDF 兜底, 无 language.json -> None): Graph-Mamba")
    md, language = _load_case(GRAPH_MAMBA_PYMUPDF)
    assert language is None
    sections = split_sections(md, language=language)
    _print_sections(sections)
    _common_invariants(md, sections)

    names = [s.name for s in sections]
    # 密排版面(无空行)下编号标题仍可识别 —— 修复前整篇只剩 Abstract/References
    for expected in ("Abstract", "Introduction", "Related Work", "Experiments", "Conclusion"):
        assert expected in names, f"缺少章节: {expected}"
    assert names[-1] == "References", "References 之后的附录条目应被尾部过滤"
    assert len(names) >= 6
    print(f"      sections={len(names)} 验收通过\n")


def main() -> None:
    for parsed_dir in (ZH_JOURNAL, GRAPH_MAMBA, LOCAGENT, GRAPH_MAMBA_PYMUPDF):
        if not (parsed_dir / "paper.md").is_file():
            print(f"缺少真实解析产物: {parsed_dir}", file=sys.stderr)
            raise SystemExit(1)

    _check_zh_journal()
    _check_graph_mamba_mineru()
    _check_locagent_mineru()
    _check_graph_mamba_pymupdf()
    print("章节切分器真实验收通过: 2 解析器 x 中英双语, 共 4 份真实产物。")


if __name__ == "__main__":
    main()
