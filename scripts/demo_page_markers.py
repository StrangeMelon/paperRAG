"""页码标记注入真实验收: 真实 MinerU 产物注标 + 标准化链路集成。

数据来自已验收的解析层真实输出(只读):
- demo-mineru-data/  MinerU 双语 GPU OCR 产物(1 篇中文期刊 + 2 篇英文会议论文)

集成段的产物持久化到 demo-page-markers-data/parsed/<id>/(已被 .gitignore 忽略,
每次运行覆盖), 便于人工查看注标后的 paper.md; 存量 demo-mineru-data/ 不被改写。

验收点(方案 A):
- 纯函数: 对 3 份真实 paper.md + layout.json 注标, 标记页码严格递增、1 基,
  覆盖率 >= 页数 - 1(纯图表页无可锚定文本, 允许优雅降级), 剥掉标记逐字节还原原文;
- 集成: 用真实 _mineru_raw 原始产物走 _normalize_into, 产出的 paper.md 剥掉
  标记后与已验收的存量 paper.md 逐字节一致(即"只多标记, 不动内容")。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from paper_rag.parse.mineru_local import _locate_outputs, _normalize_into
from paper_rag.parse.page_markers import inject_page_markers

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "demo-page-markers-data/parsed"

CASES = (
    ("中文期刊(MinerU, zh)", "sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566"),
    ("Graph-Mamba(MinerU, en)", "sha1_28acb520c921be7a1968207519dfa95d6af88800"),
    ("LocAgent(MinerU, en)", "sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932"),
)

_MARKER_RE = re.compile(r"<!-- page (\d+) -->")


def _strip_markers(md: str) -> str:
    return re.sub(r"<!-- page \d+ -->\n\n", "", md)


def _layout_pages(layout: list) -> list[int]:
    return sorted(
        {
            b["page_idx"]
            for b in layout
            if isinstance(b, dict) and isinstance(b.get("page_idx"), int)
        }
    )


def _check_pure_function(idx: int, label: str, parsed_dir: Path) -> None:
    print(f"[{idx}/{len(CASES)}] {label}: 纯函数注标")
    md = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    layout = json.loads((parsed_dir / "layout.json").read_text(encoding="utf-8"))
    pages = _layout_pages(layout)

    out = inject_page_markers(md, layout)
    marks = [int(m.group(1)) for m in _MARKER_RE.finditer(out)]

    assert marks == sorted(set(marks)), f"页码非严格递增: {marks}"
    assert all(1 <= p <= len(pages) + 1 for p in marks), f"页码越界: {marks}"
    missing = sorted(set(p + 1 for p in pages) - set(marks))
    assert len(missing) <= 1, f"缺页超过 1 页: {missing}"
    for page in missing:  # 允许的缺页必须是无可锚定文本的纯图表页
        blocks = [b for b in layout if isinstance(b, dict) and b.get("page_idx") == page - 1]
        assert all(not (b.get("text") or "").strip() for b in blocks), f"第 {page} 页有文本却未注标"
    assert _strip_markers(out) == md, "剥掉标记未能逐字节还原原文"
    for m in _MARKER_RE.finditer(out):  # 标记必须独占行首
        assert m.start() == 0 or out[m.start() - 1] == "\n", f"标记不在行首: 偏移 {m.start()}"
    print(f"      页数={len(pages)} 注入={len(marks)} 缺页={missing or '无'} 还原一致  通过")


def _check_normalize_integration(idx: int, label: str, parsed_dir: Path) -> None:
    print(f"[{idx}/{len(CASES)}] {label}: _normalize_into 集成")
    raw_root = parsed_dir / "_mineru_raw"
    source_md, assets = _locate_outputs(raw_root)
    assert source_md is not None, f"找不到 _mineru_raw 的 markdown: {raw_root}"

    stored = (parsed_dir / "paper.md").read_text(encoding="utf-8")
    out_dir = OUT_ROOT / parsed_dir.name
    if out_dir.exists():
        shutil.rmtree(out_dir)  # 只清理本 Demo 自己的上一轮产物
    out_dir.mkdir(parents=True)
    _normalize_into(out_dir, source_md, assets)
    normalized = (out_dir / "paper.md").read_text(encoding="utf-8")
    assert (out_dir / "layout.json").is_file(), "layout.json 未产出"

    n_marks = len(_MARKER_RE.findall(normalized))
    assert n_marks > 0, "标准化产物没有页码标记"
    assert _strip_markers(normalized) == stored, "标准化产物剥掉标记后与存量 paper.md 不一致"
    print(f"      注入标记={n_marks} 剥标后与存量产物逐字节一致  通过")
    print(f"      注标产物已保存: {out_dir.relative_to(REPO_ROOT)}/paper.md")


def main() -> None:
    for _, name in CASES:
        if not (REPO_ROOT / "demo-mineru-data/parsed" / name / "layout.json").is_file():
            print(f"缺少真实解析产物: {name}", file=sys.stderr)
            raise SystemExit(1)

    for i, (label, name) in enumerate(CASES, start=1):
        parsed_dir = REPO_ROOT / "demo-mineru-data/parsed" / name
        _check_pure_function(i, label, parsed_dir)
        _check_normalize_integration(i, label, parsed_dir)
        print()
    print("页码标记真实验收通过: 3 份真实 MinerU 产物, 纯函数 + 标准化集成双通道。")


if __name__ == "__main__":
    main()
