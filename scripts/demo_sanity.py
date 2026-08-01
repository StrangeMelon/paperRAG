"""章节完整性打分器真实验收: 用 build_chunks 真实产物的节名打分。

输入: demo-builder-data/parsed/<id>/chunks.json(需先跑 scripts/demo_builder.py),
只读不落盘。

验收点:
- 中文期刊在 zh 路由判 complete; 同一节名表在 en 路由(=基准的真实行为)只得
  minimal——这就是基准英文关键词表对真实中文论文的降级误判, 本课修复的动机;
- 两篇英文 MinerU 论文 en 路由判 complete; PyMuPDF 产物 None 路由(双表并集)
  判 complete;
- 顺带核对同批产物的参考文献打标(sanity 课决策点 a): 4 份产物均存在
  metadata["is_references"]=True 的块且节名为 References/参考文献,
  普通块不带该键。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from paper_rag.chunk.sanity import grade_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "demo-builder-data/parsed"

CASES = (
    # (标签, 产物目录, 语言路由, 期望打分, en 路由对照期望)
    (
        "中文期刊(MinerU, zh)",
        "sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru",
        "zh",
        "complete",
        "minimal",  # 基准英文表只命中 method 区(英文副标题里的 architecture/model)
    ),
    (
        "Graph-Mamba(MinerU, en)",
        "sha1_28acb520c921be7a1968207519dfa95d6af88800--mineru",
        "en",
        "complete",
        "complete",
    ),
    (
        "LocAgent(MinerU, en)",
        "sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932--mineru",
        "en",
        "complete",
        "complete",
    ),
    (
        "Graph-Mamba(PyMuPDF, None)",
        "sha1_28acb520c921be7a1968207519dfa95d6af88800--pymupdf",
        None,
        "complete",
        "complete",
    ),
)


def _check_references_flags(label: str, chunks: list[dict]) -> None:
    refs = [c for c in chunks if c["metadata"].get("is_references")]
    others = [c for c in chunks if not c["metadata"].get("is_references")]
    assert all("is_references" not in c["metadata"] for c in others), (
        f"{label}: 普通块混入 is_references 键"
    )
    assert refs, f"{label}: 未见任何参考文献打标块"
    assert all(c["section"] in ("References", "参考文献") for c in refs), (
        f"{label}: 打标块节名异常: {sorted({c['section'] for c in refs})}"
    )
    print(f"      参考文献打标块: {len(refs)}/{len(chunks)}")


def main() -> None:
    for _, dirname, *_ in CASES:
        if not (DATA_ROOT / dirname / "chunks.json").is_file():
            print(
                f"缺少输入: {dirname}/chunks.json(需先跑 scripts/demo_builder.py)", file=sys.stderr
            )
            raise SystemExit(1)

    for i, (label, dirname, language, expect, expect_en) in enumerate(CASES, start=1):
        payload = json.loads((DATA_ROOT / dirname / "chunks.json").read_text(encoding="utf-8"))
        names = [s["name"] for s in payload["sections"]]

        grade = grade_sections(names, language=language)
        grade_en = grade_sections(names, language="en")
        print(f"[{i}/{len(CASES)}] {label}")
        print(
            f"      sections={len(names)} 路由({language}) -> {grade} | en 路由对照 -> {grade_en}"
        )
        assert grade == expect, f"{label}: 打分 {grade} != {expect}"
        assert grade_en == expect_en, f"{label}: en 路由对照 {grade_en} != {expect_en}"

        # 顺带核对同批产物的参考文献打标(4 份真实产物都有 References/参考文献 节)
        _check_references_flags(label, payload["chunks"])
        print()

    print("章节完整性打分器真实验收通过: 中文期刊 zh 路由 complete(基准 en 路由仅 minimal)。")


if __name__ == "__main__":
    main()
