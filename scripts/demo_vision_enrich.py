"""视觉增强真实验收: 对比同一批 chunk 在 enrich 前后的 context_text。

数据来自已验收的解析层真实产物(只读, 不写任何文件):
- 中文期刊(zh, 19 图): demo-mineru-data/parsed/sha1_ab3d...566
- LocAgent(en, 16 图):  demo-mineru-data/parsed/sha1_a3e2...932

做法: 同一份 chunk 列表先深拷贝存快照, 再真实调用 vision 增强, 逐块 diff
context_text。context_text 是送进 BGE-M3 与 FTS5/BM25 的那一份文本, 所以
"增强前 vs 增强后"的差异就是这一课对检索侧的全部影响。

验收点:
- zh 论文: 追加行是"视觉摘要:", 描述性文字为中文, 模型/数据集/指标保留原文形态;
- en 论文: 追加行是 "Visual summary:", 与基准行为一致;
- 每张图的 metadata 记账 status/provider/model, 失败图不改 context_text(fail-open);
- 第二次运行全部命中缓存(status=cached), 不重复计费。

用法(需 .env 里备好 VISION_BASE_URL / VISION_API_KEY / VISION_MODEL):
    uv run python scripts/demo_vision_enrich.py              # zh + en, 各 3 张图
    uv run python scripts/demo_vision_enrich.py --limit 5    # 每篇多看几张
    uv run python scripts/demo_vision_enrich.py --lang zh    # 只跑中文
    uv run python scripts/demo_vision_enrich.py --no-cache   # 强制真调, 不读缓存
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

from paper_rag.chunk import builder
from paper_rag.config import load
from paper_rag.vision.enrich import enrich_chunks
from paper_rag.vision.schema import STATUS_CACHED, STATUS_FALLBACK, STATUS_OK

REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    (
        "zh",
        "中文期刊《综合能源服务区块链》",
        "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566",
    ),
    (
        "en",
        "LocAgent (英文会议论文)",
        "demo-mineru-data/parsed/sha1_a3e2e21da0bdde69e3bc5feda948db5d4c02e932",
    ),
)

SUCCESS = (STATUS_OK, STATUS_CACHED, STATUS_FALLBACK)


def _load_dotenv() -> None:
    """与 scripts/ask.py 同款: 让脚本直接可用, 不要求用户先 export。"""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _preflight() -> None:
    cfg = load().vision
    missing = [
        name
        for name, value in (
            ("VISION_BASE_URL", cfg.base_url),
            ("VISION_API_KEY", cfg.api_key),
            ("VISION_MODEL", cfg.model),
        )
        if not value
    ]
    if missing:
        print(f"缺少环境变量 {', '.join(missing)}; 请在 .env 中填好后重跑。")
        print("智谱示例: VISION_BASE_URL=https://open.bigmodel.cn/api/paas/v4/")
        print("          VISION_MODEL=glm-4.6v")
        sys.exit(1)
    print(f"视觉模型: {cfg.model}  端点: {cfg.base_url}")
    print(f"temperature={cfg.temperature}  extra_body={cfg.extra_body or '{}'}")
    if not cfg.enabled:
        print(
            "(vision.enabled=false: 生产 ingest 不会自动增强; Demo 显式注入 summarizer, 不受影响)"
        )


def _visual_chunks(parsed_dir: Path, paper_id: str, title: str, limit: int) -> tuple[list, str]:
    """构建 chunk 并取前 limit 个带真实图片的 figure/table 块。"""
    language = builder.read_language(parsed_dir)
    _, chunks = builder.build_chunks(paper_id, parsed_dir, title=title)
    picked = [
        c
        for c in chunks
        if c.get("modality") in ("figure", "table")
        and c.get("asset_path")
        and Path(str(c["asset_path"])).exists()
    ][:limit]
    return picked, language or "None"


def _show_diff(before: dict, after: dict, index: int) -> None:
    meta = after.get("metadata") or {}
    status = meta.get("visual_summary_status", "(未记账)")
    print(f"\n  [{index}] chunk {after['chunk_id']}  modality={after['modality']}")
    print(f"      图片: {Path(str(after['asset_path'])).name}")
    print(f"      status={status}  provider={meta.get('visual_summary_provider', '-')}")
    if meta.get("visual_summary_error"):
        print(f"      error={meta['visual_summary_error'][:160]}")

    old_text, new_text = str(before.get("context_text") or ""), str(after.get("context_text") or "")
    print(
        f"      context_text 长度: {len(old_text)} → {len(new_text)}  (+{len(new_text) - len(old_text)})"
    )
    if new_text == old_text:
        print("      —— 无变化(fail-open: 原文逐字保留)")
        return
    added = new_text[len(old_text) :].strip() if new_text.startswith(old_text) else new_text
    print("      —— 新增内容 ——")
    for line in added.splitlines():
        if line.strip():
            print(f"      | {line}")


def _build_summarizer():
    """显式构造 API summarizer 注入 enrich_chunks。

    生产 ingest 由 vision.enabled 决定是否增强(默认 false); Demo 是显式验收
    动作, 不受该开关约束——若依赖缺省构造, enabled=false 会静默早退,
    连 metadata 记账都不会发生。
    """
    cfg = load().vision
    from paper_rag.vision.api import OpenAIVisionSummarizer

    return OpenAIVisionSummarizer(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        extra_body=cfg.extra_body,
    ).summarize


def run_case(language_tag: str, title: str, rel_dir: str, *, limit: int, use_cache: bool) -> dict:
    parsed_dir = REPO_ROOT / rel_dir
    print("\n" + "=" * 78)
    print(f"{title}   [{rel_dir}]")
    print("=" * 78)
    if not parsed_dir.is_dir():
        print(f"跳过: 解析产物不存在 {parsed_dir}")
        return {"total": 0, "ok": 0}

    paper_id = f"demo-vision-{language_tag}"
    chunks, detected = _visual_chunks(parsed_dir, paper_id, title, limit)
    print(f"language.json 判定: {detected}   本次送检图片: {len(chunks)} 张")
    if not chunks:
        print("跳过: 该产物没有带图片资产的 figure/table chunk")
        return {"total": 0, "ok": 0}

    before = copy.deepcopy(chunks)
    enriched = enrich_chunks(
        paper_id,
        chunks,
        summarizer=_build_summarizer(),
        cache_enabled=use_cache,
        language=builder.read_language(parsed_dir),
    )
    for i, (old, new) in enumerate(zip(before, enriched, strict=True), start=1):
        _show_diff(old, new, i)

    ok = sum(
        1 for c in enriched if (c.get("metadata") or {}).get("visual_summary_status") in SUCCESS
    )
    print(f"\n  小结: {ok}/{len(enriched)} 张成功增强")
    return {"total": len(enriched), "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉增强前后 context_text 对比")
    ap.add_argument("--limit", type=int, default=3, help="每篇论文送检的图片数(默认 3)")
    ap.add_argument("--lang", choices=("zh", "en", "both"), default="both", help="只跑某一语言")
    ap.add_argument("--no-cache", action="store_true", help="不读写缓存, 强制真实调用")
    args = ap.parse_args()

    _load_dotenv()
    _preflight()

    cases = [c for c in CASES if args.lang in ("both", c[0])]
    totals = {"total": 0, "ok": 0}
    for language_tag, title, rel_dir in cases:
        stats = run_case(
            language_tag, title, rel_dir, limit=args.limit, use_cache=not args.no_cache
        )
        totals["total"] += stats["total"]
        totals["ok"] += stats["ok"]

    print("\n" + "=" * 78)
    print(f"总计: {totals['ok']}/{totals['total']} 张图片成功增强")
    if totals["total"] and totals["ok"] == totals["total"]:
        print("提示: 再跑一次同样命令, status 应全部变成 cached(不再计费)。")
    print("=" * 78)
    return 0 if totals["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
