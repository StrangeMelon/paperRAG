"""视觉增强并发真实验收: context_text 与顺序双重对比。

数据来自已验收的解析层真实产物(只读, 不写任何文件):
- 中文期刊(zh, 19 图): demo-mineru-data/parsed/sha1_ab3d...566
- LocAgent(en, 16 图):  demo-mineru-data/parsed/sha1_a3e2...932

做法: 用独立临时缓存强制真实 API 请求, 记录 API 完成顺序与缓存
写入顺序。同一份 chunk 列表先深拷贝存快照, 增强后逐块输出完整
context_text 前后对比。

验收点:
- zh 论文: 追加行是"视觉摘要:", 描述性文字为中文, 模型/数据集/指标保留原文形态;
- en 论文: 追加行是 "Visual summary:", 与基准行为一致;
- 每张图的 metadata 记账 status/provider/model, 失败图不改 context_text(fail-open);
- 请求至少两路真实重叠, 证明不是伪并发;
- 增强前 chunk 顺序 == 增强后 chunk 顺序 == 缓存写入顺序;
- 同一进程第二轮全部命中缓存(status=cached), 不重复计费。

用法(需 .env 里备好 VISION_BASE_URL / VISION_API_KEY / VISION_MODEL):
    uv run python scripts/demo_vision_enrich.py                  # zh + en, 各 3 张图
    uv run python scripts/demo_vision_enrich.py --limit 5        # 每篇多看几张
    uv run python scripts/demo_vision_enrich.py --lang zh        # 只跑中文
    uv run python scripts/demo_vision_enrich.py --concurrency 4  # 显式验收四路并发
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from unittest.mock import patch

from paper_rag.chunk import builder
from paper_rag.config import load
from paper_rag.vision.cache import VisionSummaryCache
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


class _TrackedSummarizer:
    """记录真实 API 完成顺序和峰值并发数。"""

    def __init__(self, summarize):
        self.summarize = summarize
        self.completion_order: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = Lock()

    def __call__(self, request):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return self.summarize(request)
        finally:
            with self._lock:
                self.active -= 1
                self.completion_order.append(request.chunk_id)


class _TrackedCache(VisionSummaryCache):
    """记录主线程真实写缓存的 chunk 顺序。"""

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self._chunk_by_key: dict[str, str] = {}
        self.write_order: list[str] = []

    def key_for(self, request):
        key = super().key_for(request)
        self._chunk_by_key[key] = request.chunk_id
        return key

    def write(self, key, result):
        self.write_order.append(self._chunk_by_key[key])
        super().write(key, result)


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
    print(
        f"temperature={cfg.temperature}  max_concurrency={cfg.max_concurrency}  "
        f"extra_body={cfg.extra_body or '{}'}"
    )
    if not cfg.enabled:
        print(
            "(vision.enabled=false: 生产 ingest 不会自动增强; Demo 显式注入 summarizer, 不受影响)"
        )


def _visual_chunks(parsed_dir: Path, paper_id: str, title: str, limit: int) -> tuple[list, str]:
    """构建 chunk 并取前 limit 个带真实图片的 figure/table 块。"""
    language = builder.read_language(parsed_dir)
    # 样本构建时不能触发 builder 的生产 Vision 钩子, 否则会先请求整篇全部图片。
    with patch("paper_rag.vision.enrich.enrich_chunks", side_effect=lambda _, chunks, **__: chunks):
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
    print(f"      context_text 长度: {len(old_text)} → {len(new_text)}")
    print("      --- BEFORE context_text ---")
    for line in old_text.splitlines() or [""]:
        print(f"      < {line}")
    print("      --- AFTER context_text ---")
    for line in new_text.splitlines() or [""]:
        print(f"      > {line}")
    if new_text == old_text:
        print("      —— 无变化(fail-open: 原文逐字保留)")


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


def _show_order_comparison(
    before_order: list[str],
    completion_order: list[str],
    after_order: list[str],
    cache_write_order: list[str],
) -> None:
    print("\n  顺序对比:")
    print(f"    API 完成顺序: {completion_order}")
    print(f"    增强前顺序:   {before_order}")
    print(f"    增强后顺序:   {after_order}")
    print(f"    缓存写入顺序: {cache_write_order}")
    for index, (before_id, after_id) in enumerate(
        zip(before_order, after_order, strict=True), start=1
    ):
        print(f"    [{index}] {before_id} == {after_id}")


def run_case(
    language_tag: str,
    title: str,
    rel_dir: str,
    *,
    limit: int,
    concurrency: int,
) -> dict:
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
    before_order = [c["chunk_id"] for c in before]
    tracked_summarizer = _TrackedSummarizer(_build_summarizer())
    with TemporaryDirectory(prefix="paper-rag-vision-acceptance-") as cache_dir:
        cache = _TrackedCache(cache_dir)
        enriched = enrich_chunks(
            paper_id,
            chunks,
            summarizer=tracked_summarizer,
            cache=cache,
            cache_enabled=True,
            language=builder.read_language(parsed_dir),
            max_concurrency=concurrency,
        )
        after_order = [c["chunk_id"] for c in enriched]
        _show_order_comparison(
            before_order,
            tracked_summarizer.completion_order,
            after_order,
            cache.write_order,
        )

        assert before_order == after_order, "并发增强后 chunk 顺序发生变化"
        assert before_order == cache.write_order, "缓存未按原 chunk 顺序写入"
        if len(chunks) > 1 and concurrency > 1:
            assert tracked_summarizer.max_active > 1, "真实 API 请求没有形成并发重叠"

        # 同进程复跑: 必须全部命中上一轮按序写入的缓存, 且不再调 API。
        second = copy.deepcopy(before)
        calls_before = len(tracked_summarizer.completion_order)
        enrich_chunks(
            paper_id,
            second,
            summarizer=tracked_summarizer,
            cache=cache,
            cache_enabled=True,
            language=builder.read_language(parsed_dir),
            max_concurrency=concurrency,
        )
        assert len(tracked_summarizer.completion_order) == calls_before, "缓存命中后仍调用了 API"
        assert all(c["metadata"]["visual_summary_status"] == STATUS_CACHED for c in second)

    for i, (old, new) in enumerate(zip(before, enriched, strict=True), start=1):
        _show_diff(old, new, i)

    ok = sum(
        1 for c in enriched if (c.get("metadata") or {}).get("visual_summary_status") in SUCCESS
    )
    print(
        f"\n  小结: {ok}/{len(enriched)} 张成功增强; "
        f"峰值并发={tracked_summarizer.max_active}; 原序提交=通过; 二轮缓存=通过"
    )
    return {"total": len(enriched), "ok": ok, "order_ok": True}


def main() -> int:
    ap = argparse.ArgumentParser(description="并发视觉增强的 context_text 与顺序真实验收")
    ap.add_argument("--limit", type=int, default=3, help="每篇论文送检的图片数(默认 3)")
    ap.add_argument("--lang", choices=("zh", "en", "both"), default="both", help="只跑某一语言")
    ap.add_argument("--concurrency", type=int, default=0, help="并发数(0 = vision.max_concurrency)")
    args = ap.parse_args()

    _load_dotenv()
    _preflight()

    cases = [c for c in CASES if args.lang in ("both", c[0])]
    concurrency = args.concurrency or load().vision.max_concurrency
    totals = {"total": 0, "ok": 0, "order_ok": True}
    for language_tag, title, rel_dir in cases:
        stats = run_case(
            language_tag,
            title,
            rel_dir,
            limit=args.limit,
            concurrency=concurrency,
        )
        totals["total"] += stats["total"]
        totals["ok"] += stats["ok"]
        totals["order_ok"] = totals["order_ok"] and stats.get("order_ok", False)

    print("\n" + "=" * 78)
    print(
        f"总计: {totals['ok']}/{totals['total']} 张图片成功增强; "
        f"顺序验收={'通过' if totals['order_ok'] else '失败'}"
    )
    print("=" * 78)
    passed = totals["total"] > 0 and totals["ok"] == totals["total"] and totals["order_ok"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
