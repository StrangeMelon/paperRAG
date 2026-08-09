"""Wiki 后台 worker: 消费 wiki_jobs 持久化队列, 独立于 ingest 主链路运行。

Usage:
    python scripts/wiki_worker.py --once     # 领取一批处理后退出
    python scripts/wiki_worker.py --drain    # 循环直到队列排空(含退避重试)
    python scripts/wiki_worker.py --drain --concurrency 8
    python scripts/wiki_worker.py            # 默认 --drain

设计要点:
- ingest 只做幂等 INSERT, 概念抽取/词条更新的 LLM 成本全部在本进程,
  批量入库与 wiki 建设解耦, 可分晚错峰运行;
- 启动先 requeue_stale 归还崩溃残留的 processing 任务(断点续跑);
- 论文级受控并发: 同一批任务并发调用模型, 单篇内部仍保持概念状态更新顺序;
- 逐任务隔离: 单篇异常走 fail_job 退避重试, 超 max_attempts 记 failed;
  质量门槛拦下的文档由 triggers 报告 skipped, 如实记账不产词条;
- 每轮结束做 Qdrant 补偿同步: SQLite 是真相源, 镜像失败不回滚,
  qdrant_dirty 词条在此 embed + upsert, 失败留脏下轮再试;
- 退出码: 0 = 本轮无永久失败; 1 = 有 failed 任务。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_rag.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
log = get_logger("wiki_worker")


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drain the persistent wiki job queue.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Process one batch, then exit")
    mode.add_argument("--drain", action="store_true", help="Loop until the queue is empty")
    p.add_argument("--batch", type=int, default=0, help="Batch size (0 = config wiki.worker)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Concurrent papers (0 = config wiki.worker.concurrency)",
    )
    p.add_argument(
        "--stale-sec",
        type=int,
        default=3600,
        help="Requeue processing jobs older than this on start (crash recovery)",
    )
    return p.parse_args(argv)


def _process_job(job: dict) -> None:
    from paper_rag.wiki import queue as wq

    # triggers 动态解析, 便于测试注入; 生产路径等价于常规 import
    triggers = importlib.import_module("paper_rag.wiki.triggers")
    try:
        report = triggers.on_paper_indexed(job["paper_id"], language=job["language"])
    except Exception as e:
        wq.fail_job(job["job_id"], error=f"{type(e).__name__}: {e}")
        log.warning(f"wiki job {job['job_id']} ({job['paper_id']}) failed: {e}")
        return
    report = report or {}
    if report.get("skipped"):
        wq.mark_skipped(job["job_id"], reason=str(report["skipped"]))
        print(f"  {job['paper_id']} -> skipped ({report['skipped']})", flush=True)
    elif report.get("error"):
        wq.fail_job(job["job_id"], error=str(report["error"]))
        print(f"  {job['paper_id']} -> error ({report['error']})", flush=True)
    else:
        wq.complete_job(job["job_id"], report=report)
        print(
            f"  {job['paper_id']} -> done "
            f"(created={report.get('created', 0)} patched={report.get('patched', 0)} "
            f"review={report.get('review', 0)})",
            flush=True,
        )


def _process_jobs(jobs: list[dict], *, concurrency: int) -> None:
    """并发处理一批论文; _process_job 负责把每篇结果独立落回持久化队列。"""
    if concurrency <= 1 or len(jobs) <= 1:
        for job in jobs:
            _process_job(job)
        return
    with ThreadPoolExecutor(
        max_workers=min(concurrency, len(jobs)),
        thread_name_prefix="wiki-paper",
    ) as executor:
        list(executor.map(_process_job, jobs))


def _compensate_qdrant() -> int:
    """把 qdrant_dirty 的词条补偿镜像到 Qdrant。失败留脏, 下轮再试。"""
    from paper_rag.embed import bge_m3
    from paper_rag.wiki import store as wstore

    synced = 0
    for entry in wstore.pending_qdrant_entries():
        try:
            vec = bge_m3.encode_one(f"{entry.name}\n{entry.definition}")
            wstore.mirror_entry(entry, vec)
            synced += 1
        except Exception as e:
            log.warning(f"qdrant mirror still pending for {entry.entry_id}: {e}")
    return synced


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_dotenv(_REPO_ROOT / ".env")

    from paper_rag import config as cfg
    from paper_rag.wiki import queue as wq

    worker_cfg = cfg.load().wiki.worker
    concurrency = args.concurrency or worker_cfg.concurrency
    if not 1 <= concurrency <= 32:
        raise SystemExit("--concurrency must be between 1 and 32")

    requeued = wq.requeue_stale(older_than_sec=args.stale_sec)
    if requeued:
        print(f"requeued {requeued} stale processing jobs", flush=True)

    batch = args.batch or None
    rounds = 0
    t0 = time.perf_counter()
    while True:
        jobs = wq.claim_jobs(limit=batch)
        if not jobs:
            break
        rounds += 1
        print(
            f"[round {rounds}] processing {len(jobs)} jobs "
            f"(concurrency={min(concurrency, len(jobs))})",
            flush=True,
        )
        _process_jobs(jobs, concurrency=concurrency)
        if args.once:
            break

    try:
        synced = _compensate_qdrant()
        if synced:
            print(f"qdrant compensation: {synced} entries synced", flush=True)
    except Exception as e:
        log.warning(f"qdrant compensation round failed (non-fatal): {e}")

    stats = wq.stats()
    print(
        f"queue stats: {json.dumps(stats, ensure_ascii=False)} "
        f"({round(time.perf_counter() - t0, 1)}s)",
        flush=True,
    )
    return 1 if stats.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
