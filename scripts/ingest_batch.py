"""批量入库一个文件夹里的论文 PDF。

Usage:
    python scripts/ingest_batch.py /abs/path/to/pdf_folder            # 全量
    python scripts/ingest_batch.py /abs/path/to/pdf_folder --dry-run  # 只列清单
    python scripts/ingest_batch.py /abs/path/to/pdf_folder --limit 3  # 试水前 3 篇
    python scripts/ingest_batch.py /abs/path/to/pdf_folder --force    # 已入库的也重建

设计要点:
- 逐篇 try/except 隔离: 一篇解析/入库失败不中断整批, 计入 failed;
- 断点续跑: 引擎幂等(已 done 跳过 -> skipped), 中断后重跑同一命令即可续传;
- 标题暂取文件名去扩展名(元数据补全推迟, 见课程记账); language 由解析层
  自动判定(mineru.lang: auto);
- 逐篇结果落 JSON 报告(缺省 <data_root>/ingest_batch_report.json), 失败含
  错误信息, 便于事后针对性重试;
- 退出码: 0 = 无失败; 1 = 有失败; 2 = 参数/目录问题。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_rag import config as cfg
from paper_rag.utils.logger import get_logger
from paper_rag.utils.paths import ensure_dirs

_REPO_ROOT = Path(__file__).resolve().parents[1]
log = get_logger("ingest_batch")


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
    p = argparse.ArgumentParser(description="Batch-ingest all PDFs in a folder.")
    p.add_argument("folder", help="Folder containing PDF files (flat, no recursion)")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N PDFs (0 = all)")
    p.add_argument("--force", action="store_true", help="Re-ingest even if status=done")
    p.add_argument("--dry-run", action="store_true", help="List the PDFs that would be processed")
    p.add_argument(
        "--report",
        help="Path for the per-file JSON report (default: <data_root>/ingest_batch_report.json)",
    )
    return p.parse_args(argv)


def _scan_pdfs(folder: Path) -> list[Path]:
    return sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"),
        key=lambda f: f.name,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        log.error(f"folder not found: {folder}")
        return 2
    pdfs = _scan_pdfs(folder)
    if not pdfs:
        log.error(f"no PDF files in {folder}")
        return 2
    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    if args.dry_run:
        print(f"would ingest {len(pdfs)} PDFs from {folder}:")
        for f in pdfs:
            print(f"  {f.name}")
        return 0

    _load_dotenv(_REPO_ROOT / ".env")
    conf = cfg.load()
    ensure_dirs()
    report_path = (
        Path(args.report)
        if args.report
        else Path(conf.paths.data_root) / "ingest_batch_report.json"
    )

    from paper_rag.ingest.local_source import LocalSource
    from paper_rag.store.ingest_pipeline import ingest

    results: list[dict] = []
    counts = {"done": 0, "skipped": 0, "failed": 0}
    batch_start = time.perf_counter()
    for i, pdf in enumerate(pdfs, 1):
        t0 = time.perf_counter()
        entry: dict = {"file": str(pdf)}
        try:
            result = LocalSource(title=pdf.stem).fetch(str(pdf))
            out = ingest(result, force=args.force)
            status = out.get("status", "failed")
            entry.update(
                paper_id=out.get("paper_id"),
                status=status,
                chunks=out.get("chunks"),
                reason=out.get("reason"),
            )
        except Exception as e:  # 逐篇隔离: 单篇失败不中断整批
            status = "failed"
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            log.warning(f"ingest failed for {pdf.name}: {e}")
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        counts[status if status in counts else "failed"] += 1
        results.append(entry)
        print(
            f"[{i}/{len(pdfs)}] {pdf.name} -> {entry['status']}"
            + (f" ({entry.get('chunks')} chunks)" if entry.get("chunks") else "")
            + (f" [{entry.get('reason')}]" if entry.get("reason") else "")
            + f" {entry['seconds']}s",
            flush=True,
        )

    total_s = round(time.perf_counter() - batch_start, 1)
    summary = {**counts, "total": len(pdfs), "seconds": total_s}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nBATCH SUMMARY: done={counts['done']} skipped={counts['skipped']} "
        f"failed={counts['failed']} total={len(pdfs)} in {total_s}s"
    )
    print(f"report: {report_path}")
    if counts["failed"]:
        failed = [r["file"] for r in results if r["status"] == "failed"]
        print(f"failed files ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
