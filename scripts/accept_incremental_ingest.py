"""两篇真实 PDF 的 Qdrant 增量入库验收。

验收步骤:

1. 从 ``demo-ingest-batch-data/pdfs`` 复制两篇样本到隔离运行目录。
2. 新建 embedded Qdrant collection 和 SQLite。
3. 第一次强制入库, 验证完整流程和阶段计时。
4. 第二次强制重跑同一批论文, 验证不重新 embedding、不整篇删除。
5. 对比 Qdrant Point 数与 SQLite Chunk 数, 输出两轮详细计时。

脚本只清理自己的 ``demo-incremental-update-data`` 目录, 不执行现有生产库迁移。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "demo-ingest-batch-data"
SOURCE_PDFS = SOURCE_ROOT / "pdfs"
WORK_ROOT = REPO_ROOT / "demo-incremental-update-data"


def _write_config() -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        "data_root": str(WORK_ROOT / "data"),
        "papers_dir": str(WORK_ROOT / "data/papers"),
        "parsed_dir": str(WORK_ROOT / "data/parsed"),
        "index_dir": str(WORK_ROOT / "data/index"),
        "sqlite_path": str(WORK_ROOT / "data/index/papers.sqlite"),
        "bm25_path": str(WORK_ROOT / "data/index/bm25.pkl"),
        "models_dir": str(REPO_ROOT / "data/index/models"),
    }
    raw["qdrant"]["local_path"] = str(WORK_ROOT / "qdrant")
    raw["vision"]["enabled"] = False
    config_path = WORK_ROOT / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return config_path


def _run(config_path: Path, label: str, *args: str) -> str:
    env = {**os.environ, "PAPER_RAG_CONFIG": str(config_path)}
    process = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0:
        print(process.stdout[-4000:])
        print(process.stderr[-4000:], file=sys.stderr)
        raise AssertionError(f"{label} failed with exit code {process.returncode}")
    return process.stdout


def _print_report(label: str, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["failed"] == 0, f"{label}: report contains failed papers"

    print(f"\n[{label}]")
    for result in report["results"]:
        timings = result.get("timings", {})
        incremental = result.get("incremental", {})
        print(
            f"  {Path(result['file']).name}: status={result['status']} "
            f"chunks={result.get('chunks')} "
            f"parse={timings.get('parse_seconds', 0):.3f}s "
            f"chunk={timings.get('chunk_seconds', 0):.3f}s "
            f"vision={timings.get('vision_seconds', 0):.3f}s "
            f"embedding={timings.get('embedding_seconds', 0):.3f}s "
            f"incremental={timings.get('incremental_update_seconds', 0):.3f}s "
            f"total={timings.get('total_seconds', 0):.3f}s "
            f"vector_updates={incremental.get('vector_updates')} "
            f"payload_updates={incremental.get('payload_updates')} "
            f"skipped={incremental.get('skipped')} "
            f"deleted={incremental.get('deleted')}"
        )
    return report


def _verify_qdrant_matches_sqlite(config_path: Path, paper_ids: list[str]) -> None:
    os.environ["PAPER_RAG_CONFIG"] = str(config_path)
    from paper_rag import config as cfg
    from paper_rag.store import qdrant_store, sqlite_store

    cfg.load.cache_clear()
    qdrant_store.close_client()
    for paper_id in paper_ids:
        qdrant_count = len(qdrant_store.list_chunks_for_paper(paper_id))
        sqlite_count = len(sqlite_store.list_chunks_for_papers([paper_id]))
        assert qdrant_count == sqlite_count, (
            f"{paper_id}: Qdrant points={qdrant_count}, SQLite chunks={sqlite_count}"
        )
    qdrant_store.close_client()


def main() -> int:
    if not SOURCE_PDFS.is_dir():
        print(f"sample PDF directory not found: {SOURCE_PDFS}", file=sys.stderr)
        return 2
    pdfs = sorted(SOURCE_PDFS.glob("*.pdf"))
    if len(pdfs) < 2:
        print(f"expected two sample PDFs in {SOURCE_PDFS}", file=sys.stderr)
        return 2

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    trial_pdfs = WORK_ROOT / "pdfs"
    trial_pdfs.mkdir(parents=True)
    for pdf in pdfs[:2]:
        shutil.copy2(pdf, trial_pdfs / pdf.name)
    config_path = _write_config()

    _run(config_path, "init_store", "scripts/init_store.py")
    first_report_path = WORK_ROOT / "data/ingest_first.json"
    second_report_path = WORK_ROOT / "data/ingest_second.json"
    batch_args = ("scripts/ingest_batch.py", str(trial_pdfs), "--force")

    _run(config_path, "first ingest", *batch_args, "--report", str(first_report_path))
    first = _print_report("FIRST INGEST", first_report_path)
    assert first["summary"]["done"] == 2

    _run(config_path, "incremental reingest", *batch_args, "--report", str(second_report_path))
    second = _print_report("INCREMENTAL REINGEST", second_report_path)
    assert second["summary"]["done"] == 2
    for result in second["results"]:
        incremental = result["incremental"]
        assert incremental["vector_updates"] == 0
        assert incremental["payload_updates"] == 0
        assert incremental["deleted"] == 0
        assert incremental["skipped"] == result["chunks"]

    paper_ids = [result["paper_id"] for result in second["results"]]
    _verify_qdrant_matches_sqlite(config_path, paper_ids)
    print(
        "\nACCEPTANCE PASSED: two real PDFs, synchronous incremental Qdrant update, and counts verified"
    )
    print(f"reports: {first_report_path}, {second_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
