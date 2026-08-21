"""强制 MinerU 与 Vision 的真实增量入库验收。

使用 ``demo-ingest-batch-data/pdfs`` 下两篇真实 PDF, 在隔离存储中执行首轮
完整入库和第二轮强制增量重跑。任一论文降级到 PyMuPDF、Vision 没有成功
处理真实图表、阶段计时缺失或第二轮重新生成向量, 脚本都会失败。
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import accept_incremental_ingest as base  # noqa: E402

SOURCE_PDFS = REPO_ROOT / "demo-ingest-batch-data/pdfs"
WORK_ROOT = REPO_ROOT / "demo-full-incremental-update-data"
_TIMING_FIELDS = (
    "parse_seconds",
    "chunk_seconds",
    "vision_seconds",
    "embedding_seconds",
    "incremental_update_seconds",
    "total_seconds",
)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def _write_config(work_root: Path = WORK_ROOT) -> Path:
    """写入强制 MinerU、开启 Vision 的隔离验收配置。"""
    raw = yaml.safe_load((REPO_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        "data_root": str(work_root / "data"),
        "papers_dir": str(work_root / "data/papers"),
        "parsed_dir": str(work_root / "data/parsed"),
        "index_dir": str(work_root / "data/index"),
        "sqlite_path": str(work_root / "data/index/papers.sqlite"),
        "bm25_path": str(work_root / "data/index/bm25.pkl"),
        "models_dir": str(REPO_ROOT / "data/index/models"),
    }
    raw["qdrant"]["local_path"] = str(work_root / "qdrant")
    raw["mineru"] = {
        "mode": "local",
        "cli": str(REPO_ROOT / ".venv/bin/magic-pdf"),
        "method": "ocr",
        "lang": "auto",
        "timeout_sec": 600,
        "fallback_to_pymupdf": False,
    }
    raw["vision"]["enabled"] = True
    raw["vision"]["cache"] = True
    raw["vision"]["cache_dir"] = str(work_root / "data/index/vision_cache")
    config_path = work_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return config_path


def _validate_timing_report(report: dict[str, Any]) -> None:
    assert report["summary"]["failed"] == 0
    for result in report["results"]:
        timings = result.get("timings") or {}
        for field in _TIMING_FIELDS:
            assert field in timings, f"missing timing field: {field}"
            assert float(timings[field]) >= 0, f"negative timing field: {field}"
        assert timings["parse_seconds"] > 0
        assert timings["total_seconds"] >= timings["incremental_update_seconds"]


def _capability_summary(work_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    database = work_root / "data/index/papers.sqlite"
    with sqlite3.connect(database) as connection:
        parsers = dict(connection.execute("SELECT paper_id, parsed_with FROM paper").fetchall())
        rows = connection.execute(
            "SELECT metadata_json FROM chunk "
            "WHERE modality IN ('figure', 'table') AND asset_path IS NOT NULL"
        ).fetchall()

    statuses: list[str] = []
    for (metadata_json,) in rows:
        metadata = json.loads(metadata_json or "{}")
        statuses.append(str(metadata.get("visual_summary_status") or "missing"))

    return {
        "parsers": {
            result["paper_id"]: str(parsers.get(result["paper_id"]) or "")
            for result in report["results"]
        },
        "visual_chunks": len(statuses),
        "vision_ok": statuses.count("ok"),
        "vision_cached": statuses.count("cached"),
        "vision_failed": statuses.count("failed"),
        "vision_other": sum(status not in {"ok", "cached", "failed"} for status in statuses),
    }


def _validate_capability_summary(
    report: dict[str, Any],
    capability: dict[str, Any],
    *,
    incremental: bool,
) -> None:
    _validate_timing_report(report)
    for paper_id, parser in capability["parsers"].items():
        assert parser.startswith("mineru+"), f"MinerU required for {paper_id}, got {parser!r}"

    assert capability["visual_chunks"] > 0, "Vision found no figure/table assets"
    assert capability["vision_failed"] == 0, "Vision contains failed summaries"
    assert capability.get("vision_other", 0) == 0, "Vision contains skipped/unavailable summaries"
    assert capability["vision_ok"] + capability["vision_cached"] == capability["visual_chunks"]

    if incremental:
        assert capability["vision_cached"] > 0, "Vision cache was not exercised"
        for result in report["results"]:
            update = result["incremental"]
            classified = update["vector_updates"] + update["payload_updates"] + update["skipped"]
            assert classified == result["chunks"]
            assert update["skipped"] > 0, "incremental run skipped no chunks"
            assert update["vector_updates"] < result["chunks"], "unexpected full vector rewrite"
    else:
        assert capability["vision_ok"] > 0, "Vision API produced no successful summary"
        assert sum(result["incremental"]["vector_updates"] for result in report["results"]) > 0


def _print_capabilities(label: str, capability: dict[str, Any]) -> None:
    print(
        f"  {label}: parsers={capability['parsers']} "
        f"visual_chunks={capability['visual_chunks']} "
        f"vision_ok={capability['vision_ok']} "
        f"vision_cached={capability['vision_cached']} "
        f"vision_failed={capability['vision_failed']} "
        f"vision_other={capability['vision_other']}"
    )


def _require_runtime() -> None:
    missing = [
        name
        for name in ("VISION_BASE_URL", "VISION_API_KEY", "VISION_MODEL")
        if not os.environ.get(name)
    ]
    assert not missing, f"missing Vision environment variables: {', '.join(missing)}"
    assert (REPO_ROOT / ".venv/bin/magic-pdf").is_file(), "MinerU CLI is missing"


def main() -> int:
    _load_dotenv(REPO_ROOT / ".env")
    _require_runtime()
    pdfs = sorted(SOURCE_PDFS.glob("*.pdf"))
    assert len(pdfs) >= 2, f"expected two real PDFs in {SOURCE_PDFS}"

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    trial_pdfs = WORK_ROOT / "pdfs"
    trial_pdfs.mkdir(parents=True)
    for pdf in pdfs[:2]:
        shutil.copy2(pdf, trial_pdfs / pdf.name)
    config_path = _write_config(WORK_ROOT)

    base._run(config_path, "init_store", "scripts/init_store.py")
    first_path = WORK_ROOT / "data/ingest_full_first.json"
    second_path = WORK_ROOT / "data/ingest_full_second.json"
    batch = ("scripts/ingest_batch.py", str(trial_pdfs), "--force")

    base._run(config_path, "full first ingest", *batch, "--report", str(first_path))
    first = base._print_report("FULL FIRST INGEST", first_path)
    first_capability = _capability_summary(WORK_ROOT, first)
    _validate_capability_summary(first, first_capability, incremental=False)
    _print_capabilities("first", first_capability)

    base._run(config_path, "full incremental ingest", *batch, "--report", str(second_path))
    second = base._print_report("FULL INCREMENTAL REINGEST", second_path)
    second_capability = _capability_summary(WORK_ROOT, second)
    _validate_capability_summary(second, second_capability, incremental=True)
    _print_capabilities("incremental", second_capability)

    base._verify_qdrant_matches_sqlite(
        config_path,
        [result["paper_id"] for result in second["results"]],
    )
    print(
        "\nFULL ACCEPTANCE PASSED: MinerU, Vision, embedding, synchronous incremental "
        "Qdrant update, timings, and store counts verified"
    )
    print(f"reports: {first_path}, {second_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
