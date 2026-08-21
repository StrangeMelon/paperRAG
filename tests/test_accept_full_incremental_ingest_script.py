"""全能力增量入库验收脚本的配置与报告契约。"""

from __future__ import annotations

import importlib

import pytest
import yaml


def _module():
    return importlib.import_module("scripts.accept_full_incremental_ingest")


def _timed_result(**overrides) -> dict:
    result = {
        "paper_id": "paper-1",
        "status": "done",
        "chunks": 4,
        "incremental": {
            "vector_updates": 4,
            "payload_updates": 0,
            "skipped": 0,
            "deleted": 0,
        },
        "timings": {
            "parse_seconds": 1.0,
            "chunk_seconds": 2.0,
            "vision_seconds": 3.0,
            "embedding_seconds": 4.0,
            "incremental_update_seconds": 5.0,
            "total_seconds": 8.0,
        },
    }
    result.update(overrides)
    return result


def test_config_forces_mineru_and_enables_vision(tmp_path) -> None:
    module = _module()

    config_path = module._write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["mineru"] == {
        "mode": "local",
        "cli": str(module.REPO_ROOT / ".venv/bin/magic-pdf"),
        "method": "ocr",
        "lang": "auto",
        "timeout_sec": 600,
        "fallback_to_pymupdf": False,
    }
    assert config["vision"]["enabled"] is True
    assert config["vision"]["cache"] is True
    assert config["vision"]["cache_dir"] == str(tmp_path / "data/index/vision_cache")


def test_timing_validation_requires_every_stage() -> None:
    module = _module()
    report = {"summary": {"failed": 0}, "results": [_timed_result()]}

    module._validate_timing_report(report)

    del report["results"][0]["timings"]["vision_seconds"]
    with pytest.raises(AssertionError, match="vision_seconds"):
        module._validate_timing_report(report)


def test_capability_validation_rejects_fallback_or_failed_vision() -> None:
    module = _module()
    report = {"summary": {"failed": 0}, "results": [_timed_result()]}

    module._validate_capability_summary(
        report,
        {
            "parsers": {"paper-1": "mineru+complete"},
            "visual_chunks": 2,
            "vision_ok": 2,
            "vision_cached": 0,
            "vision_failed": 0,
        },
        incremental=False,
    )

    with pytest.raises(AssertionError, match="MinerU"):
        module._validate_capability_summary(
            report,
            {
                "parsers": {"paper-1": "pymupdf+complete"},
                "visual_chunks": 2,
                "vision_ok": 2,
                "vision_cached": 0,
                "vision_failed": 0,
            },
            incremental=False,
        )

    with pytest.raises(AssertionError, match="Vision"):
        module._validate_capability_summary(
            report,
            {
                "parsers": {"paper-1": "mineru+complete"},
                "visual_chunks": 2,
                "vision_ok": 0,
                "vision_cached": 0,
                "vision_failed": 2,
            },
            incremental=False,
        )


def test_incremental_validation_accepts_partial_upstream_changes() -> None:
    module = _module()
    report = {
        "summary": {"failed": 0},
        "results": [
            _timed_result(
                incremental={
                    "vector_updates": 1,
                    "payload_updates": 1,
                    "skipped": 2,
                    "deleted": 0,
                }
            )
        ],
    }
    capability = {
        "parsers": {"paper-1": "mineru+complete"},
        "visual_chunks": 2,
        "vision_ok": 0,
        "vision_cached": 2,
        "vision_failed": 0,
    }

    module._validate_capability_summary(report, capability, incremental=True)
