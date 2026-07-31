from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "mineru_doctor.py"


def _load_script() -> ModuleType:
    assert SCRIPT_PATH.is_file(), "请先创建 scripts/mineru_doctor.py"

    spec = importlib.util.spec_from_file_location("mineru_doctor_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeReport:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return self._payload


def _doctor_payload(*, ok: bool) -> dict[str, object]:
    return {
        "ok": ok,
        "cli_path": "/project/.venv/bin/magic-pdf",
        "config_path": "/project/config/magic-pdf.json",
        "checks": [
            {
                "name": "model weight layout:doclayout_yolo",
                "ok": ok,
                "detail": "/project/data/index/mineru_models/Layout/YOLO/model.pt",
                "hint": "下载布局模型权重。" if not ok else "",
            }
        ],
    }


def test_json_mode_prints_complete_doctor_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    payload = _doctor_payload(ok=True)
    monkeypatch.setattr(module.mineru_local, "diagnose", lambda: _FakeReport(payload))
    monkeypatch.setattr(sys, "argv", ["mineru_doctor.py", "--json"])

    exit_code = module.main()

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_human_mode_marks_failed_checks_and_prints_hints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    payload = _doctor_payload(ok=False)
    monkeypatch.setattr(module.mineru_local, "diagnose", lambda: _FakeReport(payload))
    monkeypatch.setattr(sys, "argv", ["mineru_doctor.py"])

    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "MinerU doctor" in output
    assert "[FAIL] model weight layout:doclayout_yolo" in output
    assert "hint: 下载布局模型权重。" in output


def test_strict_mode_returns_nonzero_when_diagnosis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module.mineru_local,
        "diagnose",
        lambda: _FakeReport(_doctor_payload(ok=False)),
    )
    monkeypatch.setattr(sys, "argv", ["mineru_doctor.py", "--strict"])

    assert module.main() == 1


def test_try_parse_adds_success_result_and_forwards_paper_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_pdf = tmp_path / "paper.pdf"
    output_dir = tmp_path / "parsed"
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        module.mineru_local,
        "diagnose",
        lambda: _FakeReport(_doctor_payload(ok=True)),
    )

    def fake_parse_pdf(paper_id: str, pdf_path: str) -> Path:
        calls.append((paper_id, pdf_path))
        return output_dir

    monkeypatch.setattr(module.mineru_local, "parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mineru_doctor.py",
            "--json",
            "--strict",
            "--try-parse",
            str(source_pdf),
            "--paper-id",
            "arxiv:1706.03762",
        ],
    )

    exit_code = module.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == [("arxiv:1706.03762", str(source_pdf))]
    assert payload["try_parse"] == {
        "ok": True,
        "out_dir": str(output_dir),
    }


def test_try_parse_classifies_mineru_failure_and_fails_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_pdf = tmp_path / "paper.pdf"

    monkeypatch.setattr(
        module.mineru_local,
        "diagnose",
        lambda: _FakeReport(_doctor_payload(ok=True)),
    )

    def fail_parse_pdf(_paper_id: str, _pdf_path: str) -> Path:
        raise module.mineru_local.MineruError("CUDA out of memory")

    monkeypatch.setattr(module.mineru_local, "parse_pdf", fail_parse_pdf)
    monkeypatch.setattr(
        module.mineru_local,
        "classify_failure",
        lambda _message: ("gpu_oom", "减少并发或换用更小模型。"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mineru_doctor.py",
            "--json",
            "--strict",
            "--try-parse",
            str(source_pdf),
        ],
    )

    exit_code = module.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["try_parse"] == {
        "ok": False,
        "reason": "gpu_oom",
        "error": "CUDA out of memory",
        "hint": "减少并发或换用更小模型。",
    }


def test_human_output_prints_successful_try_parse_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    payload = _doctor_payload(ok=True)
    payload["try_parse"] = {
        "ok": True,
        "out_dir": "/project/data/parsed/mineru_doctor",
    }

    module._print_human(payload)
    output = capsys.readouterr().out

    assert "Try parse:" in output
    assert "ok: True" in output
    assert "out_dir: /project/data/parsed/mineru_doctor" in output


def test_human_output_prints_failed_try_parse_reason_and_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    payload = _doctor_payload(ok=True)
    payload["try_parse"] = {
        "ok": False,
        "reason": "gpu_oom",
        "error": "CUDA out of memory",
        "hint": "减少并发或换用更小模型。",
    }

    module._print_human(payload)
    output = capsys.readouterr().out

    assert "Try parse:" in output
    assert "ok: False" in output
    assert "reason: gpu_oom" in output
    assert "hint: 减少并发或换用更小模型。" in output
