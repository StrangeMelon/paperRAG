"""MinerU 本地解析器的分阶段边界测试。"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _mineru_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.parse.mineru_local")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.parse.mineru_local":
            raise
        pytest.fail(
            "尚未实现 paper_rag.parse.mineru_local",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _assume_ocr_weights_for_boundary_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认假设 OCR 权重齐备, 使不涉及模型文件的边界测试不读取真实权重目录。

    需要验证降级行为的测试会在用例内再次覆盖 ``_ocr_weights_available``。
    """

    module = _mineru_module()
    monkeypatch.setattr(
        module,
        "_ocr_weights_available",
        lambda *_args: True,
        raising=False,
    )


def test_runtime_environment_creates_isolated_cache_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    index_dir = tmp_path / "index"
    custom_mpl = tmp_path / "custom-matplotlib"
    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: SimpleNamespace(
            paths=SimpleNamespace(index_dir=str(index_dir))
        ),
    )
    environment = {"MPLCONFIGDIR": str(custom_mpl)}

    result = module._ensure_runtime_env(environment)

    assert result is environment
    assert result == {
        "MPLCONFIGDIR": str(custom_mpl),
        "YOLO_CONFIG_DIR": str(
            index_dir / "runtime_cache" / "ultralytics"
        ),
        "XDG_CACHE_HOME": str(index_dir / "runtime_cache" / "xdg"),
    }
    assert custom_mpl.is_dir()
    assert Path(result["YOLO_CONFIG_DIR"]).is_dir()
    assert Path(result["XDG_CACHE_HOME"]).is_dir()


def test_cli_resolution_prefers_executable_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _mineru_module()
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/opt/mineru/bin/mineru" if name == "mineru" else None,
    )

    assert module._resolve_cli("mineru") == "/opt/mineru/bin/mineru"


@pytest.mark.parametrize(
    ("detail", "expected_reason", "hint_fragment"),
    [
        (
            "ModuleNotFoundError: No module named 'cv2'",
            "missing_cv2",
            ".[mineru]",
        ),
        (
            "ModuleNotFoundError: No module named 'ultralytics'",
            "missing_mineru_full_extra",
            ".[mineru]",
        ),
        (
            "https://huggingface.co failed to resolve",
            "missing_models_or_offline",
            "network",
        ),
        (
            "layout model checkpoint not found",
            "missing_models",
            "models-dir",
        ),
        (
            "an unexpected parser failure",
            "unknown",
            "",
        ),
    ],
)
def test_failure_classifier_returns_actionable_reason(
    detail: str,
    expected_reason: str,
    hint_fragment: str,
) -> None:
    module = _mineru_module()

    reason, hint = module.classify_failure(detail)

    assert reason == expected_reason
    assert hint_fragment.lower() in hint.lower()


def test_doctor_report_serializes_nested_checks() -> None:
    module = _mineru_module()
    report = module.MineruDoctorReport(
        ok=False,
        cli_path=None,
        config_path="/tmp/magic-pdf.json",
        checks=[
            module.MineruCheck(
                name="cv2",
                ok=False,
                detail="missing",
                hint="install dependencies",
            )
        ],
    )

    assert report.to_dict() == {
        "ok": False,
        "cli_path": None,
        "config_path": "/tmp/magic-pdf.json",
        "checks": [
            {
                "name": "cv2",
                "ok": False,
                "detail": "missing",
                "hint": "install dependencies",
            }
        ],
    }


def test_locate_outputs_selects_largest_markdown_and_sibling_assets(
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    raw_dir = tmp_path / "_mineru_raw"
    auto_dir = raw_dir / "paper" / "auto"
    images_dir = auto_dir / "images"
    images_dir.mkdir(parents=True)
    small_markdown = auto_dir / "preview.md"
    main_markdown = auto_dir / "paper.md"
    small_markdown.write_text("preview", encoding="utf-8")
    main_markdown.write_text("main paper content", encoding="utf-8")

    markdown_path, assets_path = module._locate_outputs(raw_dir)

    assert markdown_path == main_markdown
    assert assets_path == images_dir


def test_locate_outputs_returns_empty_result_without_markdown(
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    (tmp_path / "images").mkdir()

    assert module._locate_outputs(tmp_path) == (None, None)


def test_normalize_into_creates_stable_parser_contract(
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    source_dir = tmp_path / "_mineru_raw" / "paper" / "auto"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    source_markdown = source_dir / "paper.md"
    source_markdown.write_text(
        "# Demo\x00\n\n![architecture](images/figure-1.png)\n"
        "![external](https://example.com/external.png)\n",
        encoding="utf-8",
    )
    (images_dir / "figure-1.png").write_bytes(b"real-image-bytes")
    (images_dir / "nested").mkdir()
    layout = [{"type": "text", "page_idx": 0}]
    (source_dir / "paper_content_list.json").write_text(
        json.dumps(layout),
        encoding="utf-8",
    )
    output_dir = tmp_path / "parsed" / "demo-paper"
    stale_figures = output_dir / "figures"
    stale_figures.mkdir(parents=True)
    (stale_figures / "stale.png").write_bytes(b"stale")

    module._normalize_into(
        output_dir,
        source_markdown,
        images_dir,
    )

    normalized_markdown = (output_dir / "paper.md").read_text(
        encoding="utf-8"
    )
    assert "\x00" not in normalized_markdown
    assert "![architecture](figures/figure-1.png)" in normalized_markdown
    assert "![external](https://example.com/external.png)" in normalized_markdown
    assert (output_dir / "figures" / "figure-1.png").read_bytes() == (
        b"real-image-bytes"
    )
    assert not (output_dir / "figures" / "stale.png").exists()
    assert json.loads(
        (output_dir / "layout.json").read_text(encoding="utf-8")
    ) == layout


def _parser_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(index_dir=str(tmp_path / "index")),
        mineru=SimpleNamespace(
            cli="mineru",
            method="auto",
            lang="auto",
            timeout_sec=12,
        ),
    )


def test_parse_pdf_runs_cli_and_normalizes_real_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-boundary-test")
    output_dir = tmp_path / "parsed" / "demo-paper"
    raw_dir = output_dir / "_mineru_raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "stale.txt").write_text("stale", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured.update(
            command=command,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
            env=env,
        )
        generated_dir = Path(command[command.index("-o") + 1]) / "input" / "auto"
        images_dir = generated_dir / "images"
        images_dir.mkdir(parents=True)
        (generated_dir / "input.md").write_text(
            "# Parsed\n\n![figure](images/chart.png)\n",
            encoding="utf-8",
        )
        (images_dir / "chart.png").write_bytes(b"chart")
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_resolve_cli",
        lambda cli_name=None: "/venv/bin/mineru",
    )
    decision = module.OcrLanguageDecision(
        document_language="en",
        mineru_language="en",
        source="pdf_text",
        reason="latin_text_detected",
        model_fallback=False,
    )
    monkeypatch.setattr(
        module,
        "resolve_ocr_language",
        lambda *args, **kwargs: decision,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = module.parse_pdf("demo-paper", pdf_path)

    assert result == output_dir
    assert captured["command"] == [
        "/venv/bin/mineru",
        "-p",
        str(pdf_path.resolve()),
        "-o",
        str(raw_dir),
        "-m",
        "auto",
        "-l",
        "en",
    ]
    # 应用层 auto 必须解析为具体 ch/en, 不得原样透传给 CLI。
    assert "auto" not in captured["command"][captured["command"].index("-l"):]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 12
    assert captured["check"] is False
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["MINERU_TOOLS_CONFIG_JSON"] == str(
        (tmp_path / "config" / "magic-pdf.json").resolve()
    )
    assert Path(environment["MPLCONFIGDIR"]).is_dir()
    assert not (raw_dir / "stale.txt").exists()
    assert (output_dir / "paper.md").read_text(encoding="utf-8").startswith(
        "# Parsed"
    )
    assert (output_dir / "figures" / "chart.png").read_bytes() == b"chart"


def test_parse_pdf_falls_back_to_chinese_when_english_weights_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-boundary-test")
    output_dir = tmp_path / "parsed" / "demo-paper"
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        captured["command"] = command
        generated_dir = Path(command[command.index("-o") + 1]) / "input" / "auto"
        generated_dir.mkdir(parents=True)
        (generated_dir / "input.md").write_text(
            "# Parsed\n\ncontent\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "done", "")

    decision = module.OcrLanguageDecision(
        document_language="en",
        mineru_language="en",
        source="pdf_text",
        reason="latin_text_detected",
        model_fallback=False,
    )
    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_resolve_cli",
        lambda cli_name=None: "/venv/bin/mineru",
    )
    monkeypatch.setattr(
        module,
        "resolve_ocr_language",
        lambda *args, **kwargs: decision,
    )
    # 英文权重缺失、中文权重可用: 覆盖 autouse 夹具的乐观假设。
    monkeypatch.setattr(
        module,
        "_ocr_weights_available",
        lambda config_path, language: language == "ch",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = module.parse_pdf("demo-paper", pdf_path)

    assert result == output_dir
    language_flag = captured["command"].index("-l")
    assert captured["command"][language_flag + 1] == "ch"
    payload = json.loads(
        (output_dir / "language.json").read_text(encoding="utf-8")
    )
    assert payload["mineru_language"] == "ch"
    assert payload["model_fallback"] is True


def test_parse_pdf_rejects_missing_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    output_dir = tmp_path / "parsed" / "demo-paper"
    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(module, "_resolve_cli", lambda cli_name=None: None)

    with pytest.raises(module.MineruError, match="CLI"):
        module.parse_pdf("demo-paper", tmp_path / "input.pdf")


def test_parse_pdf_classifies_nonzero_cli_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    output_dir = tmp_path / "parsed" / "demo-paper"

    def failed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        return subprocess.CompletedProcess(
            command,
            2,
            "",
            "ModuleNotFoundError: No module named 'cv2'",
        )

    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_resolve_cli",
        lambda cli_name=None: "/venv/bin/mineru",
    )
    monkeypatch.setattr(subprocess, "run", failed_run)

    with pytest.raises(
        module.MineruError,
        match="reason=missing_cv2",
    ) as error:
        module.parse_pdf("demo-paper", tmp_path / "input.pdf")

    assert ".[mineru]" in str(error.value)


def test_parse_pdf_converts_timeout_to_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    output_dir = tmp_path / "parsed" / "demo-paper"

    def timed_out(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=12)

    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_resolve_cli",
        lambda cli_name=None: "/venv/bin/mineru",
    )
    monkeypatch.setattr(subprocess, "run", timed_out)

    with pytest.raises(module.MineruError, match="timeout after 12s"):
        module.parse_pdf("demo-paper", tmp_path / "input.pdf")


def test_parse_pdf_rejects_successful_cli_without_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    output_dir = tmp_path / "parsed" / "demo-paper"

    def empty_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(module.cfg, "load", lambda: _parser_config(tmp_path))
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda paper_id: output_dir,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_resolve_cli",
        lambda cli_name=None: "/venv/bin/mineru",
    )
    monkeypatch.setattr(subprocess, "run", empty_run)

    with pytest.raises(module.MineruError, match="produced no markdown"):
        module.parse_pdf("demo-paper", tmp_path / "input.pdf")


def test_import_check_reports_success_and_failure() -> None:
    module = _mineru_module()

    successful = module._import_check(
        "json",
        "json",
        "unused",
    )
    failed = module._import_check(
        "missing",
        "paper_rag_module_that_does_not_exist",
        "install it",
    )

    assert successful.name == "json"
    assert successful.ok is True
    assert successful.detail.startswith("import ok")
    assert successful.hint == ""
    assert failed.name == "missing"
    assert failed.ok is False
    assert "ModuleNotFoundError" in failed.detail
    assert failed.hint == "install it"


def test_model_directory_checks_real_relative_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    config_path = tmp_path / "magic-pdf.json"
    models_dir = tmp_path / "data" / "index" / "mineru_models"
    models_dir.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"models-dir": "./data/index/mineru_models"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)

    empty_checks = module._model_dir_checks(config_path)
    (models_dir / "weight.bin").write_bytes(b"weight")
    populated_checks = module._model_dir_checks(config_path)

    assert [check.ok for check in empty_checks] == [True, False]
    assert [check.ok for check in populated_checks] == [True, True]
    assert all(str(models_dir) in check.detail for check in populated_checks)


def test_enabled_layout_weight_check_uses_magic_pdf_resource_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    config_path = tmp_path / "magic-pdf.json"
    models_dir = tmp_path / "models"
    config_path.write_text(
        json.dumps(
            {
                "models-dir": "./models",
                "layout-config": {"model": "doclayout_yolo"},
                "formula-config": {"enable": False},
                "table-config": {"enable": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    relative_weight = module._mineru_weight_map()["doclayout_yolo"]
    expected_weight = models_dir / relative_weight

    missing_checks = module._enabled_model_weight_checks(config_path)
    expected_weight.parent.mkdir(parents=True)
    expected_weight.write_bytes(b"layout-weight")
    ready_checks = module._enabled_model_weight_checks(config_path)

    assert len(missing_checks) == 1
    assert missing_checks[0].name == "model weight layout:doclayout_yolo"
    assert missing_checks[0].ok is False
    assert ready_checks[0].ok is True
    assert ready_checks[0].detail == str(expected_weight)


def test_ocr_weight_checks_require_language_specific_real_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    config_path = tmp_path / "magic-pdf.json"
    config_path.write_text(
        json.dumps({"models-dir": "./models"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)

    missing_checks = module._ocr_model_weight_checks(config_path, "en")
    expected_paths = [Path(check.detail) for check in missing_checks]
    for expected_path in expected_paths:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(b"ocr-weight")
    ready_checks = module._ocr_model_weight_checks(config_path, "en")

    assert [check.name for check in missing_checks] == [
        "OCR detection weight:en",
        "OCR recognition weight:en",
    ]
    assert [check.ok for check in missing_checks] == [False, False]
    assert [check.ok for check in ready_checks] == [True, True]
    assert all(path.parent.name == "paddleocr_torch" for path in expected_paths)


def test_ocr_weight_checks_require_bilingual_weights_in_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    config_path = tmp_path / "magic-pdf.json"
    config_path.write_text(
        json.dumps({"models-dir": "./models"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)

    checks = module._ocr_model_weight_checks(config_path, "auto")

    assert [check.name for check in checks] == [
        "OCR detection weight:ch",
        "OCR recognition weight:ch",
        "OCR detection weight:en",
        "OCR recognition weight:en",
    ]


def test_ocr_weight_checks_reject_missing_language(tmp_path: Path) -> None:
    module = _mineru_module()
    config_path = tmp_path / "magic-pdf.json"
    config_path.write_text(
        json.dumps({"models-dir": str(tmp_path / "models")}),
        encoding="utf-8",
    )

    checks = module._ocr_model_weight_checks(config_path, None)

    assert len(checks) == 1
    assert checks[0].name == "OCR language"
    assert checks[0].ok is False
    assert "mineru.lang" in checks[0].hint


def test_layout_reader_weight_checks_require_config_and_model_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    model_dir = tmp_path / "models" / "Layout" / "LayoutReader"
    model_dir.mkdir(parents=True)
    config_path = tmp_path / "magic-pdf.json"
    config_path.write_text(
        json.dumps(
            {
                "layoutreader-model-dir": "./models/Layout/LayoutReader",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", tmp_path)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    model_path = model_dir / "model.safetensors"
    model_path.write_bytes(b"layout-reader-weight")

    ready_checks = module._layout_reader_weight_checks(config_path)
    model_path.unlink()
    missing_checks = module._layout_reader_weight_checks(config_path)

    assert [check.name for check in ready_checks] == [
        "LayoutReader weight:config.json",
        "LayoutReader weight:model.safetensors",
    ]
    assert [check.ok for check in ready_checks] == [True, True]
    assert [check.ok for check in missing_checks] == [True, False]


def test_cli_version_check_runs_real_executable(tmp_path: Path) -> None:
    module = _mineru_module()
    cli_path = tmp_path / "magic-pdf"
    cli_path.write_text(
        "#!/bin/sh\nprintf 'magic-pdf, version 1.3.12\\n'\n",
        encoding="utf-8",
    )
    cli_path.chmod(0o755)

    check = module._cli_version_check(str(cli_path))

    assert check.name == "cli version"
    assert check.ok is True
    assert check.detail == "magic-pdf, version 1.3.12"


def test_diagnose_aggregates_current_runtime_without_hiding_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _mineru_module()
    project_root = tmp_path / "project"
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "magic-pdf.json").write_text(
        json.dumps(
            {
                "models-dir": "./models",
                "layoutreader-model-dir": "./models/Layout/LayoutReader",
                "layout-config": {"model": "doclayout_yolo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.cfg, "PROJECT_ROOT", project_root)

    report = module.diagnose()

    check_names = {check.name for check in report.checks}
    assert report.config_path.endswith("config/magic-pdf.json")
    assert "cli" in check_names
    assert "magic-pdf config" in check_names
    assert "models-dir" in check_names
    assert {
        "OCR detection weight:ch",
        "OCR recognition weight:ch",
        "OCR detection weight:en",
        "OCR recognition weight:en",
    } <= check_names
    assert {
        "LayoutReader weight:config.json",
        "LayoutReader weight:model.safetensors",
    } <= check_names
    assert report.ok is all(check.ok for check in report.checks)
