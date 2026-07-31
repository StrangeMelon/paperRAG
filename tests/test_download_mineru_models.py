"""MinerU 双语模型下载脚本的边界测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_mineru_models.py"


def _load_script() -> ModuleType:
    assert SCRIPT_PATH.is_file(), "请先创建 scripts/download_mineru_models.py"

    spec = importlib.util.spec_from_file_location(
        "download_mineru_models_script", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pins_official_revision_and_local_layout() -> None:
    module = _load_script()

    assert module.REPO_ID == "opendatalab/PDF-Extract-Kit-1.0"
    assert module.REVISION == "a4f6a8d29a4d96730f90ea174a9322e842b93552"
    assert len(module.MODEL_FILES) == 7
    assert module.MODEL_FILES[
        "models/OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth"
    ][0] == Path("OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth")
    assert module.MODEL_FILES[
        "models/ReadingOrder/layout_reader/model.safetensors"
    ][0] == Path("Layout/LayoutReader/model.safetensors")


def _install_fake_download(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    """把 hf_hub_download 替换为写出真实临时文件的桩。"""

    def fake_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == module.REPO_ID
        assert kwargs["revision"] == module.REVISION
        filename = str(kwargs["filename"])
        calls.append(filename)
        cache_dir = Path(str(kwargs["cache_dir"]))
        source = cache_dir / filename.replace("/", "__")
        source.parent.mkdir(parents=True, exist_ok=True)
        min_size = module.MODEL_FILES[filename][1]
        source.write_bytes(b"x" * (min_size + 1))
        return str(source)

    monkeypatch.setattr(module, "hf_hub_download", fake_download, raising=False)


def test_downloads_every_file_into_models_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    calls: list[str] = []
    _install_fake_download(module, monkeypatch, calls)
    models_dir = tmp_path / "mineru_models"

    exit_code = module.main(["--models-dir", str(models_dir)])

    assert exit_code == 0
    assert sorted(calls) == sorted(module.MODEL_FILES)
    for _remote, (local_rel, min_size) in module.MODEL_FILES.items():
        target = models_dir / local_rel
        assert target.is_file()
        assert target.stat().st_size >= min_size


def test_second_run_reuses_existing_non_empty_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    calls: list[str] = []
    _install_fake_download(module, monkeypatch, calls)
    models_dir = tmp_path / "mineru_models"

    assert module.main(["--models-dir", str(models_dir)]) == 0
    first_run = list(calls)
    calls.clear()

    assert module.main(["--models-dir", str(models_dir)]) == 0

    assert first_run  # 第一次确实下载过
    assert calls == []  # 第二次全部复用, 不再触发下载
