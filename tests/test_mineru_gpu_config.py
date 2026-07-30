"""MinerU GPU OCR 生产配置契约。"""

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_uses_installed_cli_and_forces_ocr() -> None:
    default_config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )

    assert default_config["mineru"] == {
        "mode": "local",
        "cli": "magic-pdf",
        "method": "ocr",
        "lang": "auto",
        "timeout_sec": 600,
        "fallback_to_pymupdf": True,
    }


def test_magic_pdf_config_uses_cuda_and_project_local_models() -> None:
    config_path = PROJECT_ROOT / "config" / "magic-pdf.json"
    mineru_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert mineru_config["device-mode"] == "cuda"
    assert mineru_config["models-dir"] == "./data/index/mineru_models"
    assert mineru_config["layoutreader-model-dir"] == (
        "./data/index/mineru_models/Layout/LayoutReader"
    )
    assert mineru_config["layout-config"] == {"model": "doclayout_yolo"}
    assert mineru_config["table-config"] == {
        "model": "rapid_table",
        "sub_model": "slanet_plus",
        "enable": False,
    }
    assert mineru_config["formula-config"] == {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": False,
    }
