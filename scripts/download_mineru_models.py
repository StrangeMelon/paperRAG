"""从固定官方修订下载 MinerU 双语最小权重集合。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "index" / "mineru_models"

REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
REVISION = "a4f6a8d29a4d96730f90ea174a9322e842b93552"
MODEL_FILES: dict[str, tuple[Path, int]] = {
    "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt": (
        Path("Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"),
        100_000,
    ),
    "models/ReadingOrder/layout_reader/config.json": (
        Path("Layout/LayoutReader/config.json"),
        100,
    ),
    "models/ReadingOrder/layout_reader/model.safetensors": (
        Path("Layout/LayoutReader/model.safetensors"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth": (
        Path("OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth": (
        Path("OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth": (
        Path("OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth"),
        100_000,
    ),
    "models/OCR/paddleocr_torch/en_PP-OCRv4_rec_infer.pth": (
        Path("OCR/paddleocr_torch/en_PP-OCRv4_rec_infer.pth"),
        100_000,
    ),
}


def _download_one(
    remote_path: str,
    local_rel: Path,
    min_size: int,
    models_dir: Path,
) -> str:
    """下载或复用单个权重文件, 返回 downloaded / reuse。"""

    target = models_dir / local_rel
    if target.is_file() and target.stat().st_size >= min_size:
        print(f"[reuse] {local_rel} ({target.stat().st_size} bytes)")
        return "reuse"

    cached = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            revision=REVISION,
            cache_dir=models_dir / ".hf-cache",
        )
    )
    size = cached.stat().st_size
    if size < min_size:
        raise RuntimeError(
            f"下载文件过小: {remote_path} 仅 {size} bytes (期望 >= {min_size})"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.parent / (target.name + ".part")
    shutil.copyfile(cached, part)
    part.replace(target)
    print(f"[downloaded] {local_rel} ({size} bytes)")
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载 MinerU 布局、LayoutReader 和中英文 OCR 最小权重集合。",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="模型输出目录 (默认: data/index/mineru_models)。",
    )
    args = parser.parse_args(argv)
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    for remote_path, (local_rel, min_size) in MODEL_FILES.items():
        _download_one(remote_path, local_rel, min_size, models_dir)

    print(f"完成: 模型位于 {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
