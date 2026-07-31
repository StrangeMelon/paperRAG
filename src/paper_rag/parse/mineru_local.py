"""MinerU 本地解析器的运行环境与错误诊断基础。"""

from __future__ import annotations

# 标准库导入区域
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

# 项目内部导入区域
from .. import config as cfg
from ..utils.logger import get_logger
from ..utils.paths import parsed_dir
from .language import OcrLanguageDecision, resolve_ocr_language

log = get_logger(__name__)


def _runtime_cache_dir() -> Path:
    """返回 MinerU 及其依赖统一使用的运行时缓存目录。"""

    return Path(cfg.load().paths.index_dir) / "runtime_cache"


def _ensure_runtime_env(
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """补充 MinerU 运行所需的缓存环境变量并创建对应目录。"""

    output = env if env is not None else os.environ
    cache_dir = _runtime_cache_dir()

    output.setdefault(
        "MPLCONFIGDIR",
        str(cache_dir / "matplotlib"),
    )
    output.setdefault(
        "YOLO_CONFIG_DIR",
        str(cache_dir / "ultralytics"),
    )
    output.setdefault(
        "XDG_CACHE_HOME",
        str(cache_dir / "xdg"),
    )

    for variable_name in (
        "MPLCONFIGDIR",
        "YOLO_CONFIG_DIR",
        "XDG_CACHE_HOME",
    ):
        Path(output[variable_name]).mkdir(
            parents=True,
            exist_ok=True,
        )

    return output


# 增加领域异常和配置路径
class MineruError(RuntimeError):
    """MinerU 本地解析失败。"""


def _mineru_config_path() -> Path:
    """返回 MinerU 工具配置文件的标准路径。"""

    return (
        cfg.PROJECT_ROOT
        / "config"
        / "magic-pdf.json"
    ).resolve()

@dataclass
class MineruCheck:
    """一项 MinerU 环境检查的结果。"""

    name: str
    ok: bool
    detail: str
    hint: str = ""


@dataclass
class MineruDoctorReport:
    """MinerU 环境诊断报告。"""

    ok: bool
    cli_path: str | None
    config_path: str
    checks: list[MineruCheck]

    def to_dict(self) -> dict[str, Any]:
        """转换成适合 CLI 或 API 输出的普通字典。"""

        result = asdict(self)
        result["checks"] = [
            asdict(check)
            for check in self.checks
        ]
        return result


def _import_check(
    name: str,
    module_name: str,
    hint: str,
) -> MineruCheck:
    """检查 Python 模块是否能够真正完成导入。"""

    try:
        imported_module = import_module(module_name)
    except Exception as exc:
        return MineruCheck(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint=hint,
        )

    version = getattr(
        imported_module,
        "__version__",
        "",
    )
    version_suffix = (
        f" ({version})"
        if version
        else ""
    )

    return MineruCheck(
        name=name,
        ok=True,
        detail=f"import ok{version_suffix}",
    )


def _model_dir_checks(
    config_path: Path,
) -> list[MineruCheck]:
    """检查 MinerU 模型根目录是否存在且包含文件。"""

    if not config_path.exists():
        return []

    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return [
            MineruCheck(
                name="models-dir",
                ok=False,
                detail=(
                    f"cannot parse {config_path}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                hint="检查 config/magic-pdf.json 的 JSON 语法。",
            )
        ]

    configured_path = payload.get("models-dir")

    if (
        not isinstance(configured_path, str)
        or not configured_path.strip()
    ):
        return [
            MineruCheck(
                name="models-dir",
                ok=False,
                detail="not configured",
                hint="在 magic-pdf.json 中配置 models-dir。",
            ),
            MineruCheck(
                name="models-dir nonempty",
                ok=False,
                detail="models-dir missing",
                hint="下载 MinerU 模型并保持官方目录结构。",
            ),
        ]

    models_dir = Path(configured_path).expanduser()
    if not models_dir.is_absolute():
        models_dir = (
            cfg.PROJECT_ROOT / models_dir
        ).resolve()

    directory_exists = models_dir.is_dir()
    contains_files = (
        directory_exists
        and any(
            path.is_file()
            for path in models_dir.rglob("*")
        )
    )

    return [
        MineruCheck(
            name="models-dir",
            ok=directory_exists,
            detail=str(models_dir),
            hint=(
                "执行 MinerU 模型下载步骤, "
                "或把 models-dir 指向已有权重目录。"
            ),
        ),
        MineruCheck(
            name="models-dir nonempty",
            ok=contains_files,
            detail=(
                str(models_dir)
                if directory_exists
                else f"{models_dir} missing"
            ),
            hint=(
                "模型目录不能为空, "
                "并且需要保持 MinerU 官方目录结构。"
            ),
        ),
    ]


def _mineru_weight_map() -> dict[str, str]:
    """读取当前 magic-pdf 版本自带的模型权重映射。"""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML 未安装, 无法读取 MinerU 模型配置。"
        ) from exc

    magic_pdf = import_module("magic_pdf")
    package_root = Path(
        magic_pdf.__file__
    ).resolve().parent

    model_config_path = (
        package_root
        / "resources"
        / "model_config"
        / "model_configs.yaml"
    )

    payload = yaml.safe_load(
        model_config_path.read_text(encoding="utf-8")
    )
    weights = payload.get("weights") or {}

    return {
        str(model_name): str(relative_path)
        for model_name, relative_path in weights.items()
    }


def _enabled_model_weight_checks(
    config_path: Path,
) -> list[MineruCheck]:
    """检查当前启用的布局、公式和表格模型权重。"""

    if not config_path.exists():
        return []

    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        weight_map = _mineru_weight_map()
    except Exception as exc:
        return [
            MineruCheck(
                name="enabled model weights",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                hint=(
                    "检查 magic-pdf.json 和当前安装的 "
                    "magic_pdf 模型配置文件。"
                ),
            )
        ]

    configured_models_dir = (
        payload.get("models-dir")
        or "/tmp/models"
    )
    models_dir = Path(
        configured_models_dir
    ).expanduser()

    if not models_dir.is_absolute():
        models_dir = (
            cfg.PROJECT_ROOT / models_dir
        ).resolve()

    expected_weights: list[tuple[str, Path]] = []

    layout_config = (
        payload.get("layout-config")
        or {}
    )
    layout_model = layout_config.get(
        "model",
        "layoutlmv3",
    )
    if layout_model in weight_map:
        expected_weights.append(
            (
                f"layout:{layout_model}",
                models_dir / weight_map[layout_model],
            )
        )

    formula_config = (
        payload.get("formula-config")
        or {}
    )
    if formula_config.get("enable", True):
        for config_key in (
            "mfd_model",
            "mfr_model",
        ):
            model_name = formula_config.get(config_key)
            if model_name in weight_map:
                expected_weights.append(
                    (
                        f"formula:{model_name}",
                        models_dir / weight_map[model_name],
                    )
                )

    table_config = (
        payload.get("table-config")
        or {}
    )
    if table_config.get("enable", False):
        table_model = table_config.get("model")
        if table_model in weight_map:
            expected_weights.append(
                (
                    f"table:{table_model}",
                    models_dir / weight_map[table_model],
                )
            )

    checks: list[MineruCheck] = []

    for model_name, weight_path in expected_weights:
        checks.append(
            MineruCheck(
                name=f"model weight {model_name}",
                ok=weight_path.exists(),
                detail=str(weight_path),
                hint=(
                    "下载 MinerU 模型权重, "
                    "并保持 model_configs.yaml 定义的目录结构。"
                ),
            )
        )

    return checks


def _ocr_model_weight_checks(
    config_path: Path,
    language: str | None,
) -> list[MineruCheck]:
    """检查一个语言或 auto 模式所需的真实 OCR 权重。"""

    if not config_path.exists():
        return []
    if not language:
        return [
            MineruCheck(
                name="OCR language",
                ok=False,
                detail="not configured",
                hint="将 mineru.lang 设置为 auto、ch 或 en。",
            )
        ]

    languages = ("ch", "en") if language == "auto" else (language,)
    checks: list[MineruCheck] = []
    for selected_language in languages:
        try:
            detection_path, recognition_path = _ocr_weight_paths(
                config_path,
                selected_language,
            )
        except Exception as exc:
            checks.append(
                MineruCheck(
                    name=f"OCR weights:{selected_language}",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                    hint="检查 magic-pdf OCR 模型配置与语言值。",
                )
            )
            continue

        for role, weight_path in (
            ("detection", detection_path),
            ("recognition", recognition_path),
        ):
            checks.append(
                MineruCheck(
                    name=f"OCR {role} weight:{selected_language}",
                    ok=weight_path.is_file(),
                    detail=str(weight_path),
                    hint=(
                        "下载对应语言的 MinerU OCR 权重, "
                        "并放入 OCR/paddleocr_torch/。"
                    ),
                )
            )
    return checks


def _full_extra_checks() -> list[MineruCheck]:
    """检查 MinerU 完整解析依赖。"""

    hint = (
        "安装 MinerU 完整依赖: "
        f"{sys.executable} -m pip install -e '.[mineru]'"
    )

    return [
        _import_check(
            "magic-pdf[full]: doclayout_yolo",
            "doclayout_yolo",
            hint,
        ),
        _import_check(
            "magic-pdf[full]: ultralytics",
            "ultralytics",
            hint,
        ),
        _import_check(
            "magic-pdf[full]: rapid_table",
            "rapid_table",
            hint,
        ),
        _import_check(
            "magic-pdf[full]: pyclipper",
            "pyclipper",
            hint,
        ),
        _import_check(
            "magic-pdf[full]: shapely",
            "shapely",
            hint,
        ),
    ]


def _cli_version_check(
    cli_path: str,
) -> MineruCheck:
    """执行真实 CLI 的 --version 检查。"""

    try:
        process = subprocess.run(
            [
                cli_path,
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return MineruCheck(
            name="cli version",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="确认 magic-pdf CLI 可执行且当前环境正常。",
        )

    output = (
        process.stdout
        or process.stderr
        or ""
    ).strip()

    return MineruCheck(
        name="cli version",
        ok=process.returncode == 0,
        detail=output or f"rc={process.returncode}",
        hint=(
            "运行 magic-pdf --version 检查 CLI 安装。"
            if process.returncode != 0
            else ""
        ),
    )


def diagnose() -> MineruDoctorReport:
    """检查本地 MinerU 是否具备开始解析的条件。"""

    application_config = cfg.load()
    _ensure_runtime_env()

    config_path = _mineru_config_path()
    checks: list[MineruCheck] = []

    cli_path = _resolve_cli(
        application_config.mineru.cli
    )
    checks.append(
        MineruCheck(
            name="cli",
            ok=cli_path is not None,
            detail=(
                cli_path
                or f"{application_config.mineru.cli} not found"
            ),
            hint=(
                "安装 MinerU: "
                f"{sys.executable} -m pip install -e '.[mineru]'"
            ),
        )
    )

    checks.append(
        _import_check(
            "magic_pdf",
            "magic_pdf",
            "安装 magic-pdf。",
        )
    )
    checks.extend(_full_extra_checks())

    checks.append(
        _import_check(
            "cv2",
            "cv2",
            (
                "安装 opencv-python-headless, "
                "MinerU 解析时会导入 cv2。"
            ),
        )
    )

    checks.append(
        MineruCheck(
            name="magic-pdf config",
            ok=config_path.is_file(),
            detail=str(config_path),
            hint=(
                "创建 config/magic-pdf.json, "
                "或设置 MINERU_TOOLS_CONFIG_JSON。"
            ),
        )
    )

    checks.extend(
        _model_dir_checks(config_path)
    )
    checks.extend(
        _enabled_model_weight_checks(config_path)
    )
    checks.extend(
        _layout_reader_weight_checks(config_path)
    )
    checks.extend(
        _ocr_model_weight_checks(
            config_path,
            application_config.mineru.lang,
        )
    )

    if cli_path is not None:
        checks.append(
            _cli_version_check(cli_path)
        )

    return MineruDoctorReport(
        ok=all(check.ok for check in checks),
        cli_path=cli_path,
        config_path=str(config_path),
        checks=checks,
    )


def _resolve_cli(
    cli_name: str | None = None,
) -> str | None:
    """从 PATH 或当前 Python 环境中定位 MinerU CLI。"""

    name = cli_name or cfg.load().mineru.cli

    executable = shutil.which(name)
    if executable is not None:
        return executable

    candidates = (
        Path(sys.executable).parent / name,
        Path(sys.prefix) / "bin" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def classify_failure(detail: str) -> tuple[str, str]:
    """将 MinerU 的底层错误文本转换成稳定的错误类别和建议。"""

    normalized = detail.lower()

    if (
        "no module named 'cv2'" in normalized
        or "no module named cv2" in normalized
    ):
        return (
            "missing_cv2",
            "安装 MinerU 依赖: "
            f"{sys.executable} -m pip install -e '.[mineru]'",
        )

    full_extra_modules = (
        "doclayout_yolo",
        "ultralytics",
        "rapid_table",
        "pyclipper",
        "shapely",
    )
    for module_name in full_extra_modules:
        if (
            f"no module named '{module_name}'" in normalized
            or f"no module named {module_name}" in normalized
        ):
            return (
                "missing_mineru_full_extra",
                "安装完整 MinerU 依赖: "
                f"{sys.executable} -m pip install -e '.[mineru]'",
            )

    if (
        "no such file or directory" in normalized
        and (
            "magic-pdf" in normalized
            or "mineru" in normalized
        )
    ):
        return (
            "missing_cli",
            "安装 magic-pdf, 或在配置中设置 mineru.cli。",
        )

    model_hosts = (
        "api.github.com",
        "github.com",
        "huggingface.co",
    )
    network_errors = (
        "failed to resolve",
        "connectionerror",
        "max retries",
        "environment is not online",
    )
    if (
        "missing_models_or_offline" in normalized
        or (
            any(host in normalized for host in model_hosts)
            and any(error in normalized for error in network_errors)
        )
    ):
        return (
            "missing_models_or_offline",
            "MinerU 需要下载模型, 但当前 network/DNS 不可用。"
            "请提前下载模型并放入配置指定的模型目录。",
        )

    if (
        "model" in normalized
        and any(
            marker in normalized
            for marker in (
                "not found",
                "no such file",
                "download",
            )
        )
    ):
        return (
            "missing_models",
            "请检查 config/magic-pdf.json 中的 models-dir, "
            "并完成 MinerU 模型下载。",
        )

    return "unknown", ""

def _locate_outputs(
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """定位 MinerU 生成的主 Markdown 和资源目录。"""

    markdown_candidates = list(output_dir.rglob("*.md"))
    if not markdown_candidates:
        return None, None

    markdown_path = max(
        markdown_candidates,
        key=lambda path: path.stat().st_size,
    )

    assets_path: Path | None = None
    for directory_name in (
        "images",
        "figures",
        "assets",
    ):
        candidate = markdown_path.parent / directory_name
        if candidate.is_dir():
            assets_path = candidate
            break

    return markdown_path, assets_path


_IMAGE_REFERENCE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)"
)


def _normalize_into(
    output_dir: Path,
    source_markdown: Path,
    mineru_assets: Path | None,
) -> None:
    """将 MinerU 产物转换为项目内部统一的解析目录结构。"""

    figures_dir = output_dir / "figures"

    if figures_dir.exists():
        shutil.rmtree(figures_dir)

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    asset_paths: dict[str, str] = {}

    if mineru_assets is not None and mineru_assets.is_dir():
        for source_asset in mineru_assets.iterdir():
            if not source_asset.is_file():
                continue

            target_asset = figures_dir / source_asset.name
            shutil.copy2(
                source_asset,
                target_asset,
            )
            asset_paths[source_asset.name] = (
                f"figures/{source_asset.name}"
            )

    markdown = source_markdown.read_text(
        encoding="utf-8",
        errors="replace",
    )
    markdown = markdown.replace("\x00", "")

    def rewrite_image_reference(
        match: re.Match[str],
    ) -> str:
        alternative_text = match.group(1)
        original_path = match.group(2)
        basename = Path(original_path).name
        normalized_path = asset_paths.get(
            basename,
            original_path,
        )

        return (
            f"![{alternative_text}]"
            f"({normalized_path})"
        )

    markdown = _IMAGE_REFERENCE_PATTERN.sub(
        rewrite_image_reference,
        markdown,
    )

    (output_dir / "paper.md").write_text(
        markdown,
        encoding="utf-8",
    )

    layout_candidates = [
        *source_markdown.parent.glob(
            "*content_list*.json"
        ),
        *source_markdown.parent.glob(
            "*middle*.json"
        ),
    ]

    for candidate in layout_candidates:
        try:
            layout = json.loads(
                candidate.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        (output_dir / "layout.json").write_text(
            json.dumps(
                layout,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        break


def _select_available_ocr_language(
    decision: OcrLanguageDecision,
    config_path: Path,
) -> OcrLanguageDecision:
    if _ocr_weights_available(config_path, decision.mineru_language):
        return decision
    if (
        decision.mineru_language == "en"
        and _ocr_weights_available(config_path, "ch")
    ):
        return replace(
            decision,
            mineru_language="ch",
            reason=f"{decision.reason};english_weights_missing",
            model_fallback=True,
        )
    raise MineruError(
        f"OCR model weights missing for {decision.mineru_language}"
    )


# 真正的解析调度函数
def parse_pdf(
    paper_id: str,
    pdf_path: str | Path,
) -> Path:
    """调用本地 MinerU CLI 并返回标准化解析目录。"""

    config = cfg.load()
    resolved_pdf_path = Path(pdf_path).resolve()

    output_dir = parsed_dir(paper_id)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_output_dir = output_dir / "_mineru_raw"
    if raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)

    raw_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cli_path = _resolve_cli(config.mineru.cli)
    if cli_path is None:
        raise MineruError(
            "MinerU CLI 未找到。"
            "请安装 MinerU, 或检查配置项 mineru.cli。"
        )

    config_path = _mineru_config_path()
    decision = resolve_ocr_language(
        resolved_pdf_path,
        config.mineru.lang,
    )
    decision = _select_available_ocr_language(decision, config_path)
    (output_dir / "language.json").write_text(
        json.dumps(asdict(decision), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = [
        cli_path,
        "-p",
        str(resolved_pdf_path),
        "-o",
        str(raw_output_dir),
        "-m",
        config.mineru.method,
    ]

    # if config.mineru.lang:
    #     command.extend(
    #         [
    #             "-l",
    #             config.mineru.lang,
    #         ]
    #     )
    command.extend(["-l", decision.mineru_language])

    environment = _ensure_runtime_env(
        os.environ.copy()
    )
    environment["MINERU_TOOLS_CONFIG_JSON"] = str(
        _mineru_config_path()
    )

    log.info(
        f"启动 MinerU 解析: {resolved_pdf_path.name}"
    )

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.mineru.timeout_sec,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise MineruError(
            "MinerU timeout after "
            f"{config.mineru.timeout_sec}s"
        ) from exc

    if process.returncode != 0:
        stdout = (process.stdout or "")[-1000:]
        stderr = (process.stderr or "")[-2000:]
        detail = "\n".join(
            part
            for part in (stdout, stderr)
            if part
        )
        reason, hint = classify_failure(detail)

        message = (
            "MinerU 执行失败 "
            f"(rc={process.returncode}, reason={reason}): "
            f"{detail}"
        )
        if hint:
            message += f"\nHint: {hint}"

        raise MineruError(message)

    markdown_path, assets_path = _locate_outputs(
        raw_output_dir
    )

    if (
        markdown_path is None
        or not markdown_path.exists()
        or markdown_path.stat().st_size == 0
    ):
        stdout = (process.stdout or "")[-1000:]
        stderr = (process.stderr or "")[-1000:]
        detail = "\n".join(
            part
            for part in (stdout, stderr)
            if part
        )
        reason, hint = classify_failure(detail)

        message = (
            "MinerU produced no markdown under "
            f"{raw_output_dir} "
            f"(reason={reason}): {detail}"
        )
        if hint:
            message += f"\nHint: {hint}"

        raise MineruError(message)

    _normalize_into(
        output_dir,
        markdown_path,
        assets_path,
    )

    log.info(
        f"MinerU 解析完成: {output_dir}"
    )

    return output_dir

def _ocr_weight_paths(
    config_path: Path,
    language: str,
) -> tuple[Path, Path]:
    """返回指定语言的检测与识别权重路径。"""

    import yaml

    mineru_config = json.loads(config_path.read_text(encoding="utf-8"))
    magic_pdf = import_module("magic_pdf")
    package_root = Path(magic_pdf.__file__).resolve().parent
    ocr_config_path = (
        package_root
        / "model"
        / "sub_modules"
        / "ocr"
        / "paddleocr2pytorch"
        / "pytorchocr"
        / "utils"
        / "resources"
        / "models_config.yml"
    )
    ocr_config = yaml.safe_load(ocr_config_path.read_text(encoding="utf-8"))
    selected = (ocr_config.get("lang") or {}).get(language)
    if not isinstance(selected, dict):
        raise ValueError(f"unsupported OCR language: {language}")

    detection_name = selected.get("det")
    recognition_name = selected.get("rec")
    if not isinstance(detection_name, str) or not isinstance(recognition_name, str):
        raise ValueError(f"incomplete OCR model config: {language}")

    configured_models_dir = mineru_config.get("models-dir") or "/tmp/models"
    models_dir = Path(configured_models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = (cfg.PROJECT_ROOT / models_dir).resolve()
    ocr_dir = models_dir / "OCR" / "paddleocr_torch"
    return ocr_dir / detection_name, ocr_dir / recognition_name


def _ocr_weights_available(config_path: Path, language: str) -> bool:
    try:
        return all(
            path.is_file()
            for path in _ocr_weight_paths(config_path, language)
        )
    except Exception:
        return False


def _layout_reader_weight_checks(config_path: Path) -> list[MineruCheck]:
    """检查阅读顺序模型的配置与权重。"""

    if not config_path.is_file():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        configured_dir = payload.get("layoutreader-model-dir")
        if not isinstance(configured_dir, str) or not configured_dir.strip():
            raise ValueError("layoutreader-model-dir not configured")
        model_dir = Path(configured_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = (cfg.PROJECT_ROOT / model_dir).resolve()
    except Exception as exc:
        return [
            MineruCheck(
                name="LayoutReader config",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                hint="配置 layoutreader-model-dir。",
            )
        ]

    checks: list[MineruCheck] = []
    for filename in ("config.json", "model.safetensors"):
        path = model_dir / filename
        checks.append(
            MineruCheck(
                name=f"LayoutReader weight:{filename}",
                ok=path.is_file(),
                detail=str(path),
                hint="下载 LayoutReader 配置和 safetensors 权重。",
            )
        )
    return checks
