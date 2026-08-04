"""pytest 会话级共享装置。

唯一职责: 在收集测试**之前**把仓库根的 `.env` 读入环境变量, 让
`tests/test_*_real.py` 这类真实测试不必各自复制一份 .env 解析, 也不必要求
用户先手工 `set -a; . ./.env`。

`override=False`: 已导出的环境变量优先于 .env 文件, 便于临时覆盖单次运行
(如 `CHAT_MODEL=qwen-turbo uv run pytest ...`)。纯逻辑测试不读这些变量,
`.env` 缺失时静默跳过, 因此本文件对无网络/无配置的环境无影响。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_once() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 依赖已在 pyproject 声明
        _load_dotenv_fallback(env_path)
        return
    load_dotenv(env_path, override=False)


def _load_dotenv_fallback(path: Path) -> None:
    """python-dotenv 缺失时的极简兜底: KEY=VALUE 行, 跳过注释, 不覆盖已有变量。"""
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_once()
