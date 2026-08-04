"""trace id 生成。"""

from __future__ import annotations

import secrets


def new_trace_id() -> str:
    """16 位 hex 字符串: 生成便宜, grep 方便。"""
    return secrets.token_hex(8)
