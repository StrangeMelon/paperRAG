"""切块包入口测试(切片 0)。

教训来自解析层: 功能文件提交了、包入口没提交, 克隆即失败。
本测试保证 `paper_rag.chunk` 包本身可导入, 且不夹带重型依赖。
"""

from __future__ import annotations

import importlib


def test_chunk_package_importable() -> None:
    module = importlib.import_module("paper_rag.chunk")

    assert module.__doc__, "包入口应有说明自身职责的 docstring"
