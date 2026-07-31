"""解析包入口的依赖轻量性测试。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_parse_package_is_explicit_and_dependency_light() -> None:
    lazy_submodules = (
        "paper_rag.parse.fallback_pymupdf",
        "paper_rag.parse.mineru_local",
        "paper_rag.parse.dispatcher",
    )
    # 其他解析测试可能已把子模块导入 sys.modules; 先快照并弹出相关模块,
    # 使本测试验证的是"重新导入 paper_rag.parse 这个动作本身不连带子模块",
    # 而非依赖全局进程状态(否则全量执行顺序会让断言假失败)。
    tracked = ("paper_rag.parse", *lazy_submodules)
    saved = {name: sys.modules.pop(name) for name in tracked if name in sys.modules}
    try:
        module = importlib.import_module("paper_rag.parse")

        assert module.__file__ is not None, (
            "paper_rag.parse 目前只是 namespace package; "
            "需要创建 parse/__init__.py"
        )
        assert Path(module.__file__).name == "__init__.py"
        assert module.__doc__

        for name in lazy_submodules:
            assert name not in sys.modules, (
                f"导入 paper_rag.parse 不应连带导入 {name}"
            )
    finally:
        # 恢复原始 sys.modules,避免污染后续测试。
        for name in tracked:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
