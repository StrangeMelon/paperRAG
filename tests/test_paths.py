"""路径辅助函数的行为契约测试。"""

from __future__ import annotations  # 让类型注解延迟解析，减少运行时依赖和前向引用问题

import importlib  # 通过字符串动态导入模块
from pathlib import Path  # 面向对象地处理文件路径。
from types import (  # 表示 Python 模块类型、快速创建支持属性访问的简单对象
    ModuleType,
    SimpleNamespace,
)


def _paths_module() -> ModuleType:
    return importlib.import_module("paper_rag.utils.paths")

# 构造假配置
def _fake_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            papers_dir=str(tmp_path / "papers"),
            parsed_dir=str(tmp_path / "parsed"),
            index_dir=str(tmp_path / "index"),
            models_dir=str(tmp_path / "models"),
        )
    )

# 测试论文目录计算
def test_paper_dir_uses_safe_paper_identifier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths_module()
    config = _fake_config(tmp_path)
    monkeypatch.setattr(paths.cfg, "load", lambda: config)

    result = paths.paper_dir("arxiv:2310.12345")

    assert result == tmp_path / "papers" / "arxiv_2310.12345"
    assert not result.exists()

# 测试解析结果目录计算
def test_parsed_dir_uses_safe_paper_identifier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths_module()
    config = _fake_config(tmp_path)
    monkeypatch.setattr(paths.cfg, "load", lambda: config)

    result = paths.parsed_dir("doi:10.1000/example")

    assert result == tmp_path / "parsed" / "doi_10.1000_example"
    assert not result.exists()

# 测试目录初始化
def test_ensure_dirs_is_idempotent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths_module()
    config = _fake_config(tmp_path)
    monkeypatch.setattr(paths.cfg, "load", lambda: config)

    paths.ensure_dirs()
    paths.ensure_dirs()

    assert (tmp_path / "papers").is_dir()
    assert (tmp_path / "parsed").is_dir()
    assert (tmp_path / "index").is_dir()
    assert (tmp_path / "models").is_dir()
