"""配置加载器的行为契约测试。"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import yaml


def _config_module() -> ModuleType:
    """在测试执行阶段导入,使模块缺失表现为测试失败而非收集错误。"""
    return importlib.import_module("paper_rag.config")


def _write_config(
    path: Path,
    *,
    model_name: str,
    data_root: str = "./data",
) -> None:
    """创建只包含必要字段的临时配置。"""
    raw = {
        "paths": {
            "data_root": data_root,
            "papers_dir": "./data/papers",
            "parsed_dir": "./data/parsed",
            "index_dir": "./data/index",
            "sqlite_path": "./data/index/papers.sqlite",
            "bm25_path": "./data/index/bm25.pkl",
            "models_dir": "./data/index/models",
        },
        "embedding": {
            "model_name": model_name,
        },
        "llm": {
            "base_url": "$TEST_OPENAI_BASE_URL",
            "api_key": "$TEST_OPENAI_API_KEY",
            "chat_model": "$TEST_CHAT_MODEL",
        },
    }
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )


def test_loads_default_config_as_typed_object(monkeypatch) -> None:
    config = _config_module()
    monkeypatch.delenv("PAPER_RAG_CONFIG", raising=False)
    config.load.cache_clear()

    loaded = config.load()

    assert isinstance(loaded, config.AppConfig)
    assert loaded.embedding.model_name == "BAAI/bge-m3"
    assert loaded.embedding.dim == 1024
    assert loaded.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert loaded.qdrant.collection_chunks == "paper_chunks"
    assert loaded.qdrant.collection_wiki == "wiki_entries"
    assert loaded.llm.temperatures.answer == 0.2
    assert loaded.vision.local_max_new_tokens == 256
    assert Path(loaded.paths.data_root) == (config.PROJECT_ROOT / "data").resolve()


def test_expands_environment_placeholders(monkeypatch, tmp_path: Path) -> None:
    config = _config_module()
    config_path = tmp_path / "environment.yaml"
    _write_config(config_path, model_name="environment-model")

    monkeypatch.setenv("TEST_OPENAI_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TEST_CHAT_MODEL", "test-chat-model")
    config.load.cache_clear()

    loaded = config.load(config_path)

    assert loaded.llm.base_url == "https://llm.example.com/v1"
    assert loaded.llm.api_key == "test-key"
    assert loaded.llm.chat_model == "test-chat-model"


def test_explicit_path_has_highest_priority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config_module()
    environment_path = tmp_path / "from-environment.yaml"
    explicit_path = tmp_path / "explicit.yaml"

    _write_config(
        environment_path,
        model_name="environment-model",
        data_root="./environment-data",
    )
    _write_config(
        explicit_path,
        model_name="explicit-model",
        data_root="./explicit-data",
    )

    monkeypatch.setenv("PAPER_RAG_CONFIG", str(environment_path))
    config.load.cache_clear()

    loaded_from_environment = config.load()
    assert loaded_from_environment.embedding.model_name == "environment-model"

    config.load.cache_clear()
    loaded_from_explicit_path = config.load(explicit_path)

    assert loaded_from_explicit_path.embedding.model_name == "explicit-model"
    assert Path(loaded_from_explicit_path.paths.data_root) == (
        config.PROJECT_ROOT / "explicit-data"
    ).resolve()
