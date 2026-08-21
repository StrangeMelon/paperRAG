"""BGE-M3 嵌入封装的边界测试(不加载真实模型, 假模型验参数与转换)。

切片 0: encode 契约(空输入不碰模型、批参数来自配置、只取 dense、tolist 转换)。
切片 1: encode_one 便捷封装与可迭代输入。

接口约定(与基准一致, 2026-08-01 确认):

    encode(texts: Iterable[str]) -> list[list[float]]   # 空输入 -> []
    encode_one(text: str) -> list[float]

真实模型加载/GPU/中英同空间语义属于真实验收(scripts/demo_bge_m3.py 与
tests/test_bge_m3_real.py), 本文件只证接口设计。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

import paper_rag.config as config
import paper_rag.embed.bge_m3 as bge


class _FakeModel:
    """记录调用参数, 返回可辨识的固定 4 维假向量。"""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"dense_vecs": np.array([[0.5, 0.25, 0.125, 0.0625]] * len(texts))}


def _patch_config(monkeypatch, *, batch_size: int = 7, max_length: int = 128) -> None:
    conf = config.load()
    conf.embedding.batch_size = batch_size
    conf.embedding.max_length = max_length
    monkeypatch.setattr(config, "load", lambda path=None: conf)


# ---------------------------------------------------------------------------
# 切片 0: encode 契约
# ---------------------------------------------------------------------------


def test_encode_empty_returns_empty_without_touching_model(monkeypatch) -> None:
    def _boom():
        raise AssertionError("空输入不应触发模型加载")

    monkeypatch.setattr(bge, "_model", _boom)
    assert bge.encode([]) == []
    assert bge.encode(iter([])) == []


def test_encode_passes_config_batch_params_and_dense_only(monkeypatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch, batch_size=7, max_length=128)

    out = bge.encode(["hello", "检索增强"])

    texts, kwargs = fake.calls[0]
    assert texts == ["hello", "检索增强"]
    assert kwargs["batch_size"] == 7
    assert kwargs["max_length"] == 128
    assert kwargs["return_dense"] is True
    assert kwargs["return_sparse"] is False
    assert kwargs["return_colbert_vecs"] is False
    assert out == [[0.5, 0.25, 0.125, 0.0625]] * 2


def test_encode_converts_numpy_to_plain_float_lists(monkeypatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch)

    out = bge.encode(["x"])
    assert isinstance(out, list) and isinstance(out[0], list)
    assert all(type(v) is float for v in out[0])  # 非 numpy 标量, 可直接进 JSON/Qdrant


# ---------------------------------------------------------------------------
# 切片 1: encode_one 与可迭代输入
# ---------------------------------------------------------------------------


def test_encode_one_returns_first_vector(monkeypatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch)

    assert bge.encode_one("单条文本") == [0.5, 0.25, 0.125, 0.0625]


def test_generator_input_accepted(monkeypatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch)

    out = bge.encode(t for t in ("a", "b", "c"))
    assert len(out) == 3
    assert fake.calls[0][0] == ["a", "b", "c"]


def test_concurrent_encode_calls_are_serialized(monkeypatch) -> None:
    class _SlowModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def encode(self, texts, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            try:
                return super().encode(texts, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    fake = _SlowModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(bge.encode_one, ["a", "b"]))

    assert fake.max_active == 1


def test_evaluation_encode_calls_can_run_concurrently(monkeypatch) -> None:
    class _SlowModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2, timeout=2)

        def encode(self, texts, **kwargs):
            self.barrier.wait()
            return super().encode(texts, **kwargs)

    fake = _SlowModel()
    monkeypatch.setattr(bge, "_model", lambda: fake)
    _patch_config(monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(bge.encode_one, text, allow_concurrent=True) for text in ("a", "b")
        ]
        assert [future.result() for future in futures]


def test_encode_holds_embedding_resource_for_model_execution(monkeypatch) -> None:
    fake = _FakeModel()
    events: list[str] = []

    @contextmanager
    def guard(name: str):
        events.append(f"enter:{name}")
        yield
        events.append(f"exit:{name}")

    monkeypatch.setattr(bge, "_model", lambda: fake)
    monkeypatch.setattr(bge, "hold_resource", guard)
    _patch_config(monkeypatch)

    bge.encode_one("query")

    assert events == ["enter:embedding", "exit:embedding"]
