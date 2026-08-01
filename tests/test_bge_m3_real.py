"""BGE-M3 无 mock 真实集成测试(单独运行, 缺依赖/模型时明确失败不 skip)。

    uv run pytest -vv -s tests/test_bge_m3_real.py

需要: FlagEmbedding 已安装(uv sync --extra embed), BAAI/bge-m3 已缓存到
data/index/models(首跑 scripts/demo_bge_m3.py 或本测试会自动下载约 2.3GB)。
"""

from __future__ import annotations

import math

from paper_rag.embed.bge_m3 import encode, encode_one


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def test_real_encode_shape_and_values() -> None:
    vecs = encode(["检索增强生成", "retrieval-augmented generation", "混合 hybrid 检索"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1024
        assert all(math.isfinite(x) for x in v)
        assert math.sqrt(sum(x * x for x in v)) > 0.1


def test_real_encode_one_matches_batch() -> None:
    text = "综合能源服务区块链的主从多链结构"
    assert _cos(encode_one(text), encode([text, "另一条无关文本"])[0]) > 0.995


def test_real_cross_lingual_semantic_space() -> None:
    zh, en_related, en_unrelated = encode(
        [
            "图神经网络难以建模长程依赖",
            "Graph neural networks struggle with long-range dependencies",
            "The cafeteria serves tomato omelette for lunch",
        ]
    )
    assert _cos(zh, en_related) > _cos(zh, en_unrelated) + 0.15
