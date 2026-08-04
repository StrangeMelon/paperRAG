"""dense 稠密检索薄封装的行为契约测试(编码器与向量库全打桩)。

验收点:
- 查询文本原样进入 bge_m3.encode_one, 得到的向量原样进入 qdrant_store.search;
- top_k/paper_ids/modality 三个参数原样透传, 默认值 8/None/None;
- search 的返回值不加工原样上交(含空结果)。

接口约定(与基准一致):

    retrieve(query: str, top_k: int = 8, paper_ids: list[str] | None = None,
             modality: str | None = None) -> list[dict]
"""

from __future__ import annotations

from types import SimpleNamespace

from paper_rag.retrieve import dense


def _install_stubs(monkeypatch, hits: list[dict]):
    """打桩编码器与向量库, 返回调用记录。"""
    calls: dict[str, object] = {}

    def fake_encode_one(text: str) -> list[float]:
        calls["encoded"] = text
        return [0.1, 0.2, 0.3]

    def fake_search(query_vector, top_k=8, paper_ids=None, modality=None):
        calls["search"] = {
            "query_vector": query_vector,
            "top_k": top_k,
            "paper_ids": paper_ids,
            "modality": modality,
        }
        return hits

    monkeypatch.setattr(dense, "bge_m3", SimpleNamespace(encode_one=fake_encode_one))
    monkeypatch.setattr(dense, "qdrant_store", SimpleNamespace(search=fake_search))
    return calls


def test_encodes_query_and_passes_vector_to_search(monkeypatch):
    calls = _install_stubs(monkeypatch, hits=[])

    dense.retrieve("什么是选择性状态空间?")

    assert calls["encoded"] == "什么是选择性状态空间?"
    assert calls["search"]["query_vector"] == [0.1, 0.2, 0.3]


def test_default_params_are_top8_no_filters(monkeypatch):
    calls = _install_stubs(monkeypatch, hits=[])

    dense.retrieve("q")

    assert calls["search"]["top_k"] == 8
    assert calls["search"]["paper_ids"] is None
    assert calls["search"]["modality"] is None


def test_filters_pass_through_verbatim(monkeypatch):
    calls = _install_stubs(monkeypatch, hits=[])

    dense.retrieve("q", top_k=3, paper_ids=["p1", "p2"], modality="table")

    assert calls["search"]["top_k"] == 3
    assert calls["search"]["paper_ids"] == ["p1", "p2"]
    assert calls["search"]["modality"] == "table"


def test_returns_search_result_as_is(monkeypatch):
    hits = [{"paper_id": "p1", "score": 0.9}, {"paper_id": "p1", "score": 0.7}]
    _install_stubs(monkeypatch, hits=hits)

    assert dense.retrieve("q") is hits
