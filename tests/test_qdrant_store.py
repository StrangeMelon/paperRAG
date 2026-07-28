"""Qdrant 客户端生命周期与 point ID 的行为契约测试。"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _qdrant_module() -> ModuleType:
    return importlib.import_module(
        "paper_rag.store.qdrant_store"
    )


def _install_fake_client_factory(
    monkeypatch,
) -> list[_FakeClient]:
    created_clients: list[_FakeClient] = []
    fake_package = ModuleType("qdrant_client")

    def create_client(**kwargs) -> _FakeClient:
        client = _FakeClient(**kwargs)
        created_clients.append(client)
        return client

    fake_package.QdrantClient = create_client
    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        fake_package,
    )

    return created_clients


def _set_qdrant_config(
    store: ModuleType,
    monkeypatch,
    *,
    url: str,
    local_path: str | None,
) -> None:
    config = SimpleNamespace(
        qdrant=SimpleNamespace(
            url=url,
            local_path=local_path,
            collection_chunks="paper_chunks",
        )
    )
    monkeypatch.setattr(store.cfg, "load", lambda: config)
    monkeypatch.setattr(store, "_CLIENT", None)
    monkeypatch.setattr(store, "_ATEXIT_REGISTERED", False)


def test_stable_point_id_is_deterministic_unsigned_64_bit() -> None:
    store = _qdrant_module()

    first = store._stable_point_id("chunk:paper-1:0")
    repeated = store._stable_point_id("chunk:paper-1:0")
    different = store._stable_point_id("chunk:paper-1:1")

    assert first == repeated
    assert first != different
    assert isinstance(first, int)
    assert 0 <= first < 2**64


def test_get_client_caches_server_client_and_registers_cleanup(
    monkeypatch,
) -> None:
    store = _qdrant_module()
    created_clients = _install_fake_client_factory(monkeypatch)
    registered_callbacks = []

    _set_qdrant_config(
        store,
        monkeypatch,
        url="http://localhost:6333",
        local_path=None,
    )
    monkeypatch.setattr(
        store.atexit,
        "register",
        registered_callbacks.append,
    )

    first = store.get_client()
    second = store.get_client()

    assert first is second
    assert len(created_clients) == 1
    assert created_clients[0].kwargs == {
        "url": "http://localhost:6333"
    }
    assert registered_callbacks == [store.close_client]


@pytest.mark.parametrize(
    ("url", "local_path", "expected_path"),
    [
        (
            "http://localhost:6333",
            "/tmp/explicit-qdrant",
            "/tmp/explicit-qdrant",
        ),
        (
            "file:///tmp/file-qdrant",
            None,
            "/tmp/file-qdrant",
        ),
        (
            "local:///tmp/local-qdrant",
            None,
            "/tmp/local-qdrant",
        ),
    ],
)
def test_get_client_supports_embedded_mode(
    monkeypatch,
    url: str,
    local_path: str | None,
    expected_path: str,
) -> None:
    store = _qdrant_module()
    created_clients = _install_fake_client_factory(monkeypatch)

    _set_qdrant_config(
        store,
        monkeypatch,
        url=url,
        local_path=local_path,
    )
    monkeypatch.setattr(
        store.atexit,
        "register",
        lambda _callback: None,
    )

    client = store.get_client()

    assert client is created_clients[0]
    assert client.kwargs == {"path": expected_path}


def test_close_client_clears_cache_and_suppresses_close_errors(
    monkeypatch,
) -> None:
    store = _qdrant_module()
    normal_client = _FakeClient()

    monkeypatch.setattr(store, "_CLIENT", normal_client)
    store.close_client()

    assert normal_client.close_calls == 1
    assert store._CLIENT is None

    class _FailingClient(_FakeClient):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close failed")

    failing_client = _FailingClient()
    monkeypatch.setattr(store, "_CLIENT", failing_client)

    store.close_client()

    assert failing_client.close_calls == 1
    assert store._CLIENT is None

# 测试写入、删除、检索、旧客户端兼容和故障降级契约
class _OperationClient:
    def __init__(self) -> None:
        self.upsert_call = None
        self.delete_call = None

    def upsert(self, **kwargs) -> None:
        self.upsert_call = kwargs

    def delete(self, **kwargs) -> None:
        self.delete_call = kwargs


def test_upsert_chunks_builds_points_and_payloads(
    monkeypatch,
) -> None:
    store = _qdrant_module()
    client = _OperationClient()

    _set_qdrant_config(
        store,
        monkeypatch,
        url="http://localhost:6333",
        local_path=None,
    )
    monkeypatch.setattr(store, "get_client", lambda: client)

    count = store.upsert_chunks(
        [
            {
                "chunk_id": "chunk:one",
                "paper_id": "paper:one",
                "text": "First chunk",
                "vector": [999.0],
            },
            {
                "chunk_id": "chunk:two",
                "paper_id": "paper:two",
                "text": "Second chunk",
            },
        ],
        vectors=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
    )

    assert count == 2
    assert client.upsert_call is not None
    assert client.upsert_call["collection_name"] == "paper_chunks"
    assert client.upsert_call["wait"] is True

    points = client.upsert_call["points"]
    assert points[0].id == store._stable_point_id("chunk:one")
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {
        "chunk_id": "chunk:one",
        "paper_id": "paper:one",
        "text": "First chunk",
    }
    assert points[1].id == store._stable_point_id("chunk:two")


def test_upsert_chunks_rejects_misaligned_items_and_vectors() -> None:
    store = _qdrant_module()

    with pytest.raises(
        ValueError,
        match=r"items\(1\) and vectors\(0\) must align",
    ):
        store.upsert_chunks(
            [{"chunk_id": "chunk:one"}],
            vectors=[],
        )


def test_delete_chunks_for_paper_uses_paper_id_filter(
    monkeypatch,
) -> None:
    store = _qdrant_module()
    client = _OperationClient()

    _set_qdrant_config(
        store,
        monkeypatch,
        url="http://localhost:6333",
        local_path=None,
    )
    monkeypatch.setattr(store, "get_client", lambda: client)

    store.delete_chunks_for_paper("paper:delete-me")

    assert client.delete_call is not None
    assert client.delete_call["collection_name"] == "paper_chunks"
    assert client.delete_call["wait"] is True

    selector = client.delete_call["points_selector"]
    condition = selector.filter.must[0]

    assert condition.key == "paper_id"
    assert condition.match.value == "paper:delete-me"


def test_search_uses_query_points_and_metadata_filters(
    monkeypatch,
) -> None:
    store = _qdrant_module()

    class _QueryPointsClient:
        def __init__(self) -> None:
            self.call = None

        def query_points(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={
                            "chunk_id": "chunk:figure",
                            "paper_id": "paper:one",
                            "modality": "figure",
                        },
                        score=0.87,
                    )
                ]
            )

    client = _QueryPointsClient()

    _set_qdrant_config(
        store,
        monkeypatch,
        url="http://localhost:6333",
        local_path=None,
    )
    monkeypatch.setattr(store, "get_client", lambda: client)

    result = store.search(
        [0.1, 0.2],
        top_k=3,
        paper_ids=["paper:one", "paper:two"],
        modality="figure",
    )

    assert result == [
        {
            "chunk_id": "chunk:figure",
            "paper_id": "paper:one",
            "modality": "figure",
            "score": 0.87,
        }
    ]
    assert client.call is not None
    assert client.call["collection_name"] == "paper_chunks"
    assert client.call["query"] == [0.1, 0.2]
    assert client.call["limit"] == 3
    assert client.call["with_payload"] is True

    query_filter = client.call["query_filter"]
    paper_condition, modality_condition = query_filter.must

    assert paper_condition.key == "paper_id"
    assert paper_condition.match.any == ["paper:one", "paper:two"]
    assert modality_condition.key == "modality"
    assert modality_condition.match.value == "figure"


def test_search_falls_back_to_legacy_client_search(
    monkeypatch,
) -> None:
    store = _qdrant_module()

    class _LegacySearchClient:
        def __init__(self) -> None:
            self.call = None

        def search(self, **kwargs):
            self.call = kwargs
            return [
                SimpleNamespace(
                    payload={"chunk_id": "chunk:legacy"},
                    score=0.42,
                )
            ]

    client = _LegacySearchClient()

    _set_qdrant_config(
        store,
        monkeypatch,
        url="http://localhost:6333",
        local_path=None,
    )
    monkeypatch.setattr(store, "get_client", lambda: client)

    result = store.search([0.9], top_k=1)

    assert result == [
        {
            "chunk_id": "chunk:legacy",
            "score": 0.42,
        }
    ]
    assert client.call is not None
    assert client.call["query_vector"] == [0.9]
    assert client.call["query_filter"] is None
    assert client.call["limit"] == 1


def test_search_returns_empty_list_when_qdrant_fails(
    monkeypatch,
) -> None:
    store = _qdrant_module()

    def raise_connection_error():
        raise ConnectionError("qdrant unavailable")

    monkeypatch.setattr(
        store,
        "get_client",
        raise_connection_error,
    )

    assert store.search([0.1, 0.2]) == []
