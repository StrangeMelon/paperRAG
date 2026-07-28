"""真实 Qdrant 集成测试, 不使用 mock。"""

from __future__ import annotations

import uuid

from paper_rag import config as cfg
from paper_rag.store import qdrant_store


def _basis_vector(
    dimension: int,
    *,
    index: int,
    value: float,
) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = value
    return vector


def test_real_qdrant_write_search_filter_delete() -> None:
    config = cfg.load()
    original_collection = config.qdrant.collection_chunks
    collection_name = (
        f"paper_chunks_integration_{uuid.uuid4().hex[:8]}"
    )
    config.qdrant.collection_chunks = collection_name

    client = None
    collection_created = False

    try:
        print("[1/6] 连接真实 Qdrant")
        client = qdrant_store.get_client()

        from qdrant_client.http import models as qdrant_models

        print("[2/6] 创建隔离 collection")
        distance = (
            qdrant_models.Distance.COSINE
            if config.qdrant.distance.lower() == "cosine"
            else qdrant_models.Distance.DOT
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=config.embedding.dim,
                distance=distance,
            ),
        )
        collection_created = True

        print("[3/6] 写入真实向量")
        query_vector = _basis_vector(
            config.embedding.dim,
            index=0,
            value=1.0,
        )
        figure_vector = _basis_vector(
            config.embedding.dim,
            index=1,
            value=1.0,
        )
        opposite_vector = _basis_vector(
            config.embedding.dim,
            index=0,
            value=-1.0,
        )

        items = [
            {
                "chunk_id": "integration:paper-a:text",
                "paper_id": "integration:paper-a",
                "modality": "text",
                "text": "retrieval text",
            },
            {
                "chunk_id": "integration:paper-a:figure",
                "paper_id": "integration:paper-a",
                "modality": "figure",
                "text": "retrieval figure",
            },
            {
                "chunk_id": "integration:paper-b:text",
                "paper_id": "integration:paper-b",
                "modality": "text",
                "text": "unrelated text",
            },
        ]

        written = qdrant_store.upsert_chunks(
            items,
            [
                query_vector,
                figure_vector,
                opposite_vector,
            ],
        )
        assert written == 3
        print(f"      written={written}")

        print("[4/6] 验证向量排序")
        all_hits = qdrant_store.search(
            query_vector,
            top_k=3,
        )
        for hit in all_hits:
            print(
                f"      {hit['chunk_id']} "
                f"score={hit['score']:.4f}"
            )

        assert len(all_hits) == 3
        assert all_hits[0]["chunk_id"] == (
            "integration:paper-a:text"
        )
        assert all_hits[0]["score"] == 1.0

        print("[5/6] 验证 metadata 过滤")
        paper_hits = qdrant_store.search(
            query_vector,
            top_k=3,
            paper_ids=["integration:paper-b"],
        )
        figure_hits = qdrant_store.search(
            query_vector,
            top_k=3,
            modality="figure",
        )

        assert len(paper_hits) == 1
        assert paper_hits[0]["paper_id"] == (
            "integration:paper-b"
        )

        assert len(figure_hits) == 1
        assert figure_hits[0]["modality"] == "figure"

        print("[6/6] 删除并验证持久化结果")
        qdrant_store.delete_chunks_for_paper(
            "integration:paper-a"
        )
        remaining_hits = qdrant_store.search(
            query_vector,
            top_k=3,
        )

        remaining_ids = {
            hit["chunk_id"]
            for hit in remaining_hits
        }
        assert remaining_ids == {
            "integration:paper-b:text"
        }

        print("      delete verified")
        print("真实 Qdrant 集成测试通过")

    finally:
        if client is not None and collection_created:
            client.delete_collection(
                collection_name=collection_name
            )

        config.qdrant.collection_chunks = original_collection
        qdrant_store.close_client()
