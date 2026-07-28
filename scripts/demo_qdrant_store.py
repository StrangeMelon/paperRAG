"""使用真实 Qdrant 演示向量写入、检索、过滤和删除。"""

from __future__ import annotations

import uuid
from typing import Any

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


def _show_hits(
    title: str,
    hits: list[dict[str, Any]],
) -> None:
    print(f"\n{title}")
    if not hits:
        print("  <empty>")
        return

    for index, hit in enumerate(hits, start=1):
        print(
            f"  {index}. "
            f"chunk_id={hit.get('chunk_id')} "
            f"paper_id={hit.get('paper_id')} "
            f"modality={hit.get('modality')} "
            f"score={hit.get('score'):.4f}"
        )


def main() -> None:
    config = cfg.load()
    original_collection = config.qdrant.collection_chunks
    demo_collection = (
        f"paper_chunks_demo_{uuid.uuid4().hex[:8]}"
    )
    config.qdrant.collection_chunks = demo_collection

    client = None
    collection_created = False

    try:
        print("[1/6] 连接真实 Qdrant")
        client = qdrant_store.get_client()
        print(f"      collection={demo_collection}")

        from qdrant_client.http import models as qdrant_models

        print("[2/6] 创建隔离 collection")
        distance = (
            qdrant_models.Distance.COSINE
            if config.qdrant.distance.lower() == "cosine"
            else qdrant_models.Distance.DOT
        )
        client.create_collection(
            collection_name=demo_collection,
            vectors_config=qdrant_models.VectorParams(
                size=config.embedding.dim,
                distance=distance,
            ),
        )
        collection_created = True
        print(
            f"      dimension={config.embedding.dim} "
            f"distance={config.qdrant.distance}"
        )

        print("[3/6] 写入 3 个真实 chunk 与向量")
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
                "chunk_id": "demo:paper-a:text",
                "paper_id": "demo:paper-a",
                "modality": "text",
                "text": "Text about retrieval.",
            },
            {
                "chunk_id": "demo:paper-a:figure",
                "paper_id": "demo:paper-a",
                "modality": "figure",
                "text": "A figure about retrieval.",
            },
            {
                "chunk_id": "demo:paper-b:text",
                "paper_id": "demo:paper-b",
                "modality": "text",
                "text": "An unrelated paper.",
            },
        ]
        vectors = [
            query_vector,
            figure_vector,
            opposite_vector,
        ]

        written = qdrant_store.upsert_chunks(items, vectors)
        assert written == 3
        print(f"      written={written}")

        print("[4/6] 执行无过滤向量检索")
        all_hits = qdrant_store.search(
            query_vector,
            top_k=3,
        )
        _show_hits("      results:", all_hits)

        assert all_hits
        assert all_hits[0]["chunk_id"] == "demo:paper-a:text"
        assert len(all_hits) == 3

        print("[5/6] 验证 paper_id 与 modality 过滤")
        paper_hits = qdrant_store.search(
            query_vector,
            top_k=3,
            paper_ids=["demo:paper-b"],
        )
        _show_hits("      paper filter:", paper_hits)

        assert len(paper_hits) == 1
        assert paper_hits[0]["paper_id"] == "demo:paper-b"

        figure_hits = qdrant_store.search(
            query_vector,
            top_k=3,
            modality="figure",
        )
        _show_hits("      modality filter:", figure_hits)

        assert len(figure_hits) == 1
        assert figure_hits[0]["modality"] == "figure"

        print("[6/6] 删除 paper-a 并确认结果消失")
        qdrant_store.delete_chunks_for_paper(
            "demo:paper-a"
        )
        remaining_hits = qdrant_store.search(
            query_vector,
            top_k=3,
        )
        _show_hits("      after delete:", remaining_hits)

        assert remaining_hits
        assert all(
            hit["paper_id"] != "demo:paper-a"
            for hit in remaining_hits
        )
        assert any(
            hit["paper_id"] == "demo:paper-b"
            for hit in remaining_hits
        )

        print("\n真实 Qdrant Demo 验收通过。")

    finally:
        if client is not None and collection_created:
            try:
                client.delete_collection(
                    collection_name=demo_collection
                )
                print(
                    f"\n清理隔离 collection: {demo_collection}"
                )
            except Exception as error:  # noqa: BLE001
                print(
                    f"\n清理 collection 失败: {error}"
                )

        config.qdrant.collection_chunks = original_collection
        qdrant_store.close_client()


if __name__ == "__main__":
    main()