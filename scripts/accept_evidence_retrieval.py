#!/usr/bin/env python3
"""Run the shared evidence service against real local retrieval infrastructure."""

from __future__ import annotations

import argparse
import json
from typing import Any


def validate_execution(execution: Any) -> dict[str, Any]:
    candidates = list(execution.candidate_chunks)
    evidence = list(execution.evidence_chunks)
    assert candidates, "real retrieval returned no candidates"
    assert evidence, "real retrieval returned no selected evidence"
    assert execution.allowed_chunk_ids, "real retrieval produced no citation allowlist"

    components = {
        "dense": any("score_dense" in chunk for chunk in candidates),
        "sparse": any("score_bm25" in chunk for chunk in candidates),
        "rrf": any("score_rrf" in chunk for chunk in candidates),
        "reranker": any("score_rerank" in chunk for chunk in candidates),
        "llm_rewrite": any(
            bool(rewrite.get("raw")) or len(rewrite.get("dense_queries") or []) > 1
            for rewrite in execution.trace.get("rewrites") or []
        ),
    }
    missing = [name for name, present in components.items() if not present]
    assert not missing, f"real retrieval component signals missing: {missing}"
    assert execution.public_decision in {"confident", "weak_evidence"}
    assert set(execution.allowed_chunk_ids) == {
        chunk["chunk_id"] for chunk in evidence if chunk.get("chunk_id")
    }

    return {
        "status": "accepted",
        "retrieval_id": execution.retrieval_id,
        "decision": execution.public_decision,
        "candidate_count": len(candidates),
        "evidence_count": len(evidence),
        "paper_ids": list(dict.fromkeys(chunk.get("paper_id") for chunk in evidence)),
        "iterations": len(execution.trace.get("iters") or []),
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="比较区块链共识、智能合约与隐私保护在论文库中的主要方法和差异",
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(override=False)

    from paper_rag.rag.evidence_retrieval import Principal, retrieve_evidence

    execution = retrieve_evidence(
        args.query,
        paper_ids=args.paper_ids,
        max_evidence=4,
        include_wiki=False,
        wiki_max_entries=0,
        principal=Principal(tenant_id="system", user_id="system"),
    )
    print(json.dumps(validate_execution(execution), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
