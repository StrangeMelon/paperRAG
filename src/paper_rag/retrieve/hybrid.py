"""RRF 融合的 dense/稀疏混合检索: 检索层对上层的主入口。

RRF(Reciprocal Rank Fusion): score = Σ 1/(k + rank + 1), 只用名次不用分数,
天然免疫 cosine 与 BM25 的量纲差, 对语言零假设。k(rrf_k, 默认 60)钝化名次
差距, 两条列表都靠前的块才显著胜出。

融合时把 dense cosine 提升为 score_dense 保留——RRF 分数只反映相对名次,
分不清"都匹配得好"与"都匹配得烂", 下游 abstain 弃答决策需要绝对相似度。

稀疏后端回退语义与基准一致: 仅 fts5 **抛异常**时回退 rank_bm25。合法空结果
不回退——基准的中文静默退化根源(unicode61 分词)已在 fts5 课修复, 空结果
现在是诚实信号(ADR-0001)。

相对基准的一处偏离: rrf_fuse 先拷贝条目再合并字段, 不原地改写调用方传入的
dict(基准的 setdefault 会把稀疏字段写进 dense_hits 的元素)。
"""

from __future__ import annotations

from collections import defaultdict

from .. import config as cfg
from ..utils.logger import get_logger
from . import dense

log = get_logger("retrieve.hybrid")


def _sparse_search(query: str, top_k: int, paper_ids: list[str] | None) -> list[dict]:
    """按配置选稀疏后端; fts5 异常时优雅回退 rank_bm25。"""
    backend = cfg.load().retrieve.sparse_backend
    if backend == "fts5":
        try:
            from . import fts5

            return fts5.search(query, top_k=top_k, paper_ids=paper_ids)
        except Exception as e:
            log.warning(f"FTS5 backend failed, falling back to rank_bm25: {e}")
    from . import sparse_bm25

    return sparse_bm25.search(query, top_k=top_k, paper_ids=paper_ids)


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """多条有序候选列表按 RRF 名次融合, 共同块分数相加, 字段并集保留。"""
    scores: dict[str, float] = defaultdict(float)
    keep: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            cid = item.get("chunk_id")
            if not cid:
                continue
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in keep:
                keep[cid] = dict(item)  # 拷贝: 不改写调用方的 dict
            else:
                for key, value in item.items():
                    keep[cid].setdefault(key, value)
    merged = []
    for cid, sc in scores.items():
        d = keep[cid]
        d["score_rrf"] = sc
        if "score" in d and "score_dense" not in d:
            d["score_dense"] = d["score"]  # abstain 需要的绝对相似度信号
        merged.append(d)
    merged.sort(key=lambda x: x["score_rrf"], reverse=True)
    return merged


def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    paper_ids: list[str] | None = None,
    modality: str | None = None,
) -> list[dict]:
    """dense + 稀疏双腿检索后 RRF 融合, 返回 top_k*2(给 reranker 留余量)。"""
    c = cfg.load().retrieve
    top_k = top_k or c.rerank_top_k
    dense_error: Exception | None = None
    sparse_error: Exception | None = None
    try:
        dense_hits = dense.retrieve(
            query,
            top_k=c.top_k_dense,
            paper_ids=paper_ids,
            modality=modality,
        )
    except Exception as exc:
        dense_error = exc
        dense_hits = []
        log.warning(f"dense retrieval failed, continuing with sparse: {exc}")
    try:
        sparse_hits = _sparse_search(query, top_k=c.top_k_bm25, paper_ids=paper_ids)
    except Exception as exc:
        sparse_error = exc
        sparse_hits = []
        log.warning(f"sparse retrieval failed, continuing with dense: {exc}")
    if dense_error is not None and sparse_error is not None:
        raise RuntimeError("both dense and sparse retrieval are unavailable") from dense_error
    if modality:
        sparse_hits = [h for h in sparse_hits if h.get("modality") == modality]

    fused = rrf_fuse([dense_hits, sparse_hits], k=c.rrf_k)
    log.info(
        f"hybrid: backend={c.sparse_backend} dense={len(dense_hits)} "
        f"sparse={len(sparse_hits)} fused={len(fused)}"
    )
    return fused[: top_k * 2]
