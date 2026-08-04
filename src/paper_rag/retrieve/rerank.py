"""BGE reranker (cross-encoder) 重排: 检索漏斗最后的质量闸门。

bi-encoder(dense)在入库时独立编码文档, 不知道将来的查询; cross-encoder 把
(查询, 文档)拼接后过注意力逐词交互, 精度高但每对一次前向, 只精排融合后的
少量候选(top_k*2 -> top_k)。score_rerank 经 normalize(sigmoid)落在 0~1,
是下游 abstain 弃答决策的另一路绝对信号。

模型 BAAI/bge-reranker-v2-m3 为多语种(与 BGE-M3 同家族), zh/en/跨语言对
零代码适配。

失败模式(与基准一致): FlagEmbedding 未装 / 模型加载失败(_LOAD_FAILED 闩锁,
不反复重试) / enabled=false / compute_score 运行时异常, 一律回退 RRF 原序
截断 top_k——重排是增强器不是依赖项。

相对基准的唯一偏离: 拷贝候选后打分排序, 不原地改写调用方的列表与 dict
(与 hybrid 课 rrf_fuse 的不可变约定一致)。
"""

from __future__ import annotations

from .. import config as cfg
from ..utils.hf_cache import resolve_cached_snapshot
from ..utils.logger import get_logger

log = get_logger("retrieve.rerank")
_MODEL = None
_LOAD_FAILED = False


def _model():
    global _MODEL, _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if _MODEL is None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            log.warning(f"FlagEmbedding not installed: {e}; reranker disabled")
            _LOAD_FAILED = True
            return None
        c = cfg.load()
        cache_dir = c.reranker.cache_dir or c.paths.models_dir
        model_name = resolve_cached_snapshot(c.reranker.model_name, cache_dir)
        log.info(f"loading reranker {model_name} (cache={cache_dir})")
        try:
            _MODEL = FlagReranker(
                model_name,
                use_fp16=c.reranker.use_fp16,
                cache_dir=cache_dir,
            )
        except Exception as e:
            log.warning(f"reranker load failed: {e}; falling back to RRF order")
            _LOAD_FAILED = True
            return None
    return _MODEL


def rerank(query: str, candidates: list[dict], *, top_k: int | None = None) -> list[dict]:
    """按查询-文档相关性重排候选; 任何故障回退 RRF 原序截断。"""
    if not candidates:
        return []
    c = cfg.load()
    top_k = top_k or c.reranker.top_k
    if not c.reranker.enabled:
        return candidates[:top_k]

    model = _model()
    if model is None:
        return candidates[:top_k]

    pairs = [(query, (item.get("text") or "")) for item in candidates]
    try:
        scores = model.compute_score(pairs, normalize=True)
    except Exception as e:
        log.warning(f"reranker compute_score failed: {e}; returning RRF order")
        return candidates[:top_k]
    if isinstance(scores, float):
        scores = [scores]

    ranked = [dict(item) for item in candidates]  # 拷贝: 不改写调用方
    for cand, sc in zip(ranked, scores, strict=False):
        cand["score_rerank"] = float(sc)
    ranked.sort(key=lambda x: x.get("score_rerank", 0.0), reverse=True)
    return ranked[:top_k]
