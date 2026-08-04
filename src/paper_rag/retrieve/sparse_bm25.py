"""rank_bm25 后备稀疏检索: 小规模后备 + 评测消融对照基线。

定位(2026-08-04 随知识库目标规模 20000 篇 ≈ 10^6 chunks 确认): 本模块是纯
内存 BM25(教科书公式的透明实现), 建索引全量拉库、查询线性扫全库, 只适用于
小规模语料——FTS5 后端异常时的降级路径, 以及评测课 dense/sparse/hybrid 消融
的对照组。规模护栏 retrieve.bm25_max_chunks 超限时拒绝建索引并明确告警,
不假装能跑。

相对基准的偏离(经用户确认):
- 中文分词从逐字 unigram 统一为 bigram, 复用 fts5.segment_cjk——两个稀疏
  后端 zh 粒度构造上一致, 降级不引起行为突变(关闭 ADR-0001 记账的不一致);
  代价: 单字查询只能命中孤立单字, BM25 无前缀算子可兜底(记账边界)。
- 删除只写不读的 pickle 持久化(基准写 bm25.pkl 但全仓无 pickle.load,
  死代码; 冷启动从 SQLite 重建, 小规模毫秒级)。
- payload 填真实 section/title(基准硬编码 None)。
- 0 分结果丢弃(基准把无词项交集的 0 分块按 top_k 当结果返回, 纯噪声)。
- search() 行数自愈: 索引条数与 chunk 表不一致时强制重建(基准的 invalidate
  全仓无人调用, 单篇入库后同进程缓存陈旧)。

paper_ids 过滤沿用基准的正确做法: 先全库打分再过滤, 避免 top-k 全被其他
论文占据时过滤出空。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config as cfg
from ..utils.logger import get_logger
from .fts5 import segment_cjk

log = get_logger("retrieve.bm25")

_INDEX = None

_WORD_RE = re.compile(r"[a-z0-9_]+|[一-鿿]+")


@dataclass
class _Index:
    bm25: object  # rank_bm25.BM25Okapi | None
    chunk_ids: list[str]
    payloads: list[dict]


def _tokenize(text: str | None) -> list[str]:
    """segment_cjk 展开 CJK bigram 后小写提取 token, 与 fts5 索引侧同构。"""
    return _WORD_RE.findall(segment_cjk(text or "").lower())


def _count_chunks() -> int:
    from ..store.sqlite_store import get_engine

    with get_engine().connect() as conn:
        return int(conn.exec_driver_sql("SELECT COUNT(*) FROM chunk").scalar() or 0)


def build_index(force: bool = False) -> _Index:
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX

    max_chunks = cfg.load().retrieve.bm25_max_chunks
    n_total = _count_chunks()
    if n_total > max_chunks:
        log.warning(
            f"bm25: corpus n={n_total} exceeds retrieve.bm25_max_chunks={max_chunks}, "
            "refusing to build in-memory index (use the fts5 sparse backend)"
        )
        _INDEX = _Index(bm25=None, chunk_ids=[], payloads=[])
        return _INDEX

    from rank_bm25 import BM25Okapi  # lazy import
    from sqlmodel import Session, select

    from ..store.sqlite_store import Chunk, get_engine

    with Session(get_engine()) as s:
        rows = list(s.exec(select(Chunk)))

    chunk_ids: list[str] = []
    corpus: list[list[str]] = []
    payloads: list[dict] = []
    for r in rows:
        chunk_ids.append(r.chunk_id)
        corpus.append(_tokenize(r.text))
        payloads.append(
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "section": r.section,
                "modality": r.modality,
                "page": r.page,
                "text": r.text,
                "title": r.title,
            }
        )

    if not corpus:
        log.warning("bm25: empty corpus")
        bm25 = None
    else:
        bm25 = BM25Okapi(corpus)
    _INDEX = _Index(bm25=bm25, chunk_ids=chunk_ids, payloads=payloads)
    log.info(f"bm25 index built (n={len(chunk_ids)})")
    return _INDEX


def search(query: str, top_k: int = 20, paper_ids: list[str] | None = None) -> list[dict]:
    """BM25 检索, 可选 paper_id 过滤(先全库打分后过滤)。"""
    idx = build_index(force=False)
    if len(idx.chunk_ids) != _count_chunks():
        idx = build_index(force=True)  # 行数自愈: 入库后缓存陈旧或护栏状态变化
    if idx.bm25 is None or not idx.chunk_ids:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scores = idx.bm25.get_scores(q_tokens)

    # 0 分 = 与查询无任何词项交集, 不是命中; 基准会把 0 分块按 top_k 当结果返回(噪声)
    order = [
        i
        for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if scores[i] > 0
    ]
    if paper_ids:
        allowed = set(paper_ids)
        order = [i for i in order if idx.payloads[i].get("paper_id") in allowed]
    order = order[:top_k]

    out = []
    for rank, i in enumerate(order):
        d = dict(idx.payloads[i])
        d["score_bm25"] = float(scores[i])
        d["rank_bm25"] = rank
        out.append(d)
    return out


def invalidate() -> None:
    """失效内存缓存, 下次检索时从 SQLite 重建(基准兼容 API)。"""
    global _INDEX
    _INDEX = None
