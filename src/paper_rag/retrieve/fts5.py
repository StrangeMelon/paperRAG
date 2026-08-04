"""SQLite FTS5 稀疏检索(BM25 打分), 含中文 CJK bigram 适配。

为什么选 FTS5 做默认稀疏后端: 零新增依赖(SQLite 已在用)、索引与 chunk 表同库
落盘、增量更新是一条 INSERT、单机可撑百万级文档。

相对基准的三处结构性偏离(实证依据与备选对比见
docs/adrs/0001-fts5-cjk-bigram.md):

1. 中文 bigram 分词镜像: unicode61 把连续中文收为单一 token(实测 fts5vocab),
   子串查询 0 命中。写入索引前与构造查询时用同一个 segment_cjk 把 CJK 串展开
   为相邻二字对, chunks_fts.text 因此存分词镜像, search() JOIN chunk 表返回
   原文。开关 retrieve.fts5_cjk_bigram。
2. porter 词干器: 基准 docstring 承诺但建表语句缺失, 补齐(dependencies/depend
   互相召回); porter 只作用 ASCII, 中文路径不受影响。
3. 去触发器, Python 侧同步: SQL 触发器无法调用 Python 分词函数。公开
   reindex_all()/sync_paper(), search() 内置行数自愈——修复基准"触发器懒建 +
   无人调 reindex_all, 先入库后首查恒空"的缺陷。已知边界: 行数相等的原地
   UPDATE 检测不到(当前入库替换语义行数必变, 暂不构成风险)。

bm25() 分数越小越好, 取负后统一为"越大越好"; rank_bm25 名次供 RRF 融合使用。
"""

from __future__ import annotations

import re

from .. import config as cfg
from ..utils.logger import get_logger

log = get_logger("retrieve.fts5")

_INITIALIZED = False

_CJK_RUN_RE = re.compile(r"[一-鿿]+")
# 清洗 FTS5 MATCH 特殊字符: 非(单词字符|CJK)一律换成空格, 连字符因此起分隔作用
# (基准是直接删除, "Graph-Mamba" 会黏成索引里不存在的单 token "GraphMamba")
_SANITIZE_RE = re.compile(r"[^\w一-鿿]+")


def _engine():
    from ..store.sqlite_store import get_engine

    return get_engine()


def _bigram_enabled() -> bool:
    return bool(cfg.load().retrieve.fts5_cjk_bigram)


def segment_cjk(text: str) -> str:
    """把连续 CJK 串展开为空格分隔的相邻二字对(单字保留 unigram)。

    纯函数, 索引写入与查询构造共用, 两侧分词自动对齐。展开处两侧补空格,
    防止 CJK 与相邻 ASCII 黏成单 token; 输出空白做归一化(多空格collapse)。
    """

    def _explode(m: re.Match[str]) -> str:
        run = m.group(0)
        if len(run) <= 2:
            return f" {run} "
        grams = " ".join(run[i : i + 2] for i in range(len(run) - 1))
        return f" {grams} "

    return " ".join(_CJK_RUN_RE.sub(_explode, text).split())


def _fts_text(text: str) -> str:
    return segment_cjk(text) if _bigram_enabled() else text


def _ensure_table() -> None:
    """懒建 chunks_fts 虚拟表(幂等); 同步交给 Python 侧, 不建触发器。"""
    global _INITIALIZED
    if _INITIALIZED:
        return

    engine = _engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                paper_id UNINDEXED,
                modality UNINDEXED,
                text,
                tokenize = "porter unicode61 remove_diacritics 2 tokenchars '_'"
            )
            """
        )
    _INITIALIZED = True
    log.info("FTS5 chunks_fts ready")


def _insert_rows(conn, rows) -> int:
    n = 0
    for chunk_id, paper_id, modality, text in rows:
        conn.exec_driver_sql(
            "INSERT INTO chunks_fts(chunk_id, paper_id, modality, text) VALUES (?, ?, ?, ?)",
            (chunk_id, paper_id, modality, _fts_text(text or "")),
        )
        n += 1
    return n


def reindex_all() -> int:
    """全量重建索引(初次回填、schema/分词方案变更后使用)。"""
    _ensure_table()
    engine = _engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM chunks_fts")
        rows = conn.exec_driver_sql(
            "SELECT chunk_id, paper_id, modality, text FROM chunk"
        ).fetchall()
        n = _insert_rows(conn, rows)
    log.info(f"FTS5 reindexed n={n}")
    return n


def sync_paper(paper_id: str) -> int:
    """单篇增量同步: 删旧插新, 与入库流水线的替换语义一致。"""
    _ensure_table()
    engine = _engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM chunks_fts WHERE paper_id = ?", (paper_id,))
        rows = conn.exec_driver_sql(
            "SELECT chunk_id, paper_id, modality, text FROM chunk WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        n = _insert_rows(conn, rows)
    log.info(f"FTS5 synced paper_id={paper_id} n={n}")
    return n


def _sync_if_stale() -> None:
    """行数自愈: chunk 表与镜像行数不一致时全量重建(覆盖先入库后首查时序)。"""
    engine = _engine()
    with engine.connect() as conn:
        n_chunk = conn.exec_driver_sql("SELECT COUNT(*) FROM chunk").scalar() or 0
        n_fts = conn.exec_driver_sql("SELECT COUNT(*) FROM chunks_fts").scalar() or 0
    if n_chunk != n_fts:
        log.info(f"FTS5 stale (chunk={n_chunk}, fts={n_fts}), reindexing")
        reindex_all()


def _build_match_query(query: str) -> str:
    """自然语言查询 -> FTS5 MATCH 语法: 原子加引号成短语, OR 连接。

    CJK 串按与索引侧同一套 bigram 展开; 串长分流(fts5_phrase_max_run, 默认 6):
    - 短串(术语形态, 如 区块链/图神经网络): bigram 短语原子, 连续命中等价于
      子串匹配, 保精度;
    - 长串(句子形态): 短语匹配是全有或全无(P6 收尾课修复的记账边界), 改拆
      bigram OR 词袋, 靠 BM25 排序保召回;
    - 单个 CJK 字: 前缀匹配 "字"* 兜底(索引里只有 bigram, 无 unigram)。
    """
    bigram = _bigram_enabled()
    max_run = int(cfg.load().retrieve.fts5_phrase_max_run)
    atoms: list[str] = []
    for token in _SANITIZE_RE.sub(" ", query).split():
        pos = 0
        for m in _CJK_RUN_RE.finditer(token):
            if m.start() > pos:
                atoms.append(f'"{token[pos : m.start()]}"')
            run = m.group(0)
            if not bigram:
                atoms.append(f'"{run}"')
            elif len(run) == 1:
                atoms.append(f'"{run}"*')
            elif len(run) <= max_run:
                atoms.append(f'"{segment_cjk(run)}"')
            else:
                atoms.extend(f'"{run[i : i + 2]}"' for i in range(len(run) - 1))
            pos = m.end()
        if pos < len(token):
            atoms.append(f'"{token[pos:]}"')
    return " OR ".join(atoms) if atoms else '""'


def search(query: str, top_k: int = 20, paper_ids: list[str] | None = None) -> list[dict]:
    """返回 top-k 命中, 取负 bm25 分数(越大越好), text 为 JOIN 回的原文。"""
    _ensure_table()
    _sync_if_stale()
    match_q = _build_match_query(query)
    if match_q == '""':
        return []

    sql = (
        "SELECT chunks_fts.chunk_id, chunks_fts.paper_id, chunks_fts.modality, "
        "  chunk.text, -bm25(chunks_fts) AS score "
        "FROM chunks_fts JOIN chunk ON chunk.chunk_id = chunks_fts.chunk_id "
        "WHERE chunks_fts MATCH :q "
    )
    params: dict = {"q": match_q}
    if paper_ids:
        placeholders = ",".join(f":p{i}" for i in range(len(paper_ids)))
        sql += f"  AND chunks_fts.paper_id IN ({placeholders}) "
        for i, pid in enumerate(paper_ids):
            params[f"p{i}"] = pid
    sql += "ORDER BY bm25(chunks_fts) LIMIT :k"
    params["k"] = top_k

    engine = _engine()
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(sql, params).fetchall()

    return [
        {
            "chunk_id": row[0],
            "paper_id": row[1],
            "modality": row[2],
            "text": row[3],
            "score_bm25": float(row[4]),
            "rank_bm25": rank,
        }
        for rank, row in enumerate(rows)
    ]
