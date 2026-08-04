"""查询改写与 HyDE——一问变多查。

给定用户问题, 产出:
  - 2~3 条改写变体(覆盖论文可能的不同措辞), 喂稠密检索
  - 1 条 HyDE 伪答案(先猜一段答案, 用答案的措辞去匹配论文正文——比问句更
    接近文档语言), 喂稠密检索
  - 一串关键词, 专供 BM25

与基准的偏离(中文语料扩展, 逐条有测试钉死):
  1. prompt 语言路由: `_query_language` 按 CJK 码位占比判 zh/en, 中文问题走中文
     模板。查询侧没有 `meta.json` 可依赖, 只能启发式判定。
  2. zh 模板要求 keywords **中英双语混出**并含 1 条英文变体: FTS5/BM25 是词面
     匹配, 纯中文关键词永远打不中英文论文块(稠密侧 BGE-M3 本身跨语言, 无需
     处理)。
  3. `_ORIGINAL_ALIAS_RE` 增中文形态: "最初/原始/最早的 X (论文)" 与 "X 的原
     (始)论文"。
  4. `_aliases_for_title` 的全大写缩写词正则改用显式 lookaround: Python `re` 把
     汉字算作 `\\w`, "基于GNN的" 中 于/G 之间没有 `\\b` 边界, 中文标题内嵌的拉丁
     缩写词提取不到。
"""

from __future__ import annotations

import json
import os
import re

from .. import config as cfg
from ..utils.logger import get_logger
from .llm import chat

log = get_logger("rag.query_rewrite")
_FORCE_LOCAL_REWRITE_ENV = "PAPER_RAG_FORCE_LOCAL_REWRITE"

_ORIGINAL_ALIAS_RE = re.compile(
    r"\b(?:the\s+)?(?:original|first|earliest)\s+([A-Za-z][A-Za-z0-9-]{1,20})"
    r"(?:\s+(?:paper|work|model))?",
    re.IGNORECASE,
)
# 中文形态一: "最初/原始/最早的 RAG 论文"(的/之 可省, 空格可有可无)。
_ORIGINAL_ALIAS_ZH_RE = re.compile(
    r"(?:最初|最早|原始|最开始)(?:的)?\s*([A-Za-z][A-Za-z0-9-]{1,20})",
)
# 中文形态二: "RAG 的原始论文 / RAG 的原论文"。
_ORIGINAL_ALIAS_ZH_SUFFIX_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9-]{1,20})\s*的(?:最初|最早|原始|原)(?:的)?(?:论文|工作|模型)",
)
_TITLE_WORD_RE = re.compile(r"[A-Za-z]+")
# 显式 lookaround 取代 \b: 汉字在 Python re 中算 \w, 会吃掉中文标题里的词边界。
_TITLE_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9-]{1,20}(?![A-Za-z0-9])")
_ACRONYM_STOPWORDS = {"a", "an", "and", "for", "of", "the", "to", "in", "on", "with"}
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_PROMPT = """You help an academic paper RAG system. Given a question, output a JSON
object with three keys:

  "variants":  array of 2-3 paraphrases that may match different wording in papers
  "keywords":  short string of 3-8 lowercase keywords (BM25 input)
  "hyde":      a 2-3 sentence hypothetical answer if you had to guess (used as
               an extra dense query). Be plausible; do NOT fabricate citations.

Question: {q}

Return only JSON.
"""

# 中文模板: 语料中英混排, BM25 是词面匹配, 故显式要求关键词中英双语混出、
# 变体里至少一条英文改写, 让中文提问也能命中英文论文块。
_PROMPT_ZH = """你在为一个学术论文 RAG 系统改写检索查询。针对下面的问题, 输出一个
JSON 对象, 含三个键:

  "variants":  2~3 条改写, 覆盖论文里可能的不同措辞; 其中**至少一条为英文改写**
               (语料含英文论文, 纯中文查询打不中英文原文)
  "keywords":  3~8 个关键词组成的短字符串, **中英文术语混合列出**(例如
               "检索增强生成 retrieval augmented generation 幻觉 hallucination"),
               供 BM25 词面匹配使用
  "hyde":      2~3 句假设性回答(如果必须猜一个答案), 用作额外的稠密检索查询;
               要写得像论文正文, **不要编造引用编号或文献名**

问题: {q}

只返回 JSON, 不要输出其他内容。
"""


def _query_language(question: str) -> str:
    """按 CJK 码位占比判定查询语言。

    查询侧没有 `meta.json` 的 language 字段可依赖, 只能靠字符启发式。中英混排
    (如 "RAG 的召回率怎么算?")只要出现 CJK 就按中文处理——中文模板本身要求
    产出中英双语关键词, 对混排问题是安全的一侧。
    """
    if not question:
        return "en"
    return "zh" if _CJK_RE.search(question) else "en"


def rewrite(question: str, wiki_context: dict | None = None) -> dict:
    c = cfg.load()
    enable = c.rag.enable_hyde
    data: dict = {}
    force_local = _truthy_env(_FORCE_LOCAL_REWRITE_ENV)
    if force_local:
        log.debug("rewrite forced to local fallback by PAPER_RAG_FORCE_LOCAL_REWRITE")
    elif c.llm.chat_model and c.llm.api_key and c.llm.base_url:
        lang = _query_language(question)
        template = _PROMPT_ZH if lang == "zh" else _PROMPT
        try:
            raw = chat(
                [{"role": "user", "content": template.replace("{q}", question)}],
                temperature=c.llm.temperatures.rewrite,
                max_tokens=400,
            )
            # 模型常在 JSON 前后加寒暄或 ```json 围栏(真实验收见 qwen3.8-max
            # 漂移记录), 用最外层大括号抠出对象。
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
            data = parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            log.warning(f"rewrite failed: {e}; using local fallback variants")
    else:
        log.debug("rewrite LLM not configured; using local fallback variants")

    wiki_hints = _wiki_hints(wiki_context)
    llm_variants = data.get("variants")
    if not isinstance(llm_variants, list):
        llm_variants = []
    variants = _dedupe(
        [
            *[str(v) for v in llm_variants if v],
            *_heuristic_variants(question),
            *wiki_hints["dense_queries"],
        ]
    )
    keywords_raw = data.get("keywords")
    keyword_parts = [str(keywords_raw) if keywords_raw else question]
    if wiki_hints["bm25_query"]:
        keyword_parts.append(wiki_hints["bm25_query"])
    keywords = " ".join(part for part in keyword_parts if part)
    hyde = data.get("hyde") if enable else None
    queries_dense = [question, *variants]
    if hyde:
        queries_dense.append(str(hyde))
    return {
        "dense_queries": _dedupe(queries_dense),
        "bm25_query": keywords,
        "raw": {
            **data,
            "wiki_context_used": bool(wiki_hints["dense_queries"]),
            "wiki_key_papers": wiki_hints["key_papers"],
        },
    }


def _wiki_hints(wiki_context: dict | None) -> dict:
    """wiki 概念提示。wiki/ 模块尚未重建时 try/except 降级, 与 vision 同款诚实信号。"""
    if not wiki_context:
        return {"dense_queries": [], "bm25_query": "", "key_papers": []}
    try:
        from ..wiki.context import wiki_rewrite_hints

        return wiki_rewrite_hints(wiki_context)
    except Exception as e:
        log.warning(f"wiki rewrite hints skipped: {e}")
        return {"dense_queries": [], "bm25_query": "", "key_papers": []}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _dedupe(items: list[str]) -> list[str]:
    """折叠空白 + 大小写归一去重, 保持首次出现的原文形态与顺序。"""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join((item or "").split())
        key = item.lower()
        if item and key not in seen:
            out.append(item)
            seen.add(key)
    return out


def _original_aliases(question: str) -> list[str]:
    """从问题中提取"最初的 X 论文"类别名(英文 + 两种中文形态)。"""
    aliases: list[str] = []
    for regex in (_ORIGINAL_ALIAS_RE, _ORIGINAL_ALIAS_ZH_RE, _ORIGINAL_ALIAS_ZH_SUFFIX_RE):
        for m in regex.finditer(question):
            aliases.append(m.group(1).upper())
    return list(dict.fromkeys(aliases))


def _heuristic_variants(question: str) -> list[str]:
    """LLM 不可用时的本地兜底改写。

    话题触发表照抄基准(演示语料专属启发式, 小写子串匹配对中文问题天然不触发,
    无害); 别名回查把 "the original RAG paper" / "最初的 RAG 论文" 映射回 SQLite
    里最早的同名论文标题。
    """
    variants: list[str] = []
    qlow = question.lower()
    if "recall@k" in qlow and "precision@k" in qlow and "retrieval" in qlow:
        variants.extend(
            [
                "RAG retrieval evaluation metrics recall precision retrieval stage",
                "retrieval augmented generation evaluation recall@k precision@k",
            ]
        )
    if "factscore" in qlow or "fact score" in qlow:
        variants.extend(
            [
                "Self-RAG FactScore biographies factuality metric atomic facts",
                "FactScore SELF-RAG evaluation factuality biographies",
                "FactScore factuality precision atomic facts knowledge source",
            ]
        )
    if "chunk" in qlow and ("size" in qlow or "embedding" in qlow or "production" in qlow):
        variants.extend(
            [
                "RAG survey chunking strategy 100 256 512 embedding chunks",
                "retrieval augmented generation chunk size embedding granularity",
                "chunking strategy larger chunks capture more context smaller chunks retrieval precision",
            ]
        )
    if "latency" in qlow and ("rerank" in qlow or "retrieval" in qlow or "rag" in qlow):
        variants.extend(
            [
                "retrieval latency reranking dense retriever latency cost",
                "BEIR retrieval latency reranking dense retrieval efficiency",
                "RAG higher latency retrieval augmentation rerank chunks",
            ]
        )
    if "rag-sequence" in qlow and "rag-token" in qlow:
        variants.extend(
            [
                "RAG-Sequence uses the same retrieved document for the whole sequence",
                "RAG-Token can use different retrieved documents for each target token",
                "RAG-Sequence RAG-Token latent documents per sequence per token model difference",
            ]
        )
    if "pre-retrieval" in qlow or "post-retrieval" in qlow:
        variants.extend(
            [
                "Advanced RAG pre-retrieval post-retrieval optimization strategies indexing retrieval generation",
                "RAG survey pre-retrieval optimization data indexing query optimization",
                "RAG survey post-retrieval processing reranking context compression optimization",
            ]
        )
    for alias in _original_aliases(question):
        try:
            papers = _papers_for_alias(alias)
        except Exception as e:  # sqlite 不可用时别名回查降级, 不拖垮主出口
            log.debug(f"paper alias lookup skipped: {e}")
            continue
        if not papers:
            continue
        p = sorted(papers, key=lambda x: (x.get("year") or 9999, x.get("title") or ""))[0]
        title = p.get("title")
        if title:
            variants.append(title)
            variants.append(f"{alias} original paper {title}")
    return _dedupe(variants)[:5]


def _papers_for_alias(alias: str) -> list[dict]:
    try:
        from sqlmodel import Session, select

        from ..store.sqlite_store import Paper, get_engine

        with Session(get_engine()) as s:
            papers = s.exec(select(Paper)).all()
    except Exception as e:
        log.debug(f"paper alias lookup skipped: {e}")
        return []

    out: list[dict] = []
    for p in papers:
        if alias in _aliases_for_title(p.title or ""):
            out.append({"title": p.title, "year": p.year, "arxiv_id": p.arxiv_id})
    return out


def _aliases_for_title(title: str) -> set[str]:
    """标题 -> 可能的缩写别名集合(首字母缩写窗口 + 标题内嵌的全大写词)。"""
    aliases: set[str] = set()
    tokens = [w for w in _TITLE_WORD_RE.findall(title) if w.lower() not in _ACRONYM_STOPWORDS]
    for i in range(len(tokens)):
        for n in range(2, 5):
            window = tokens[i : i + n]
            if len(window) != n:
                continue
            acronym = "".join(w[0].upper() for w in window)
            if 2 <= len(acronym) <= 8:
                aliases.add(acronym)
    for token in _TITLE_ACRONYM_RE.findall(title):
        aliases.add(token.upper())
    return aliases


__all__ = [
    "rewrite",
]
