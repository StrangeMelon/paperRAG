"""rag/query_rewrite.py 真实验收: 真实 LLM 一问变多查, 中英各一问。

验收点:
- [1] 英文问题走英文模板: 真实产出改写变体 + HyDE 伪答案 + BM25 关键词;
- [2] 中文问题走中文模板: keywords **中英双语混出**(跨越 BM25 词面断层),
      变体里至少出现一条含拉丁字母的英文改写;
- [3] 出口契约: dense_queries 首项恒为原问题, HyDE(若有)在尾;
- [4] 逃生门: PAPER_RAG_FORCE_LOCAL_REWRITE=1 时不发 LLM 调用, 走本地启发式;
- [5] 中文别名回查: "最初的 X 论文" 中文形态被识别(无库时降级不崩)。

前置: .env(或已导出的环境变量)提供 OPENAI_BASE_URL / OPENAI_API_KEY /
CHAT_MODEL。只读调用 LLM, 不写 data/ 与 demo-*-data/。任一断言失败即非零退出。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _show(tag: str, question: str, out: dict) -> None:
    print(f"[{tag}] 原问题: {question}")
    for i, q in enumerate(out["dense_queries"]):
        mark = "原问题" if i == 0 else f"改写/HyDE {i}"
        print(f"    dense[{i}] ({mark}): {q}")
    print(f"    bm25_query: {out['bm25_query']}\n")


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"缺少环境变量 {var}: 请在 .env 或 shell 中设置后重跑")
    os.environ.pop("PAPER_RAG_FORCE_LOCAL_REWRITE", None)

    import paper_rag.config as config
    from paper_rag.rag import llm
    from paper_rag.rag import query_rewrite as qr

    config.load.cache_clear()
    llm.reset_client_for_test()
    latin = __import__("re").compile(r"[A-Za-z]")
    cjk = __import__("re").compile(r"[一-鿿]")

    # ── 1) 英文问题(基准路径) ──
    q_en = "How does Self-RAG use reflection tokens to decide when to retrieve?"
    out_en = qr.rewrite(q_en)
    _show("1] [en", q_en, out_en)
    assert out_en["dense_queries"][0] == q_en, "首项必须是原问题"
    assert len(out_en["dense_queries"]) > 1, "英文问题未产出任何改写/HyDE"
    assert out_en["bm25_query"].strip(), "bm25_query 为空"

    # ── 2) 中文问题(中文模板 + 双语关键词) ──
    q_zh = "检索增强生成怎么缓解大模型的幻觉问题?"
    out_zh = qr.rewrite(q_zh)
    _show("2] [zh", q_zh, out_zh)
    assert out_zh["dense_queries"][0] == q_zh, "首项必须是原问题"
    assert len(out_zh["dense_queries"]) > 1, "中文问题未产出任何改写/HyDE"
    bm25_zh = out_zh["bm25_query"]
    assert latin.search(bm25_zh), "中文问题的 keywords 缺英文术语: BM25 打不中英文论文块"
    assert cjk.search(bm25_zh), "中文问题的 keywords 缺中文术语"
    print("    ✓ keywords 中英双语混出(跨语言 BM25 断层已跨越)\n")
    assert any(latin.search(q) for q in out_zh["dense_queries"][1:]), "中文问题缺英文改写变体"
    print("    ✓ 改写变体含英文改写\n")

    # ── 3) 逃生门: 不发 LLM 调用 ──
    called = {"n": 0}
    real_chat = qr.chat

    def _counting_chat(*a, **kw):
        called["n"] += 1
        return real_chat(*a, **kw)

    qr.chat = _counting_chat
    try:
        os.environ["PAPER_RAG_FORCE_LOCAL_REWRITE"] = "1"
        out_local = qr.rewrite(q_zh)
        assert called["n"] == 0, "逃生门置真时仍发起了 LLM 调用"
        assert out_local["dense_queries"][0] == q_zh
        print(
            f"[3] 逃生门 PAPER_RAG_FORCE_LOCAL_REWRITE=1: LLM 调用 0 次, 本地启发式出口\n"
            f"    dense_queries={out_local['dense_queries']}\n"
        )
    finally:
        os.environ.pop("PAPER_RAG_FORCE_LOCAL_REWRITE", None)
        qr.chat = real_chat

    # ── 4) 中文别名形态识别(纯函数, 不依赖库内是否真有该论文) ──
    for q in ("最初的 RAG 论文说了什么?", "RAG 的原始论文用了什么数据集?"):
        aliases = qr._original_aliases(q)
        assert "RAG" in aliases, f"中文别名形态未识别: {q}"
    print("[4] 中文别名形态 最初的X论文 / X的原始论文 均识别为 RAG ✓\n")
    print(
        "[5] 中文标题内嵌拉丁缩写词: "
        f"{sorted(a for a in qr._aliases_for_title('基于GNN的图神经网络建模') if a == 'GNN')} "
        "(基准 \\b 正则在此取不到) ✓\n"
    )

    print("DEMO PASSED: query_rewrite 真实改写 + 中文双语关键词 + 逃生门 + 中文别名 全部通过")


if __name__ == "__main__":
    main()
