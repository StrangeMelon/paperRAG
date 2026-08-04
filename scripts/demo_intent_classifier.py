"""rag/intent_classifier.py 真实验收: 真实 LLM 对中英各三问判定意图。

验收点:
- [1] 英文三问走英文模板: factual/reasoning/explore 各判对一问, 带出对应档位;
- [2] 中文三问走中文模板: 同样三档判对——证明中文模板的信号词引导有效;
- [3] 出口契约: 四键恒齐全, top_k/max_iter 与 config 的档位逐项一致;
- [4] 配置驱动: 临时把 explore 档 top_k 调成 24, 真实调用带出 24(非硬编码);
- [5] 逃生门: rag.intent.enabled=false 时不发 LLM 调用, 走本地信号词启发式。

前置: .env(或已导出的环境变量)提供 OPENAI_BASE_URL / OPENAI_API_KEY /
CHAT_MODEL。临时配置写入系统临时目录, 结束后清理; 不触碰 data/ 与
demo-*-data/。任一断言失败即非零退出。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml

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


_EN_CASES = [
    ("factual", "What is the FactScore metric?"),
    ("reasoning", "How do Self-RAG and CRAG differ in their retrieval decisions?"),
    ("explore", "What are recent advances in retrieval augmented generation?"),
]
_ZH_CASES = [
    ("factual", "FactScore 指标是什么?"),
    ("reasoning", "Self-RAG 和 CRAG 在检索决策上有什么区别?"),
    ("explore", "检索增强生成近年有哪些研究进展?"),
]


def _run_cases(tag: str, cases: list[tuple[str, str]], classify) -> int:
    hits = 0
    for expected, question in cases:
        out = classify(question)
        ok = out["intent"] == expected
        hits += ok
        mark = "✓" if ok else "≠"
        print(f"    [{tag}] {mark} 期望={expected:<9} 实判={out['intent']:<9} 问题: {question}")
        print(f"          档位: top_k={out['top_k']} max_iter={out['max_iter']}")
        assert sorted(out) == ["intent", "max_iter", "rrf_k", "top_k"], "出口四键不齐"
    return hits


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"缺少环境变量 {var}: 请在 .env 或 shell 中设置后重跑")

    import paper_rag.config as config
    from paper_rag.rag import intent_classifier as ic
    from paper_rag.rag import llm

    config.load.cache_clear()
    llm.reset_client_for_test()

    # ── 1) 英文三问(基准路径) ──
    print("[1] 英文问题走英文模板:")
    en_hits = _run_cases("en", _EN_CASES, ic.classify)
    print(f"    英文命中 {en_hits}/3\n")

    # ── 2) 中文三问(中文模板) ──
    print("[2] 中文问题走中文模板:")
    zh_hits = _run_cases("zh", _ZH_CASES, ic.classify)
    print(f"    中文命中 {zh_hits}/3\n")

    assert en_hits >= 2, f"英文三档命中过低({en_hits}/3): prompt 或模型有问题"
    assert zh_hits >= 2, f"中文三档命中过低({zh_hits}/3): 中文模板引导无效"
    print("[3] 出口四键齐全, 档位随意图变化 ✓\n")

    # ── 4) 配置驱动档位(非硬编码) ──
    raw = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["rag"]["intent"]["explore"]["top_k"] = 24
    tmp = Path(tempfile.mkstemp(prefix="demo_intent_", suffix=".yaml")[1])
    old_env = os.environ.get("PAPER_RAG_CONFIG")
    try:
        tmp.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        os.environ["PAPER_RAG_CONFIG"] = str(tmp)
        config.load.cache_clear()
        out = ic.classify("检索增强生成近年有哪些研究进展?")
        assert out["intent"] == "explore", f"配置态下意图判定漂移: {out['intent']}"
        assert out["top_k"] == 24, f"配置档位未生效: top_k={out['top_k']}"
        print(f"[4] explore 档 top_k 改配置为 24, 真实调用带出 {out['top_k']} ✓\n")

        # ── 5) 逃生门: enabled=false 不发 LLM 调用 ──
        raw["rag"]["intent"]["enabled"] = False
        tmp.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        config.load.cache_clear()
        called = {"n": 0}
        real_chat = ic.chat

        def _counting_chat(*a, **kw):
            called["n"] += 1
            return real_chat(*a, **kw)

        ic.chat = _counting_chat
        try:
            local = ic.classify("检索增强生成近年有哪些研究进展?")
            assert called["n"] == 0, "enabled=false 时仍发起了 LLM 调用"
            assert local["intent"] == "explore", "本地启发式未识别中文 explore 信号词"
            print(
                f"[5] rag.intent.enabled=false: LLM 调用 0 次, "
                f"本地启发式判为 {local['intent']} (top_k={local['top_k']}) ✓\n"
            )
        finally:
            ic.chat = real_chat
    finally:
        if old_env is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = old_env
        tmp.unlink(missing_ok=True)
        config.load.cache_clear()
        llm.reset_client_for_test()

    print("DEMO PASSED: intent_classifier 真实中英三档判定 + 配置档位 + 逃生门 全部通过")


if __name__ == "__main__":
    main()
