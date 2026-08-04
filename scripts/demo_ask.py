"""scripts/ask.py 真实进程级验收: 以用户实际使用的形态运行 CLI。

与此前 Demo 的区别: 不是进程内调用引擎, 而是组装隔离数据 + 生成完整配置
文件, 经 `PAPER_RAG_CONFIG` 注入后用 subprocess 运行真实 CLI 进程, 断言
退出码与 stdout 结构——init_store -> ingest_one -> ask 三步闭环的收尾验收。

五次调用:
- [1] --no-llm(英文): 裸检索, 零 LLM;
- [2] 默认模式(英文): qa_simple, ANSWER/CITATIONS 标头;
- [3] --agentic(中文): 中文答案 + TRACE 摘要;
- [4] --stream(英文): 事件行 + 流式 ANSWER + CITATIONS;
- [5] --stream 域外("上海天气"): abstain 短路, 拒答文案流出, 仍 exit 0。

临时数据隔离在 demo-ask-data/; 任一断言失败即非零退出。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-ask-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _write_demo_config() -> Path:
    """基于 default.yaml 生成指向隔离数据的完整配置文件。"""
    raw = yaml.safe_load((REPO_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["paths"]["sqlite_path"] = str(DEMO_ROOT / "papers.sqlite")
    raw["qdrant"]["local_path"] = str(DEMO_ROOT / "qdrant")
    cfg_path = DEMO_ROOT / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return cfg_path


def _run_cli(cfg_path: Path, label: str, *cli_args: str) -> str:
    cmd = [sys.executable, "scripts/ask.py", *cli_args]
    env = {**os.environ, "PAPER_RAG_CONFIG": str(cfg_path)}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    shown = " ".join(a if " " not in a else f'"{a}"' for a in cmd[1:])
    print(f"$ PAPER_RAG_CONFIG=demo-ask-data/config.yaml python {shown}")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
    assert proc.returncode == 0, f"{label}: CLI 退出码 {proc.returncode}"
    return proc.stdout


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    if not (SRC_DATA / "qdrant").is_dir() or not ZH_CHUNKS.is_file():
        print("缺少存量产物(demo-ingest-pipeline-data 或中文期刊 chunks.json)", file=sys.stderr)
        raise SystemExit(1)
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"缺少环境变量 {var}: 请在 .env 或 shell 中设置后重跑")
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_DATA / "data/index/papers.sqlite", DEMO_ROOT / "papers.sqlite")
    shutil.copytree(SRC_DATA / "qdrant", DEMO_ROOT / "qdrant")
    cfg_path = _write_demo_config()

    # 中文期刊入库(进程内准备数据; CLI 验收本体在下方 subprocess)
    import paper_rag.config as config

    conf = config.load(cfg_path)
    config.load = lambda path=None: conf  # type: ignore[assignment]
    from paper_rag.embed import bge_m3
    from paper_rag.store import qdrant_store, sqlite_store

    zh_payload = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))
    zh_chunks = zh_payload["chunks"]
    sqlite_store.upsert_sections_and_chunks(
        zh_chunks[0]["paper_id"], zh_payload["sections"], zh_chunks
    )
    qdrant_store.upsert_chunks(zh_chunks, bge_m3.encode([c["context_text"] for c in zh_chunks]))
    qdrant_store.close_client()  # 释放 embedded Qdrant 目录锁, 否则 CLI 子进程读不到数据
    print(f"[0/5] 双语双库 + 独立配置就绪: {cfg_path}\n")

    q_en = "How does Graph-Mamba handle long-range dependencies?"
    q_zh = "综合能源服务里区块链的网络架构是怎样设计的"

    # ── 1) --no-llm: 裸检索 ──
    out = _run_cli(cfg_path, "no-llm", q_en, "--no-llm", "--top-k", "4")
    assert "[chunk:" in out and "ANSWER" not in out
    print(f"[1/5] --no-llm: 输出 {out.count('EVIDENCE CHUNK')} 个证据块, 零 LLM\n")

    # ── 2) 默认模式: qa_simple ──
    out = _run_cli(cfg_path, "default", q_en, "--top-k", "6")
    assert "=== ANSWER ===" in out and "=== CITATIONS (" in out
    assert "  - " in out, "应列出至少一条引用"
    print(f"[2/5] 默认(qa_simple): ANSWER/CITATIONS 结构成立\n{_tail(out)}\n")

    # ── 3) --agentic(中文): 中文答案 + TRACE 摘要 ──
    out = _run_cli(cfg_path, "agentic", q_zh, "--agentic")
    assert "=== ANSWER ===" in out and "=== TRACE ===" in out
    assert "intent=" in out and "abstain=confident" in out
    assert any("一" <= ch <= "鿿" for ch in out), "中文问题应得中文答案"
    print(f"[3/5] --agentic(中文): ANSWER + TRACE 摘要成立\n{_tail(out)}\n")

    # ── 4) --stream(英文): 事件行 + 流式 ANSWER ──
    out = _run_cli(cfg_path, "stream", q_en, "--stream")
    for marker in ("[intent]", "[abstain]", "=== ANSWER (streaming) ===", "=== CITATIONS ("):
        assert marker in out, f"stream 输出缺 {marker}"
    print("[4/5] --stream: 事件行 + 流式答案 + 引用清单成立\n")

    # ── 5) --stream 域外: abstain 短路 ──
    out = _run_cli(cfg_path, "stream-ood", "上海明天的天气怎么样", "--stream")
    assert "no_evidence" in out
    assert "未在已索引文献中找到" in out, "拒答文案应流出"
    assert "=== CITATIONS (0) ===" in out
    print("[5/5] --stream 域外: abstain 短路, 拒答文案流出, citations=0\n")

    print("DEMO PASSED: CLI 四模式进程级验收全部通过 (init_store -> ingest_one -> ask 闭环)")


def _tail(out: str, n: int = 6) -> str:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join("      " + ln for ln in lines[-n:])


if __name__ == "__main__":
    main()
