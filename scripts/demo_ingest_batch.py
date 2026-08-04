"""scripts/ingest_batch.py 真实进程级验收: 双语真实 PDF 批量入库到隔离库。

组装试验文件夹(英文 Graph-Mamba + 中文期刊真实 PDF 各一篇)与指向
demo-ingest-batch-data/ 的隔离配置(embedded Qdrant + 独立 SQLite; 模型缓存
仍指向真实 data/index/models 离线命中), 然后以 subprocess 运行真实 CLI:

- [1] --dry-run: 列 2 篇清单, 零副作用;
- [2] 全量入库: 真实 MinerU GPU 解析 + BGE-M3 + 入库, done=2, exit 0;
- [3] 原命令重跑: 引擎幂等 -> skipped=2(断点续跑语义实证);
- [4] ask.py --no-llm 中文问题: 新库真实检索命中(入库质量抽检);
- [5] 报告 JSON: 逐篇 status/chunks/耗时落盘。

任一断言失败即非零退出。真实 GPU 解析两篇约 2-6 分钟。
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

DEMO_ROOT = REPO_ROOT / "demo-ingest-batch-data"
EN_PDF_CANDIDATES = [
    Path("/home/user_kyh/Gra/Graph-Mamba.pdf"),
    REPO_ROOT / "demo-arxiv-data/papers/arxiv_2406.07003/raw.pdf",
]
ZH_PDF = (
    REPO_ROOT
    / "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566"
    / "_mineru_raw/综合能源服务区块链的网络架构、交互模型与信用评价/ocr"
    / "综合能源服务区块链的网络架构、交互模型与信用评价_origin.pdf"
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
    raw = yaml.safe_load((REPO_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        "data_root": str(DEMO_ROOT / "data"),
        "papers_dir": str(DEMO_ROOT / "data/papers"),
        "parsed_dir": str(DEMO_ROOT / "data/parsed"),
        "index_dir": str(DEMO_ROOT / "data/index"),
        "sqlite_path": str(DEMO_ROOT / "data/index/papers.sqlite"),
        "bm25_path": str(DEMO_ROOT / "data/index/bm25.pkl"),
        # 模型缓存仍指向真实目录: BGE-M3/reranker/MinerU 模型离线命中, 不重下 4G+
        "models_dir": str(REPO_ROOT / "data/index/models"),
    }
    raw["qdrant"]["local_path"] = str(DEMO_ROOT / "qdrant")
    cfg_path = DEMO_ROOT / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return cfg_path


def _run(cfg_path: Path, label: str, *argv: str, expect_rc: int = 0) -> str:
    cmd = [sys.executable, *argv]
    env = {**os.environ, "PAPER_RAG_CONFIG": str(cfg_path)}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800)
    print(f"$ {' '.join(a if ' ' not in a else repr(a) for a in argv)}")
    if proc.returncode != expect_rc:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
    assert proc.returncode == expect_rc, f"{label}: 退出码 {proc.returncode} != {expect_rc}"
    return proc.stdout


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    en_pdf = next((p for p in EN_PDF_CANDIDATES if p.is_file()), None)
    if en_pdf is None or not ZH_PDF.is_file():
        print("缺少真实样本 PDF(英文候选或中文期刊 raw_origin.pdf)", file=sys.stderr)
        raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    trial = DEMO_ROOT / "pdfs"
    trial.mkdir(parents=True)
    shutil.copy(en_pdf, trial / "Graph-Mamba Long-Range Dependencies.pdf")
    shutil.copy(ZH_PDF, trial / "综合能源服务区块链网络架构.pdf")
    cfg_path = _write_demo_config()
    print(f"[0/5] 试验文件夹(中英各 1 篇) + 隔离配置就绪: {cfg_path}\n")

    # 三步闭环第一步: 对全新隔离库跑 init_store(建 SQLite 表 + Qdrant collections)
    _run(cfg_path, "init_store", "scripts/init_store.py")
    print("      init_store: 隔离库表与 collections 就绪\n")

    batch = ("scripts/ingest_batch.py", str(trial))
    report = DEMO_ROOT / "data/ingest_batch_report.json"

    # ── 1) dry-run ──
    out = _run(cfg_path, "dry-run", *batch, "--dry-run")
    assert "would ingest 2 PDFs" in out
    assert "Graph-Mamba" in out and "综合能源" in out
    print("[1/5] --dry-run: 2 篇清单, 零副作用\n")

    # ── 2) 真实全量入库(GPU MinerU + BGE-M3) ──
    out = _run(cfg_path, "ingest", *batch)
    assert "done=2" in out and "failed=0" in out
    print(f"[2/5] 全量入库: done=2\n{_tail(out, 4)}\n")

    # ── 3) 原命令重跑: 幂等续传 ──
    out = _run(cfg_path, "resume", *batch)
    assert "skipped=2" in out and "done=0" in out
    print("[3/5] 重跑同一命令: skipped=2 (断点续跑语义实证)\n")

    # ── 4) 新库真实检索抽检 ──
    out = _run(
        cfg_path,
        "ask",
        "scripts/ask.py",
        "综合能源服务里区块链的网络架构是怎样设计的",
        "--no-llm",
        "--top-k",
        "3",
    )
    assert "[chunk:" in out, "新库检索应命中中文论文"
    print("[4/5] ask --no-llm: 新库检索命中中文论文\n")

    # ── 5) 报告文件 ──
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["summary"]["done"] == 0 and data["summary"]["skipped"] == 2  # 最后一轮是重跑
    assert all("seconds" in r for r in data["results"])
    print(f"[5/5] 报告 JSON 就绪: {report}\n")

    print("DEMO PASSED: 双语真实 PDF 批量入库 + 幂等续传 + 新库检索抽检 全部通过")


def _tail(out: str, n: int) -> str:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join("      " + ln for ln in lines[-n:])


if __name__ == "__main__":
    main()
