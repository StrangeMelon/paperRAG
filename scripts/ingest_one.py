"""单篇论文端到端入库。

Usage:
    python scripts/ingest_one.py --arxiv 2310.12345
    python scripts/ingest_one.py --pdf /path/to/paper.pdf --title "..."
    python scripts/ingest_one.py --pdf /path/to/paper.pdf --force   # 已 done 也重建

与基准一致(--arxiv/--pdf/--title/--force); 补 .env 自加载(ask 课教训:
CLI 独立进程没有 conftest 帮忙)与 argv 形参(可测性)。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_rag import config as cfg
from paper_rag.utils.logger import get_logger
from paper_rag.utils.paths import ensure_dirs

_REPO_ROOT = Path(__file__).resolve().parents[1]
log = get_logger("ingest_one")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a single paper.")
    p.add_argument("--arxiv", help="arXiv id, e.g. 2310.12345")
    p.add_argument("--pdf", help="Local PDF path")
    p.add_argument("--title", help="Title (used when --pdf without metadata)")
    p.add_argument("--force", action="store_true", help="Re-ingest even if status=done")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.arxiv or args.pdf):
        log.error("Need one of --arxiv / --pdf")
        return 2

    _load_dotenv(_REPO_ROOT / ".env")
    cfg.load()
    ensure_dirs()

    if args.arxiv:
        from paper_rag.ingest.arxiv_source import ArxivSource

        result = ArxivSource().fetch(args.arxiv)
    else:
        from paper_rag.ingest.local_source import LocalSource

        result = LocalSource(title=args.title).fetch(args.pdf)

    log.info(f"fetched: {result.meta.paper_id} title={result.meta.title!r}")

    from paper_rag.store.ingest_pipeline import ingest

    out = ingest(result, force=args.force)
    log.info(f"ingest result: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
