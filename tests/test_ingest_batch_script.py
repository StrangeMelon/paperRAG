"""scripts/ingest_batch.py 批量入库的行为契约测试(引擎全打桩, 不发网络)。

切片 0: 扫描契约(平铺 *.pdf 大小写不敏感、排序稳定、非 PDF 忽略;
        目录不存在/无 PDF -> rc 2)。
切片 1: --dry-run 只列清单零引擎调用; --limit 截断。
切片 2: 逐篇隔离(单篇抛异常不中断整批, 计入 failed) 与状态汇总
        (done/skipped/failed), 任一 failed -> rc 1。
切片 3: 报告文件(逐篇结果 JSON 落盘, 失败含错误信息); --force 透传;
        标题取文件名去扩展名。
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

ingest_batch = importlib.import_module("scripts.ingest_batch")


def _mk_pdfs(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_bytes(b"%PDF-1.4 fake")
    return tmp_path


def _stub_engine(monkeypatch, *, fail_on: set[str] | None = None, status: str = "done"):
    """LocalSource+ingest 桩; fail_on 中的文件名在 fetch 阶段抛异常。"""
    seen: list[dict] = []
    fail_on = fail_on or set()

    class _FakeLocal:
        def __init__(self, title=None):
            self.title = title

        def fetch(self, identifier):
            name = identifier.rsplit("/", 1)[-1]
            if name in fail_on:
                raise RuntimeError(f"parse boom: {name}")
            seen.append({"pdf": identifier, "title": self.title})
            return SimpleNamespace(
                meta=SimpleNamespace(paper_id=f"id-{name}", title=self.title),
                pdf_path=identifier,
            )

    def _ingest(result, *, force=False):
        seen[-1]["force"] = force
        return {"paper_id": result.meta.paper_id, "status": status, "chunks": 5}

    monkeypatch.setattr("paper_rag.ingest.local_source.LocalSource", _FakeLocal)
    monkeypatch.setattr("paper_rag.store.ingest_pipeline.ingest", _ingest)
    return seen


# ---------- 切片 0: 扫描契约 ----------


def test_missing_dir_returns_2(tmp_path):
    assert ingest_batch.main([str(tmp_path / "nope")]) == 2


def test_dir_without_pdfs_returns_2(tmp_path):
    (tmp_path / "note.txt").write_text("x")
    assert ingest_batch.main([str(tmp_path)]) == 2


def test_scan_case_insensitive_sorted(tmp_path, monkeypatch, capsys):
    _mk_pdfs(tmp_path, "b.pdf", "A.PDF", "c.txt")
    rc = ingest_batch.main([str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.index("A.PDF") < out.index("b.pdf")
    assert "c.txt" not in out


# ---------- 切片 1: dry-run 与 limit ----------


def test_dry_run_makes_no_engine_calls(tmp_path, monkeypatch, capsys):
    _mk_pdfs(tmp_path, "a.pdf")

    def _boom(*a, **k):
        raise AssertionError("dry-run 不应触碰引擎")

    monkeypatch.setattr("paper_rag.ingest.local_source.LocalSource", _boom)
    rc = ingest_batch.main([str(tmp_path), "--dry-run"])
    assert rc == 0
    assert "a.pdf" in capsys.readouterr().out


def test_limit_truncates(tmp_path, monkeypatch):
    _mk_pdfs(tmp_path, "a.pdf", "b.pdf", "c.pdf")
    seen = _stub_engine(monkeypatch)
    rc = ingest_batch.main([str(tmp_path), "--limit", "2"])
    assert rc == 0
    assert len(seen) == 2


# ---------- 切片 2: 逐篇隔离与汇总 ----------


def test_single_failure_does_not_abort_batch(tmp_path, monkeypatch, capsys):
    _mk_pdfs(tmp_path, "a.pdf", "b.pdf", "c.pdf")
    seen = _stub_engine(monkeypatch, fail_on={"b.pdf"})
    rc = ingest_batch.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1, "有失败时退出码应为 1"
    assert len(seen) == 2, "b.pdf 失败后 c.pdf 仍应继续"
    assert "done=2" in out and "failed=1" in out


def test_all_ok_returns_0(tmp_path, monkeypatch, capsys):
    _mk_pdfs(tmp_path, "a.pdf", "b.pdf")
    _stub_engine(monkeypatch)
    rc = ingest_batch.main([str(tmp_path)])
    assert rc == 0
    assert "done=2" in capsys.readouterr().out


def test_skipped_counts_separately(tmp_path, monkeypatch, capsys):
    _mk_pdfs(tmp_path, "a.pdf")
    _stub_engine(monkeypatch, status="skipped")
    rc = ingest_batch.main([str(tmp_path)])
    assert rc == 0
    assert "skipped=1" in capsys.readouterr().out


# ---------- 切片 3: 报告/透传/标题 ----------


def test_report_written_with_failure_details(tmp_path, monkeypatch):
    _mk_pdfs(tmp_path, "a.pdf", "b.pdf")
    _stub_engine(monkeypatch, fail_on={"b.pdf"})
    report = tmp_path / "report.json"
    ingest_batch.main([str(tmp_path), "--report", str(report)])
    data = json.loads(report.read_text(encoding="utf-8"))
    by_file = {r["file"].rsplit("/", 1)[-1]: r for r in data["results"]}
    assert by_file["a.pdf"]["status"] == "done"
    assert by_file["b.pdf"]["status"] == "failed"
    assert "parse boom" in by_file["b.pdf"]["error"]
    assert data["summary"]["failed"] == 1


def test_force_and_filename_title_passthrough(tmp_path, monkeypatch):
    _mk_pdfs(tmp_path, "Graph-Mamba Long Range.pdf")
    seen = _stub_engine(monkeypatch)
    ingest_batch.main([str(tmp_path), "--force"])
    assert seen[0]["title"] == "Graph-Mamba Long Range", "标题取文件名去扩展名"
    assert seen[0]["force"] is True
