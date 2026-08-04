"""scripts/ask.py CLI 问答入口的行为契约测试(引擎全打桩, 不发网络)。

切片 0: 参数契约(缺省值、--paper-id append、三模式互斥)。
切片 1: 模式分发(--no-llm 裸检索 / 默认 qa_simple / --agentic / --stream,
        断言只有目标路径被调用)。
切片 2: 输出结构(ANSWER/CITATIONS 标头、agentic trace 摘要、stream 打字机
        文本与 done 明细)与退出码(stream error 事件 -> 1)。
"""

from __future__ import annotations

import importlib

import pytest

ask = importlib.import_module("scripts.ask")

_ID_A = "a3f09b2c17d4e8f0a1b2"


def _qa_out(answer: str = "the answer", citations: list | None = None) -> dict:
    return {
        "answer": answer,
        "citations": citations if citations is not None else [_ID_A],
        "chunks": [],
        "suspicious_citations": {"numeric": [], "author_year": [], "count": 0},
    }


@pytest.fixture(autouse=True)
def _no_engine(monkeypatch):
    """缺省把三条 QA 路径与裸检索都设为'不许被调用', 各用例按需放行。"""

    def _boom(name):
        def _f(*a, **k):
            raise AssertionError(f"{name} 不应被调用")

        return _f

    monkeypatch.setattr("paper_rag.rag.qa_simple.answer", _boom("qa_simple"), raising=True)
    monkeypatch.setattr("paper_rag.rag.qa_agentic.answer", _boom("qa_agentic"), raising=True)
    monkeypatch.setattr("paper_rag.rag.qa_stream.stream_answer", _boom("qa_stream"), raising=True)
    monkeypatch.setattr("paper_rag.retrieve.dense.retrieve", _boom("dense"), raising=True)


# ---------- 切片 0: 参数契约 ----------


def test_defaults():
    args = ask.parse_args(["What is X?"])
    assert args.question == "What is X?"
    assert args.top_k == 8
    assert args.paper_id is None
    assert not args.no_llm and not args.agentic and not args.stream


def test_paper_id_appends():
    args = ask.parse_args(["q", "--paper-id", "p1", "--paper-id", "p2", "--top-k", "3"])
    assert args.paper_id == ["p1", "p2"]
    assert args.top_k == 3


def test_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as e:
        ask.parse_args(["q", "--no-llm", "--stream"])
    assert e.value.code == 2


# ---------- 切片 1+2: 模式分发与输出 ----------


def test_no_llm_prints_evidence_only(monkeypatch, capsys):
    chunks = [{"chunk_id": _ID_A, "paper_id": "p1", "section": "Body", "text": "evidence body"}]
    monkeypatch.setattr(
        "paper_rag.retrieve.dense.retrieve", lambda q, top_k=8, paper_ids=None: chunks
    )
    rc = ask.main(["What is X?", "--no-llm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"[chunk:{_ID_A}]" in out
    assert "ANSWER" not in out


def test_default_mode_calls_qa_simple(monkeypatch, capsys):
    calls = {}

    def _answer(question, *, top_k=8, paper_ids=None):
        calls.update(question=question, top_k=top_k, paper_ids=paper_ids)
        return _qa_out()

    monkeypatch.setattr("paper_rag.rag.qa_simple.answer", _answer)
    rc = ask.main(["What is X?", "--top-k", "5", "--paper-id", "p1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == {"question": "What is X?", "top_k": 5, "paper_ids": ["p1"]}
    assert "=== ANSWER ===" in out
    assert "the answer" in out
    assert "=== CITATIONS (1) ===" in out
    assert _ID_A in out


def test_agentic_mode_prints_trace_summary(monkeypatch, capsys):
    out_dict = _qa_out("agentic answer")
    out_dict["trace"] = {
        "intent": {"intent": "reasoning"},
        "iters": [{}, {}],
        "stopped_by": "answered",
        "abstain": {"decision": "confident", "evidence_score": 0.99},
        "loop": {"latency_ms": 1234},
    }
    monkeypatch.setattr("paper_rag.rag.qa_agentic.answer", lambda q, **kw: out_dict)
    rc = ask.main(["What is X?", "--agentic"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "agentic answer" in out
    assert "=== TRACE ===" in out
    assert "intent=reasoning" in out
    assert "iters=2" in out
    assert "abstain=confident(0.99)" in out
    assert "latency=1234ms" in out


def test_stream_mode_renders_events(monkeypatch, capsys):
    events = [
        {"event": "intent", "data": {"intent": "factual", "top_k": 5, "max_iter": 1}},
        {"event": "rewrite", "data": {"queries": ["q1", "q2"], "keywords": "kw"}},
        {"event": "retrieved", "data": {"iter": 0, "n_chunks": 3}},
        {
            "event": "abstain",
            "data": {"decision": "confident", "evidence_score": 0.9, "score_field": "score_rerank"},
        },
        {"event": "answer_chunk", "data": {"text": "Hello "}},
        {"event": "answer_chunk", "data": {"text": "world"}},
        {
            "event": "done",
            "data": {"answer": "Hello world", "citations": [_ID_A], "suspicious": {"count": 0}},
        },
    ]
    monkeypatch.setattr(
        "paper_rag.rag.qa_stream.stream_answer", lambda q, paper_ids=None: iter(events)
    )
    rc = ask.main(["What is X?", "--stream"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Hello world" in out
    assert "[intent]" in out and "[abstain]" in out
    assert _ID_A in out


def test_stream_error_event_returns_nonzero(monkeypatch, capsys):
    events = [
        {"event": "intent", "data": {"intent": "factual", "top_k": 5, "max_iter": 1}},
        {"event": "error", "data": {"message": "retrieve failed: qdrant down"}},
    ]
    monkeypatch.setattr(
        "paper_rag.rag.qa_stream.stream_answer", lambda q, paper_ids=None: iter(events)
    )
    rc = ask.main(["What is X?", "--stream"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "qdrant down" in out


# ---------- 切片 3: CLI 自加载 .env(用户直跑修复) ----------


def test_load_dotenv_sets_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nCHAT_MODEL=from-dotenv\nOPENAI_API_KEY='sk-quoted'\nEXISTING=dotenv-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.setenv("EXISTING", "shell-value")
    import os

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ask._load_dotenv(env_file)
    assert os.environ["CHAT_MODEL"] == "from-dotenv"
    assert os.environ["OPENAI_API_KEY"] == "sk-quoted", "引号应剥除"
    assert os.environ["EXISTING"] == "shell-value", "已导出的变量不得被覆盖"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    ask._load_dotenv(tmp_path / "no-such.env")  # 不抛异常即可
