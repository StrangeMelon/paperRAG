"""rag.qa_simple 单轮 QA 的行为契约测试(检索与 LLM 全打桩, 不发网络)。

切片 0: 空检索短路(zh/en 双语文案、四键 schema、不发 LLM 调用)。
切片 1: prompt 组装(系统模板 zh/en 路由、证据与问题注入、消息结构)。
切片 2: 引用管道(合法引用收集保序、编造 id 从答案与列表中剔除)。
切片 3: 可疑形态(count>0 才 strip; 干净答案不动; suspicious 报告透出)。
切片 4: 输出 schema(四键恒齐全, chunks 原样透传)。

接口约定(与基准一致):

    answer(question, *, top_k=8, paper_ids=None) -> dict
        {"answer": str, "citations": [id...], "chunks": [...],
         "suspicious_citations": {"numeric": [...], "author_year": [...],
                                  "count": int}}
"""

from __future__ import annotations

from paper_rag.rag import qa_simple as qs

_ID_A = "a3f09b2c17d4e8f0a1b2"
_ID_B = "b4e18c3d28e5f9a0b2c3"


def _chunk(cid: str) -> dict:
    return {
        "chunk_id": cid,
        "paper_id": "p1",
        "section": "Body",
        "modality": "text",
        "score": 0.9,
        "text": f"body of {cid}",
    }


class _FakeChat:
    def __init__(self, reply: str):
        self.calls: list[list[dict]] = []
        self._reply = reply

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return self._reply

    @property
    def system(self) -> str:
        return self.calls[0][0]["content"]

    @property
    def user(self) -> str:
        return self.calls[0][1]["content"]


def _stub(monkeypatch, *, chunks: list[dict], reply: str = "ok") -> _FakeChat:
    fake = _FakeChat(reply)
    monkeypatch.setattr(qs, "retrieve", lambda q, top_k=8, paper_ids=None: chunks)
    monkeypatch.setattr(qs, "chat", fake)
    return fake


# ---------- 切片 0: 空检索短路 ----------


def test_no_chunks_short_circuits_without_llm_call(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(qs, "retrieve", lambda q, top_k=8, paper_ids=None: [])
    monkeypatch.setattr(qs, "chat", _boom)
    out = qs.answer("What is X?")

    assert out["answer"] == "(no evidence found)"
    assert out["citations"] == [] and out["chunks"] == []
    assert out["suspicious_citations"]["count"] == 0


def test_no_chunks_chinese_question_gets_chinese_message(monkeypatch):
    monkeypatch.setattr(qs, "retrieve", lambda q, top_k=8, paper_ids=None: [])
    out = qs.answer("图神经网络的贡献是什么?")
    assert out["answer"] == "(未检索到证据)"


# ---------- 切片 1: prompt 组装 ----------


def test_english_question_uses_english_system_prompt(monkeypatch):
    fake = _stub(monkeypatch, chunks=[_chunk(_ID_A)])
    qs.answer("What does Graph-Mamba do?")
    assert "academic research assistant" in fake.system
    assert "[chunk:<chunk_id>]" in fake.system
    assert "NEVER" in fake.system


def test_chinese_question_uses_chinese_system_prompt(monkeypatch):
    fake = _stub(monkeypatch, chunks=[_chunk(_ID_A)])
    qs.answer("区块链网络架构是怎样的?")
    assert "证据" in fake.system
    assert "[chunk:<chunk_id>]" in fake.system
    assert "【1】" in fake.system, "中文系统 prompt 应明确禁止全角引用形态"


def test_user_prompt_carries_question_and_evidence(monkeypatch):
    fake = _stub(monkeypatch, chunks=[_chunk(_ID_A)])
    qs.answer("What does Graph-Mamba do?")
    assert "What does Graph-Mamba do?" in fake.user
    assert f"[chunk:{_ID_A}]" in fake.user, "证据块的引用令牌要进 prompt"


def test_message_structure_is_system_then_user(monkeypatch):
    fake = _stub(monkeypatch, chunks=[_chunk(_ID_A)])
    qs.answer("q")
    roles = [m["role"] for m in fake.calls[0]]
    assert roles == ["system", "user"]


# ---------- 切片 2: 引用管道 ----------


def test_valid_citations_collected_in_order(monkeypatch):
    reply = f"B first [chunk:{_ID_B}], then A [chunk:{_ID_A}]."
    _stub(monkeypatch, chunks=[_chunk(_ID_A), _chunk(_ID_B)], reply=reply)
    out = qs.answer("q")
    assert out["citations"] == [_ID_B, _ID_A]
    assert out["answer"] == reply


def test_fabricated_citation_dropped(monkeypatch):
    reply = f"Real [chunk:{_ID_A}]. Fake [chunk:ffffffffffffffffffff]."
    _stub(monkeypatch, chunks=[_chunk(_ID_A)], reply=reply)
    out = qs.answer("q")
    assert out["citations"] == [_ID_A]
    assert "ffffffff" not in out["answer"]


# ---------- 切片 3: 可疑形态 ----------


def test_suspicious_forms_reported_and_stripped(monkeypatch):
    reply = f"Claim [chunk:{_ID_A}] but also [1] and (Vaswani et al., 2017)."
    _stub(monkeypatch, chunks=[_chunk(_ID_A)], reply=reply)
    out = qs.answer("q")
    assert out["suspicious_citations"]["count"] == 2
    assert "[1]" not in out["answer"]
    assert "Vaswani" not in out["answer"]
    assert f"[chunk:{_ID_A}]" in out["answer"]


def test_clean_answer_not_stripped(monkeypatch):
    reply = f"Only good cites [chunk:{_ID_A}]."
    _stub(monkeypatch, chunks=[_chunk(_ID_A)], reply=reply)
    out = qs.answer("q")
    assert out["suspicious_citations"]["count"] == 0
    assert out["answer"] == reply


def test_fullwidth_suspicious_forms_stripped(monkeypatch):
    reply = f"结论 [chunk:{_ID_A}], 另见【3】与（张三等，2020）。"  # noqa: RUF001
    _stub(monkeypatch, chunks=[_chunk(_ID_A)], reply=reply)
    out = qs.answer("区块链?")
    assert out["suspicious_citations"]["count"] == 2
    assert "【3】" not in out["answer"]
    assert "张三" not in out["answer"]


# ---------- 切片 4: 输出 schema ----------


def test_output_schema_and_chunks_passthrough(monkeypatch):
    chunks = [_chunk(_ID_A), _chunk(_ID_B)]
    _stub(monkeypatch, chunks=chunks, reply=f"x [chunk:{_ID_A}]")
    out = qs.answer("q", top_k=2)
    assert set(out.keys()) == {"answer", "citations", "chunks", "suspicious_citations"}
    assert out["chunks"] == chunks
    assert set(out["suspicious_citations"].keys()) == {"numeric", "author_year", "count"}
