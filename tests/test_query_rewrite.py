"""rag.query_rewrite 一问变多查的行为契约测试(LLM 与 sqlite 全打桩, 不发网络)。

切片 0: 语言判定(_query_language 按 CJK 码位占比判 zh/en; 空串/纯符号/中英混排)。
切片 1: prompt 语言路由(zh 问题走中文模板并要求中英双语 keywords;
        en 走基准英文模板)。
切片 2: 脏输出鲁棒(JSON 带前后废话仍能抠出、非 JSON、缺键、LLM 抛异常
        ——四种都回退启发式而非崩溃; 真实验收发现 qwen3.8-max 会漂移话题)。
切片 3: 出口契约(dense_queries 首项恒为原问题、HyDE 追加在尾、
        enable_hyde=false 时不带 HyDE、bm25_query 非空、去重与大小写归一)。
切片 4: 未配置 LLM / 逃生门(base_url/api_key/chat_model 缺失或
        PAPER_RAG_FORCE_LOCAL_REWRITE 置真时不调 LLM, 走本地启发式)。
切片 5: 别名回查中文扩展(中文"最初/原始/最早的 X"形态; 中文标题内嵌拉丁
        缩写词的 CJK 邻接词边界修复——基准 \\b 正则在"基于Mamba的"取不到 Mamba)。
切片 6: wiki 钩子(hints 合并进 dense/bm25 并记账; 钩子异常时降级不崩)。

接口约定(与基准一致):

    rewrite(question: str, wiki_context: dict | None = None) -> dict
        {"dense_queries": [...], "bm25_query": "...", "raw": {...}}
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paper_rag.rag import query_rewrite as qr


def _conf(
    monkeypatch,
    *,
    base_url: str | None = "https://llm.example/v1",
    api_key: str | None = "sk-test",
    chat_model: str | None = "qwen-plus",
    enable_hyde: bool = True,
    rewrite_temp: float = 0.3,
):
    import paper_rag.config as config

    conf = SimpleNamespace(
        llm=SimpleNamespace(
            base_url=base_url,
            api_key=api_key,
            chat_model=chat_model,
            temperatures=SimpleNamespace(rewrite=rewrite_temp),
        ),
        rag=SimpleNamespace(enable_hyde=enable_hyde),
    )
    monkeypatch.setattr(config, "load", lambda path=None: conf)
    return conf


class _FakeChat:
    """记录 chat() 调用并按脚本返回, 不发网络。"""

    def __init__(self, reply: str | Exception):
        self.calls: list[dict] = []
        self._reply = reply

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    @property
    def prompt(self) -> str:
        return self.calls[0]["messages"][0]["content"]


def _stub_chat(monkeypatch, reply: str | Exception) -> _FakeChat:
    fake = _FakeChat(reply)
    monkeypatch.setattr(qr, "chat", fake)
    return fake


def _payload(**over) -> str:
    data = {
        "variants": ["variant one", "variant two"],
        "keywords": "rag retrieval augmented generation",
        "hyde": "A hypothetical answer about retrieval augmented generation.",
    }
    data.update(over)
    return json.dumps(data, ensure_ascii=False)


def _no_papers(monkeypatch):
    """默认切断 sqlite 别名回查, 避免用例依赖真实库。"""
    monkeypatch.setattr(qr, "_papers_for_alias", lambda alias: [])


# ---------- 切片 0: 语言判定 ----------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("什么是检索增强生成?", "zh"),
        ("What is retrieval augmented generation?", "en"),
        ("RAG 的召回率怎么算?", "zh"),  # 中英混排但中文为主
        ("Self-RAG reflection tokens", "en"),
        ("", "en"),  # 空串不猜, 落英文默认
        ("???  ...", "en"),  # 纯符号无 CJK
    ],
)
def test_query_language(question, expected):
    assert qr._query_language(question) == expected


# ---------- 切片 1: prompt 语言路由 ----------


def test_zh_question_uses_chinese_prompt_and_bilingual_keywords(monkeypatch):
    """中文问题走中文模板, 并显式要求 keywords 中英混出(跨越 BM25 词面断层)。"""
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    fake = _stub_chat(monkeypatch, _payload())

    qr.rewrite("什么是检索增强生成?")

    prompt = fake.prompt
    assert "什么是检索增强生成?" in prompt
    assert any("一" <= ch <= "鿿" for ch in prompt), "中文问题应使用中文模板"
    assert "英文" in prompt, "zh 模板必须要求中英双语 keywords / 英文变体"


def test_en_question_uses_english_prompt(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    fake = _stub_chat(monkeypatch, _payload())

    qr.rewrite("What is retrieval augmented generation?")

    prompt = fake.prompt
    assert "What is retrieval augmented generation?" in prompt
    assert "variants" in prompt and "keywords" in prompt and "hyde" in prompt
    assert not any("一" <= ch <= "鿿" for ch in prompt), "英文问题不应混入中文模板"


def test_rewrite_temperature_from_config(monkeypatch):
    _conf(monkeypatch, rewrite_temp=0.42)
    _no_papers(monkeypatch)
    fake = _stub_chat(monkeypatch, _payload())
    qr.rewrite("What is RAG?")
    assert fake.calls[0]["temperature"] == 0.42


# ---------- 切片 2: 脏输出鲁棒 ----------


def test_json_with_surrounding_prose(monkeypatch):
    """模型爱加"好的,以下是JSON:"之类前后废话, 正则应抠出对象。"""
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, f"好的,以下是结果:\n```json\n{_payload()}\n```\n希望有帮助!")

    out = qr.rewrite("What is RAG?")

    assert "variant one" in out["dense_queries"]
    assert "rag retrieval augmented generation" in out["bm25_query"]


@pytest.mark.parametrize(
    "reply",
    [
        "完全不是 JSON 的一段话。",
        "",
        '{"variants": "not-a-list"}',  # 类型不符
        "{ broken json ",
    ],
)
def test_dirty_reply_falls_back_without_raising(monkeypatch, reply):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, reply)

    out = qr.rewrite("What is retrieval augmented generation?")

    assert out["dense_queries"][0] == "What is retrieval augmented generation?"
    assert out["bm25_query"]


def test_llm_exception_falls_back(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, RuntimeError("429 rate limited"))

    out = qr.rewrite("什么是检索增强生成?")

    assert out["dense_queries"][0] == "什么是检索增强生成?"
    assert out["bm25_query"]


# ---------- 切片 3: 出口契约 ----------


def test_original_question_first_and_hyde_last(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(hyde="HYDE ANSWER TEXT"))

    out = qr.rewrite("What is RAG?")

    assert out["dense_queries"][0] == "What is RAG?"
    assert out["dense_queries"][-1] == "HYDE ANSWER TEXT"


def test_hyde_disabled_by_config(monkeypatch):
    _conf(monkeypatch, enable_hyde=False)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(hyde="HYDE ANSWER TEXT"))

    out = qr.rewrite("What is RAG?")

    assert "HYDE ANSWER TEXT" not in out["dense_queries"]


def test_dense_queries_deduped_case_insensitively(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(
        monkeypatch,
        _payload(variants=["What is RAG?", "what is rag?", "  What   is  RAG?  "], hyde=None),
    )

    out = qr.rewrite("What is RAG?")

    assert out["dense_queries"] == ["What is RAG?"]


def test_bm25_query_falls_back_to_question_when_keywords_missing(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(keywords=""))

    out = qr.rewrite("什么是检索增强生成?")

    assert out["bm25_query"] == "什么是检索增强生成?"


# ---------- 切片 4: 未配置 LLM / 逃生门 ----------


@pytest.mark.parametrize(
    "missing",
    [{"base_url": None}, {"api_key": None}, {"chat_model": None}],
)
def test_llm_not_configured_skips_call(monkeypatch, missing):
    _conf(monkeypatch, **missing)
    _no_papers(monkeypatch)
    fake = _stub_chat(monkeypatch, _payload())

    out = qr.rewrite("What is RAG?")

    assert fake.calls == [], "未配置 LLM 时不应发起调用"
    assert out["dense_queries"] == ["What is RAG?"]
    assert out["bm25_query"] == "What is RAG?"


def test_force_local_env_skips_llm(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    fake = _stub_chat(monkeypatch, _payload())
    monkeypatch.setenv("PAPER_RAG_FORCE_LOCAL_REWRITE", "1")

    out = qr.rewrite("What is RAG?")

    assert fake.calls == []
    assert out["dense_queries"] == ["What is RAG?"]


# ---------- 切片 5: 别名回查(含中文扩展) ----------


def test_english_original_alias_expands_to_title(monkeypatch):
    _conf(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))
    monkeypatch.setattr(
        qr,
        "_papers_for_alias",
        lambda alias: (
            [
                {"title": "Later RAG Work", "year": 2023},
                {"title": "Retrieval-Augmented Generation for NLP", "year": 2020},
            ]
            if alias == "RAG"
            else []
        ),
    )

    out = qr.rewrite("What does the original RAG paper say about latent documents?")

    assert "Retrieval-Augmented Generation for NLP" in out["dense_queries"], "应取最早的一篇"
    assert "Later RAG Work" not in out["dense_queries"]


@pytest.mark.parametrize(
    "question",
    [
        "最初的 RAG 论文说了什么?",
        "原始的 RAG 论文有哪些贡献?",
        "最早的 RAG 论文怎么做检索?",
        "RAG 的原始论文用了什么数据集?",
    ],
)
def test_chinese_original_alias_forms(monkeypatch, question):
    """中文"最初/原始/最早的 X 论文"与"X 的原始论文"两类形态都要识别。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))
    monkeypatch.setattr(
        qr,
        "_papers_for_alias",
        lambda alias: (
            [{"title": "Retrieval-Augmented Generation", "year": 2020}] if alias == "RAG" else []
        ),
    )

    out = qr.rewrite(question)

    assert "Retrieval-Augmented Generation" in out["dense_queries"]


def test_aliases_for_latin_acronym_inside_chinese_title():
    """基准缺陷: \\b[A-Z][A-Z0-9-]+\\b 在"基于GNN的"中, 因 Python re 把汉字算作 \\w,
    于/G 之间没有词边界 -> 中文标题内嵌的全大写缩写词提取不到。改用显式 lookaround。"""
    aliases = qr._aliases_for_title("基于GNN的图神经网络建模")
    assert "GNN" in aliases


def test_aliases_for_acronym_between_ascii_still_bounded():
    """修复不得放宽英文侧边界: 单词内部的大写片段不应被当作缩写词。"""
    aliases = qr._aliases_for_title("xxGNNyy retrieval")
    assert "GNN" not in aliases


def test_aliases_for_english_title_acronyms():
    aliases = qr._aliases_for_title("Retrieval-Augmented Generation for Knowledge Tasks")
    assert "RAG" in aliases
    assert "SELF" not in aliases  # 无该词, 防止过度生成


def test_alias_lookup_failure_is_non_fatal(monkeypatch):
    """sqlite 不可用时别名回查降级, 不影响主出口。"""
    _conf(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))

    def _boom(alias):
        raise RuntimeError("no database")

    monkeypatch.setattr(qr, "_papers_for_alias", _boom)

    out = qr.rewrite("最初的 RAG 论文说了什么?")

    assert out["dense_queries"][0] == "最初的 RAG 论文说了什么?"


# ---------- 切片 6: wiki 钩子 ----------


def test_wiki_hints_merged_and_accounted(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))
    monkeypatch.setattr(
        qr,
        "_wiki_hints",
        lambda ctx: {
            "dense_queries": ["self-reflective retrieval"],
            "bm25_query": "reflection tokens",
            "key_papers": ["p1"],
        },
    )

    out = qr.rewrite("What is Self-RAG?", wiki_context={"entries": [{"name": "Self-RAG"}]})

    assert "self-reflective retrieval" in out["dense_queries"]
    assert "reflection tokens" in out["bm25_query"]
    assert out["raw"]["wiki_context_used"] is True
    assert out["raw"]["wiki_key_papers"] == ["p1"]


def test_no_wiki_context_is_accounted_as_unused(monkeypatch):
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))

    out = qr.rewrite("What is RAG?")

    assert out["raw"]["wiki_context_used"] is False
    assert out["raw"]["wiki_key_papers"] == []


def test_wiki_module_available_hints_flow_through(monkeypatch):
    """wiki 模块已重建: 真实 wiki_rewrite_hints 生效, 词条名进入 dense 扩展。"""
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))

    out = qr.rewrite("What is RAG?", wiki_context={"entries": [{"name": "RAG"}]})

    assert out["dense_queries"][0] == "What is RAG?"
    assert "RAG" in out["dense_queries"]
    assert out["raw"]["wiki_context_used"] is True


def test_wiki_hint_exception_degrades(monkeypatch):
    """wiki 钩子异常时 try/except 降级(与 vision 同款诚实信号)。"""
    _conf(monkeypatch)
    _no_papers(monkeypatch)
    _stub_chat(monkeypatch, _payload(variants=[], hyde=None))
    import paper_rag.wiki.context as wctx

    def _boom(ctx):
        raise RuntimeError("wiki down")

    monkeypatch.setattr(wctx, "wiki_rewrite_hints", _boom)

    out = qr.rewrite("What is RAG?", wiki_context={"entries": [{"name": "RAG"}]})

    assert out["dense_queries"][0] == "What is RAG?"
    assert out["raw"]["wiki_context_used"] is False
