"""rag/llm.py 无 mock 真实集成测试(单独运行):

    uv run pytest -vv -s tests/test_llm_real.py

真实调用 DashScope(Qwen) OpenAI 兼容端点。按验收协议, 缺
OPENAI_BASE_URL / OPENAI_API_KEY / CHAT_MODEL 时**明确失败**, 不 skip。
仓库根目录若有 .env 会自动读入(不覆盖已导出的环境变量)。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
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


_load_dotenv(REPO_ROOT / ".env")


@pytest.fixture(autouse=True)
def _fresh_state():
    """真实测试前后清空配置缓存与客户端单例, 不污染同进程的其他用例。"""
    import paper_rag.config as config
    from paper_rag.rag import llm

    config.load.cache_clear()
    llm.reset_client_for_test()
    yield
    config.load.cache_clear()
    llm.reset_client_for_test()


def _require_llm_env() -> None:
    missing = [
        var
        for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL")
        if not os.environ.get(var)
    ]
    if missing:
        pytest.fail(
            f"真实 LLM 配置缺失: {', '.join(missing)}。请在 .env 或环境变量中设置后重跑"
            "(验收协议: 缺配置明确失败, 不 skip)。"
        )


def test_real_chat_nonstream_zh():
    """默认配置真实非流式调用: 中文问题拿到非空回复。"""
    _require_llm_env()
    from paper_rag.rag import llm

    reply = llm.chat(
        [{"role": "user", "content": "用一句话解释什么是图检索增强生成(GraphRAG)。"}],
        max_tokens=150,
    )
    print(f"\n[real] 默认配置回复: {reply.strip()}")
    assert isinstance(reply, str)
    assert reply.strip()


def test_real_extra_body_enable_thinking_false(tmp_path, monkeypatch):
    """extra_body={enable_thinking: false} 经临时配置真实透传, DashScope 接受。"""
    _require_llm_env()
    import paper_rag.config as config
    from paper_rag.rag import llm

    raw = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["llm"]["extra_body"] = {"enable_thinking": False}
    cfg_path = tmp_path / "llm_extra_body.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("PAPER_RAG_CONFIG", str(cfg_path))
    config.load.cache_clear()
    llm.reset_client_for_test()

    assert config.load().llm.extra_body == {"enable_thinking": False}
    reply = llm.chat(
        [{"role": "user", "content": "What is retrieval-augmented generation? One sentence."}],
        max_tokens=150,
    )
    print(f"\n[real] extra_body 透传回复: {reply.strip()}")
    assert isinstance(reply, str)
    assert reply.strip()
