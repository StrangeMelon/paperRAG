"""rag/llm.py 真实验收: DashScope(Qwen) OpenAI 兼容端点真实非流式调用。

验收点:
- 默认配置(extra_body 空表): chat() 真实调用返回非空回复(qwen-plus 稳定版
  非流式默认关思考, 主链路无需任何特殊参数);
- extra_body 透传: 临时配置把 llm.extra_body 设为 {enable_thinking: false},
  真实调用同样成功——证明确认偏离被 DashScope 端点接受(思考型模型非流式
  调用的 400 防御从此只是一行本地配置);
- 客户端单例: 两次调用之间 get_client() 返回同一对象(HTTP 连接复用);
- model 形参覆盖: 若设置了 SMALL_MODEL, 用它真实调用一次。

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


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"缺少环境变量 {var}: 请在 .env 或 shell 中设置后重跑")

    import paper_rag.config as config
    from paper_rag.rag import llm

    config.load.cache_clear()
    llm.reset_client_for_test()

    # ── 1) 默认配置真实非流式调用 ──
    question = [{"role": "user", "content": "用一句话解释什么是检索增强生成(RAG)。"}]
    client_before = llm.get_client()
    reply = llm.chat(question, max_tokens=150)
    assert isinstance(reply, str) and reply.strip(), "默认配置回复为空"
    print(f"[1] chat_model={os.environ['CHAT_MODEL']} 默认配置回复:\n    {reply.strip()}\n")

    # ── 2) 客户端单例(连接复用) ──
    assert llm.get_client() is client_before, "同配置下客户端未复用"
    print("[2] get_client() 两次调用返回同一对象: 单例复用 OK\n")

    # ── 3) extra_body 透传真实验证 ──
    raw = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["llm"]["extra_body"] = {"enable_thinking": False}
    tmp = Path(tempfile.mkstemp(prefix="demo_llm_qwen_", suffix=".yaml")[1])
    old_env = os.environ.get("PAPER_RAG_CONFIG")
    try:
        tmp.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        os.environ["PAPER_RAG_CONFIG"] = str(tmp)
        config.load.cache_clear()
        llm.reset_client_for_test()
        assert config.load().llm.extra_body == {"enable_thinking": False}
        reply2 = llm.chat(
            [{"role": "user", "content": "What is retrieval-augmented generation? One sentence."}],
            max_tokens=150,
        )
        assert isinstance(reply2, str) and reply2.strip(), "extra_body 配置下回复为空"
        print(
            f"[3] extra_body={{enable_thinking: false}} 透传被 DashScope 接受, 回复:\n    {reply2.strip()}\n"
        )
    finally:
        if old_env is None:
            os.environ.pop("PAPER_RAG_CONFIG", None)
        else:
            os.environ["PAPER_RAG_CONFIG"] = old_env
        tmp.unlink(missing_ok=True)
        config.load.cache_clear()
        llm.reset_client_for_test()

    # ── 4) model 形参覆盖(可选, 仅当设置了 SMALL_MODEL) ──
    small = os.environ.get("SMALL_MODEL")
    if small:
        reply3 = llm.chat(
            [{"role": "user", "content": "回复两个字: 收到"}], model=small, max_tokens=50
        )
        assert isinstance(reply3, str) and reply3.strip(), "SMALL_MODEL 回复为空"
        print(f"[4] model 形参覆盖 -> {small} 回复: {reply3.strip()}\n")
    else:
        print("[4] 未设置 SMALL_MODEL, 跳过 model 形参覆盖真实调用\n")

    print("DEMO PASSED: rag/llm.py 真实非流式调用 + extra_body 透传 + 单例复用 全部通过")


if __name__ == "__main__":
    main()
