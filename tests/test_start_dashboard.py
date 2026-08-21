"""Dashboard launcher environment contracts."""

from __future__ import annotations


def test_load_dotenv_populates_missing_values_without_overwriting(monkeypatch, tmp_path) -> None:
    from scripts.start_dashboard import _load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_BASE_URL=https://example.test/v1\nOPENAI_API_KEY=file-key\nCHAT_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    monkeypatch.delenv("CHAT_MODEL", raising=False)

    _load_dotenv(env_file)

    import os

    assert os.environ["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert os.environ["OPENAI_API_KEY"] == "exported-key"
    assert os.environ["CHAT_MODEL"] == "file-model"
