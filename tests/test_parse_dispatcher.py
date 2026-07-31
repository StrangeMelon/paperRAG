"""解析调度器的后端降级、空结果拒绝与单篇失败隔离边界测试。

约定:
- MinerU 后端在边界测试中用假实现替换(真实 GPU OCR 已在 Task 14 单独验收);
- PyMuPDF 降级链路使用真实 PyMuPDF 和真实临时 PDF, 不做 mock;
- 每条路径都必须在 ``parsed/<paper_id>/parse_status.json`` 留下可审计状态。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from paper_rag.utils.ids import to_safe_dirname

PAPER_ID = "arxiv:2401.00001"


def _dispatcher_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.parse.dispatcher")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.parse.dispatcher":
            raise  # 调度器存在, 但它内部导入的模块缺失, 原样抛出真正的错误
        pytest.fail(
            "尚未实现 paper_rag.parse.dispatcher.parse_pdf",
            pytrace=False,
        )


def _mineru_module() -> ModuleType:
    return importlib.import_module("paper_rag.parse.mineru_local")


def _fallback_module() -> ModuleType:
    return importlib.import_module("paper_rag.parse.fallback_pymupdf")


def _fitz_module() -> ModuleType:
    try:
        import fitz
    except ImportError:
        pytest.fail(
            "PyMuPDF 未安装; 请执行: uv sync --extra dev --extra ingest",
            pytrace=False,
        )
    return fitz


def _write_pdf(pdf_path: Path, text: str) -> None:
    """生成真实 PDF; ``text`` 为空时模拟没有文字层的扫描件。"""

    fitz = _fitz_module()
    document = fitz.open()
    try:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=12)
        document.save(str(pdf_path))
    finally:
        document.close()


@pytest.fixture()
def parsed_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """把调度器和 PyMuPDF 后端的解析输出根目录重定向到临时目录。"""

    root = tmp_path / "parsed"

    def resolve(received_id: str) -> Path:
        return root / to_safe_dirname(received_id)

    monkeypatch.setattr(_dispatcher_module(), "parsed_dir", resolve)
    monkeypatch.setattr(_fallback_module(), "parsed_dir", resolve)
    return root


def _expected_dir(parsed_root: Path, paper_id: str = PAPER_ID) -> Path:
    return parsed_root / to_safe_dirname(paper_id)


def _use_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "local",
    fallback_to_pymupdf: bool = True,
) -> None:
    """替换应用配置, 只暴露调度器需要的 MinerU 字段。"""

    module = _dispatcher_module()
    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: SimpleNamespace(
            mineru=SimpleNamespace(
                mode=mode,
                fallback_to_pymupdf=fallback_to_pymupdf,
            )
        ),
    )


def _fake_mineru_success(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    *,
    markdown: str,
) -> None:
    """让 MinerU 后端"成功"返回, 并写出指定内容的标准化 Markdown。"""

    def _parse_pdf(paper_id: str, pdf_path: str | Path) -> Path:
        output_dir = _expected_dir(parsed_root, paper_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper.md").write_text(markdown, encoding="utf-8")
        return output_dir

    monkeypatch.setattr(_mineru_module(), "parse_pdf", _parse_pdf)


def _fake_mineru_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    detail: str = "CUDA out of memory",
) -> None:
    """让 MinerU 后端抛出真实的领域异常 ``MineruError``。"""

    mineru = _mineru_module()

    def _parse_pdf(paper_id: str, pdf_path: str | Path) -> Path:
        raise mineru.MineruError(detail)

    monkeypatch.setattr(mineru, "parse_pdf", _parse_pdf)


def _read_status(output_dir: Path) -> dict[str, object]:
    payload = json.loads(
        (output_dir / "parse_status.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def test_parse_error_is_a_runtime_error() -> None:
    module = _dispatcher_module()

    assert issubclass(module.ParseError, RuntimeError)


def test_mineru_success_returns_mineru_backend(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    module = _dispatcher_module()
    pdf_path = tmp_path / "paper.pdf"
    _write_pdf(pdf_path, "Retrieval augmented generation.")
    _use_config(monkeypatch)
    _fake_mineru_success(
        monkeypatch,
        parsed_root,
        markdown="# Self-RAG\n\n真实正文段落。",
    )
    monkeypatch.setattr(
        _fallback_module(),
        "parse_pdf",
        lambda *_args, **_kwargs: pytest.fail(
            "MinerU 成功时不得调用 PyMuPDF 降级",
            pytrace=False,
        ),
    )

    output_dir, parser_name = module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(output_dir)
    assert output_dir == _expected_dir(parsed_root)
    assert parser_name == "mineru"
    assert status["paper_id"] == PAPER_ID
    assert status["status"] == "succeeded"
    assert status["parser"] == "mineru"


def test_mineru_failure_degrades_to_pymupdf_with_real_text_pdf(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    module = _dispatcher_module()
    pdf_path = tmp_path / "text.pdf"
    _write_pdf(pdf_path, "Dense and sparse retrieval fusion.")
    _use_config(monkeypatch)
    _fake_mineru_failure(monkeypatch, detail="CUDA out of memory")

    output_dir, parser_name = module.parse_pdf(PAPER_ID, pdf_path)

    markdown = (output_dir / "paper.md").read_text(encoding="utf-8")
    status = _read_status(output_dir)
    assert parser_name == "pymupdf"
    assert "Dense and sparse retrieval fusion." in markdown
    assert status["status"] == "degraded"
    assert status["parser"] == "pymupdf"
    assert "CUDA out of memory" in str(status["reason"])


def test_mineru_empty_markdown_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    """MinerU 返回只有页标记的空结果时, 不得伪装成 succeeded。"""

    module = _dispatcher_module()
    pdf_path = tmp_path / "text.pdf"
    _write_pdf(pdf_path, "Reflective retrieval loop.")
    _use_config(monkeypatch)
    _fake_mineru_success(
        monkeypatch,
        parsed_root,
        markdown="<!-- page 1 -->\n\n   \n",
    )

    output_dir, parser_name = module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(output_dir)
    assert parser_name == "pymupdf"
    assert status["status"] == "degraded"
    assert status["parser"] == "pymupdf"
    assert str(status["reason"]).strip() != ""


def test_scanned_pdf_without_text_layer_fails_with_parse_error(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    """扫描件 MinerU 失败且 PyMuPDF 只有页标记时必须失败, 而不是空结果成功。"""

    module = _dispatcher_module()
    pdf_path = tmp_path / "scan.pdf"
    _write_pdf(pdf_path, "")
    _use_config(monkeypatch)
    _fake_mineru_failure(monkeypatch, detail="layout model checkpoint not found")

    with pytest.raises(module.ParseError):
        module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(_expected_dir(parsed_root))
    assert status["paper_id"] == PAPER_ID
    assert status["status"] == "failed"
    assert status["parser"] == "pymupdf"
    assert str(status["reason"]).strip() != ""


def test_pymupdf_exception_is_wrapped_in_parse_error(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    module = _dispatcher_module()
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf at all")
    _use_config(monkeypatch)
    _fake_mineru_failure(monkeypatch)

    def _raise(paper_id: str, received_path: str | Path) -> Path:
        raise ValueError("cannot open broken document")

    monkeypatch.setattr(_fallback_module(), "parse_pdf", _raise)

    with pytest.raises(module.ParseError) as excinfo:
        module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(_expected_dir(parsed_root))
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert status["status"] == "failed"
    assert status["parser"] == "pymupdf"
    assert "ValueError" in str(status["reason"])


def test_disabled_fallback_reraises_mineru_error(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    module = _dispatcher_module()
    mineru = _mineru_module()
    pdf_path = tmp_path / "text.pdf"
    _write_pdf(pdf_path, "Hybrid retrieval with reciprocal rank fusion.")
    _use_config(monkeypatch, fallback_to_pymupdf=False)
    _fake_mineru_failure(monkeypatch, detail="mineru cli exited with code 1")
    monkeypatch.setattr(
        _fallback_module(),
        "parse_pdf",
        lambda *_args, **_kwargs: pytest.fail(
            "禁用降级时不得调用 PyMuPDF",
            pytrace=False,
        ),
    )

    with pytest.raises(mineru.MineruError):
        module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(_expected_dir(parsed_root))
    assert status["status"] == "failed"
    assert status["parser"] == "mineru"
    assert "mineru cli exited with code 1" in str(status["reason"])


def test_non_local_mineru_mode_uses_pymupdf_as_primary_backend(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    """未启用本地 MinerU 时直接走 PyMuPDF, 状态是 succeeded 而不是 degraded。"""

    module = _dispatcher_module()
    pdf_path = tmp_path / "text.pdf"
    _write_pdf(pdf_path, "Evidence selection and citation check.")
    _use_config(monkeypatch, mode="disabled")
    monkeypatch.setattr(
        _mineru_module(),
        "parse_pdf",
        lambda *_args, **_kwargs: pytest.fail(
            "mineru.mode 非 local 时不得调用 MinerU",
            pytrace=False,
        ),
    )

    output_dir, parser_name = module.parse_pdf(PAPER_ID, pdf_path)

    status = _read_status(output_dir)
    assert parser_name == "pymupdf"
    assert "Evidence selection and citation check." in (
        output_dir / "paper.md"
    ).read_text(encoding="utf-8")
    assert status["status"] == "succeeded"
    assert status["parser"] == "pymupdf"
    assert str(status["reason"]) == ""


def test_batch_continues_after_single_paper_failure(
    monkeypatch: pytest.MonkeyPatch,
    parsed_root: Path,
    tmp_path: Path,
) -> None:
    """单篇扫描件失败必须被隔离, 后一篇论文仍能正常解析。"""

    module = _dispatcher_module()
    scanned_pdf = tmp_path / "scan.pdf"
    text_pdf = tmp_path / "text.pdf"
    _write_pdf(scanned_pdf, "")
    _write_pdf(text_pdf, "Second paper parses normally.")
    _use_config(monkeypatch)
    _fake_mineru_failure(monkeypatch, detail="mineru timeout")

    failed_id = "local:scanned"
    succeeded_id = "local:text"
    results: dict[str, str] = {}

    for paper_id, pdf_path in ((failed_id, scanned_pdf), (succeeded_id, text_pdf)):
        try:
            _, parser_name = module.parse_pdf(paper_id, pdf_path)
        except module.ParseError:
            results[paper_id] = "failed"
        else:
            results[paper_id] = parser_name

    assert results == {failed_id: "failed", succeeded_id: "pymupdf"}
    assert _read_status(_expected_dir(parsed_root, failed_id))["status"] == "failed"
    assert _read_status(_expected_dir(parsed_root, succeeded_id))["status"] == "degraded"
