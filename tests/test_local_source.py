"""本地 PDF 采集器的边界行为测试。"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from paper_rag.ingest.schema import FetchResult
from paper_rag.utils.ids import make_paper_id, to_safe_dirname


def _local_source_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.ingest.local_source")
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.ingest.local_source":
            raise
        pytest.fail("尚未实现 paper_rag.ingest.local_source.LocalSource", pytrace=False)


def _isolate_paper_storage(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    storage_root: Path,
) -> None:
    monkeypatch.setattr(
        module,
        "paper_dir",
        lambda paper_id: storage_root / to_safe_dirname(paper_id),
    )

# 在 pytest 的临时目录中创建一个"测试用 PDF 文件", 供 LocalSource 做复制、哈希和落盘测试
def _write_source_pdf(tmp_path: Path, name: str = "example-paper.pdf") -> Path:
    source_pdf = tmp_path / "incoming" / name
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.7\nlocal source boundary test\n%%EOF\n")
    return source_pdf


# 当用户传入一个不存在的本地 PDF 路径时, LocalSource.fetch() 必须明确抛出 FileNotFoundError
def test_fetch_rejects_a_missing_local_pdf(tmp_path: Path) -> None:
    module = _local_source_module()
    missing_pdf = tmp_path / "missing.pdf"  # 只构造一个路径, 不创建文件, 例如/tmp/pytest-of-user/.../missing.pdf, 这个文件故意不存在

    # 下面这个代码表示: 当调用 LocalSource().fetch() 并传入一个不存在的 PDF 路径时, 必须抛出 FileNotFoundError 异常, 并且异常消息中必须包含 "PDF not found: <绝对路径>"
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"PDF not found: {missing_pdf.resolve()}"),
    ):
        module.LocalSource().fetch(str(missing_pdf))    # 创建本地采集器, 并传入不存在的 PDF 路径


# 这是 LocalSource 的核心成功路径测试
# 它验证一个本地 PDF 被采集后, 文件、元数据和来源记录是否都正确保存
# 临时输入 PDF
#    ↓ LocalSource.fetch()
# 规范论文目录
#    ├── raw.pdf
#    ├── meta.json
#    └── source.txt
def test_fetch_copies_pdf_and_persists_standard_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 准备隔离环境
    module = _local_source_module() # 加载 local_source 模块
    source_pdf = _write_source_pdf(tmp_path)    # 在 pytest 临时目录中创建测试 PDF
    storage_root = tmp_path / "papers"  # 指定临时的论文存储根目录
    _isolate_paper_storage(monkeypatch, module, storage_root)   # 把 local_source 模块使用的 paper_dir() 临时替换掉. 正常情况下论文会写入项目的 data/papers/, 测试中则写到: /tmp/pytest-.../papers/, 测试结束后自动清理, 不污染真实数据

    result = module.LocalSource().fetch(str(source_pdf))    # 执行采集

    expected_id = make_paper_id(pdf_path=source_pdf)    # 获取 PDF 的 SHA-1 哈希作为 paper_id
    target_dir = storage_root / to_safe_dirname(expected_id)    # 规范论文目录, 例如 /tmp/pytest-.../papers/<SHA-1 哈希>/
    copied_pdf = target_dir / "raw.pdf" # 最终这篇论文的 PDF 会被复制到这个位置, 例如 /tmp/pytest-.../papers/<SHA-1 哈希>/raw.pdf

    assert isinstance(result, FetchResult)  # 验证返回的对象类型是 FetchResult
    assert result.meta.paper_id == expected_id  # 确认采集器使用文件内容生成了正确 ID
    assert result.meta.title == "example-paper" # 没有显式传入标题时, 使用文件名去掉 .pdf 后的部分
    assert result.meta.source == "local"    # 标记论文来自本地文件。
    assert result.meta.urls == [source_pdf.resolve().as_uri()]
    assert result.pdf_path == str(copied_pdf)   # 返回的 PDF 路径必须指向规范目录中的副本
    assert copied_pdf.read_bytes() == source_pdf.read_bytes()   # 确认复制后的文件内容与原文件完全一致

    persisted_meta = json.loads((target_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted_meta == result.meta.model_dump(mode="json")
    assert (target_dir / "source.txt").read_text(encoding="utf-8") == (
        f"source=local\nquery={source_pdf}\n"
    )

    # 人工在 meta.json 顶层补写语言标注后, 再次采集同一 PDF 必须保留该标注,
    # 而不是用来源返回的空值覆盖它。
    meta_path = target_dir / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["language"] = "zh"
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    second = module.LocalSource().fetch(str(source_pdf))

    assert second.meta.language == "zh"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["language"] == "zh"

# 如果在 LocalSource 构造函数中显式传入了 title 参数, 那么 fetch() 返回的 PaperMeta.title 必须使用这个显式标题, 而不是默认的文件名
def test_explicit_title_overrides_the_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _local_source_module()
    source_pdf = _write_source_pdf(tmp_path)
    _isolate_paper_storage(monkeypatch, module, tmp_path / "papers")

    result = module.LocalSource(title="A Deliberate Title").fetch(str(source_pdf))

    assert result.meta.title == "A Deliberate Title"

# 如果连续两次采集同一个本地 PDF, LocalSource.fetch() 必须复用同一个规范论文目录, 而不是创建两个不同的目录
def test_fetching_the_same_pdf_twice_reuses_the_canonical_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _local_source_module()
    source_pdf = _write_source_pdf(tmp_path)
    storage_root = tmp_path / "papers"
    _isolate_paper_storage(monkeypatch, module, storage_root)
    source = module.LocalSource()

    first = source.fetch(str(source_pdf))
    second = source.fetch(str(source_pdf))

    assert second.meta.paper_id == first.meta.paper_id
    assert second.pdf_path == first.pdf_path
    assert [path.name for path in storage_root.iterdir()] == [
        to_safe_dirname(first.meta.paper_id)
    ]
    assert Path(second.pdf_path).read_bytes() == source_pdf.read_bytes()
