"""vision/cache.py 边界契约: 键的敏感面与读写往返。

用真实临时目录与真实图片字节, 不 mock 文件系统。
"""

from __future__ import annotations

from pathlib import Path

from paper_rag.vision.cache import VisionSummaryCache
from paper_rag.vision.schema import STATUS_OK, VisualSummaryRequest, VisualSummaryResult


def _png(tmp_path: Path, name: str = "fig.png", payload: bytes = b"\x89PNG-A") -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def _req(asset: Path, **over) -> VisualSummaryRequest:
    base = {
        "paper_id": "p1",
        "chunk_id": "c1",
        "modality": "figure",
        "asset_path": asset,
        "caption": "Figure 1 accuracy",
        "surrounding_context": "ctx",
        "model": "glm-4.6v",
        "language": "en",
    }
    base.update(over)
    return VisualSummaryRequest(**base)  # type: ignore[arg-type]


def test_key_is_sha256_prefixed_and_stable(tmp_path):
    cache = VisionSummaryCache(tmp_path / "cache")
    req = _req(_png(tmp_path))
    key = cache.key_for(req)
    assert key.startswith("sha256:")
    assert key == cache.key_for(req)  # 同输入必须同键


def test_key_changes_with_language():
    # 核心中文扩展: 语言改变提示词与输出语种, 不入键会发生跨语言脏命中。
    def key(tmp: Path, lang: str | None) -> str:
        return VisionSummaryCache(tmp / "c").key_for(_req(_png(tmp), language=lang))

    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        zh = key(Path(d1), "zh")
        en = key(Path(d2), "en")
    assert zh != en


def test_key_changes_with_prompt_version_and_model(tmp_path):
    cache = VisionSummaryCache(tmp_path / "cache")
    asset = _png(tmp_path)
    base = cache.key_for(_req(asset))
    assert base != cache.key_for(_req(asset, prompt_version="v1"))
    assert base != cache.key_for(_req(asset, model="other-vl"))


def test_key_changes_with_image_bytes_and_caption(tmp_path):
    cache = VisionSummaryCache(tmp_path / "cache")
    a = _png(tmp_path, "a.png", b"\x89PNG-A")
    b = _png(tmp_path, "b.png", b"\x89PNG-B")
    assert cache.key_for(_req(a)) != cache.key_for(_req(b))
    assert cache.key_for(_req(a)) != cache.key_for(_req(a, caption="other"))


def test_write_then_read_roundtrip(tmp_path):
    cache = VisionSummaryCache(tmp_path / "cache")
    key = cache.key_for(_req(_png(tmp_path)))
    cache.write(
        key,
        VisualSummaryResult(
            status=STATUS_OK,
            summary="视觉类型: 折线图",
            provider="api",
            model="glm-4.6v",
            raw={"visual_type": "折线图"},
        ),
    )
    got = cache.read(key)
    assert got is not None
    assert got.summary == "视觉类型: 折线图"  # 中文不被转义成 \uXXXX
    assert got.cache_key == key
    assert got.raw == {"visual_type": "折线图"}


def test_read_missing_returns_none(tmp_path):
    assert VisionSummaryCache(tmp_path / "cache").read("sha256:deadbeef") is None


def test_read_corrupt_json_returns_none_instead_of_raising(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "sha256_bad.json").write_text("{not json", encoding="utf-8")
    assert VisionSummaryCache(cache_dir).read("sha256:bad") is None


def test_read_unknown_keys_returns_none_not_typeerror(tmp_path):
    # 缓存文件是历史产物, schema 演进后不得炸穿 ingest。
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "sha256_old.json").write_text('{"status": "ok", "gone": 1}', encoding="utf-8")
    assert VisionSummaryCache(cache_dir).read("sha256:old") is None


def test_write_creates_dir_and_file_name_is_colon_free(tmp_path):
    cache = VisionSummaryCache(tmp_path / "nested" / "cache")
    cache.write("sha256:abc", VisualSummaryResult(status=STATUS_OK, summary="s"))
    files = list((tmp_path / "nested" / "cache").glob("*.json"))
    assert len(files) == 1
    assert ":" not in files[0].name
