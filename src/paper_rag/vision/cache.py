"""视觉摘要的文件级缓存。

视觉调用是本项目最贵的单步(整图 base64 上传 + 长输出), 20000 篇规模下重复
ingest / force 重建必须命中缓存而不是重新计费。键取图片字节与全部提示词输入
的 sha256, 因此内容一变即天然失效。

相对基准两处加固:
- ``language`` 进键: 语言决定提示词与输出语种, 不入键会让中文论文命中此前的
  英文摘要(跨语言脏命中)。
- ``read`` 对未知键的历史缓存文件返回 None 而非抛 TypeError: 缓存是长寿产物,
  schema 演进不得炸穿 ingest。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from .schema import VisualSummaryRequest, VisualSummaryResult


class VisionSummaryCache:
    """以图片字节、上下文、模型、语言与提示词版本为键的 JSON 缓存。"""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def key_for(self, request: VisualSummaryRequest) -> str:
        h = hashlib.sha256()
        h.update(request.asset_path.read_bytes())
        for value in (
            request.paper_id,
            request.chunk_id,
            request.modality,
            request.caption,
            request.surrounding_context,
            request.model or "",
            request.language or "",
            request.prompt_version,
        ):
            h.update(b"\0")  # 分隔符防止字段拼接歧义
            h.update(value.encode("utf-8", errors="replace"))
        return f"sha256:{h.hexdigest()}"

    def read(self, key: str) -> VisualSummaryResult | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        allowed = {f.name for f in fields(VisualSummaryResult)}
        if not payload.keys() <= allowed or "status" not in payload:
            return None
        try:
            return VisualSummaryResult(**payload)
        except TypeError:
            return None

    def write(self, key: str, result: VisualSummaryResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": result.status,
            "summary": result.summary,
            "provider": result.provider,
            "model": result.model,
            "raw": result.raw,
            "error": result.error,
            "cache_key": key,
            "warnings": result.warnings,
        }
        self._path_for(key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _path_for(self, key: str) -> Path:
        safe = key.replace(":", "_")
        return self.cache_dir / f"{safe}.json"
