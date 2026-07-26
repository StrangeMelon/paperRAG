"""
用于在本地 Hugging Face 缓存中寻找已完整下载的模型快照。
它不会下载模型、不会设置 HF_HOME 等环境变量；
它的作用是把模型名，例如 BAAI/bge-m3，转换为本地快照目录

这样 src/paper_rag/embed/bge_m3.py:32 和 src/paper_rag/retrieve/rerank.py:33 加载模型时可以直接传本地路径，
避免在离线演示、测试或网络不稳定时，transformers / huggingface_hub 试图联网检查或下载模型。
"""



from __future__ import annotations  # 必须只位于模块文档字符串之后、任何普通语句之前。也就是前面最多只能有一个"""""""包裹的文档字符串

import json
from pathlib import Path

# 输入是一个候选快照目录，输出是布尔值，表示其中是否有完整模型权重
def _has_model_weights(snapshot: Path) -> bool:
    direct_weights = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    )
    if any((snapshot / name).is_file() for name in direct_weights):
        return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        except (KeyError, TypeError, ValueError):
            continue
        shards = set(weight_map.values())
        if shards and all((snapshot / shard).is_file() for shard in shards):
            return True
    return False

# 找到完整缓存时，返回本地快照目录字符串；
# 找不到时，返回原始模型名字符串
def resolve_cached_snapshot(model_name: str, cache_dir: str | Path) -> str:
    """Return a local snapshot path when available, otherwise ``model_name``.

    Passing the snapshot path avoids opportunistic network calls from
    transformers/huggingface_hub in offline demo and test environments.
    """
    p = Path(model_name)
    if p.exists():
        return str(p)
    if "/" not in model_name:
        return model_name

    repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        snapshot = repo_dir / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if snapshot.exists() and _has_model_weights(snapshot):
            return str(snapshot)

    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir()):
            if snapshot.is_dir() and _has_model_weights(snapshot):
                return str(snapshot)
    return model_name
