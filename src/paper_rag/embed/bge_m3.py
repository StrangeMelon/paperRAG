"""BGE-M3 嵌入器: 全系统唯一的稠密向量出口, 入库与查询共用。

封装 FlagEmbedding.BGEM3FlagModel 为惰性单例(import 时不加载, 首次 encode 才
初始化), 只暴露 dense 向量——稀疏检索走 BM25/FTS5, 不用 M3 的 sparse 头。
BGE-M3 原生多语种(中英同空间), 语言链路在此由模型天然承接, 无需代码级
中文扩展; 中文约束落在真实验收断言(demo_bge_m3 / test_bge_m3_real)。

与基准的差异: 无(行为 1:1 保真); 仅 Iterable 改从 collections.abc 导入。
设备策略沿用基准: auto 时 macOS 强制 CPU(MPS 对 bge-m3 有 23GB 级内存分配
问题), 有 CUDA 用 CUDA; fp16 只在非 CPU 开(CPU 上有数值问题)。
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from .. import config as cfg
from ..mcp.resource_guards import hold_resource
from ..utils.hf_cache import resolve_cached_snapshot
from ..utils.logger import get_logger

log = get_logger("embed.bge_m3")
_MODEL = None
_MODEL_LOCK = Lock()
_ENCODE_LOCK = Lock()


def _model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from FlagEmbedding import BGEM3FlagModel

                c = cfg.load()
                device = c.embedding.device
                if device == "auto":
                    import platform

                    import torch

                    if platform.system() == "Darwin":
                        device = "cpu"
                    elif torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                use_fp16 = device != "cpu"
                model_name = resolve_cached_snapshot(c.embedding.model_name, c.paths.models_dir)
                log.info(
                    f"loading {model_name} on {device} "
                    f"(fp16={use_fp16}, cache={c.paths.models_dir})"
                )
                _MODEL = BGEM3FlagModel(
                    model_name,
                    use_fp16=use_fp16,
                    cache_dir=c.paths.models_dir,
                    devices=[device] if device != "auto" else None,
                )
    return _MODEL


def encode(texts: Iterable[str]) -> list[list[float]]:
    c = cfg.load().embedding
    texts = list(texts)
    if not texts:
        return []
    # Wiki worker 可并发多篇论文; 单个 GPU 模型的推理与首次加载保持串行。
    with hold_resource("embedding"), _ENCODE_LOCK:
        out = _model().encode(
            texts,
            batch_size=c.batch_size,
            max_length=c.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    dense = out["dense_vecs"]
    return [vec.tolist() for vec in dense]


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
