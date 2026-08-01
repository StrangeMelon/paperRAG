"""BGE-M3 真实验收: 真加载模型, 用真实中英 chunk 验证向量质量与中英同空间。

输入: demo-builder-data 的真实 chunks.json(需先跑 scripts/demo_builder.py);
模型: BAAI/bge-m3 本地缓存(data/index/models, 首跑自动下载约 2.3GB)。

验收点:
- 形状与数值: 真实中英 chunk 批量编码全部 1024 维、有限值、范数非零;
- 批次一致性: encode_one 与并批编码同文本的余弦相似度 > 0.995;
- 语义区分(中文): 真实查询在中文期刊 62 个文本块上排序, 最相关节的块
  显著高于全体均值, 且 top-3 命中"信用评价"相关章节;
- 中英同空间: 同义中英句对的相似度显著高于不相关中英句对——这是
  语言链路(zh/en 同一索引)成立的模型侧前提。
产出 demo-bge-m3-data/embeddings.json 落盘(查询与 top-3 chunk 的完整
1024 维向量、相似度、文本预览)供人工查看向量长什么样。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from paper_rag.embed.bge_m3 import encode, encode_one

REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)
EN_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_28acb520c921be7a1968207519dfa95d6af88800--mineru/chunks.json"
)
OUT_FILE = REPO_ROOT / "demo-bge-m3-data/embeddings.json"


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def main() -> None:
    for p in (ZH_CHUNKS, EN_CHUNKS):
        if not p.is_file():
            print(
                f"缺少输入: {p.relative_to(REPO_ROOT)}(需先跑 scripts/demo_builder.py)",
                file=sys.stderr,
            )
            raise SystemExit(1)

    zh = json.loads(ZH_CHUNKS.read_text(encoding="utf-8"))["chunks"]
    en = json.loads(EN_CHUNKS.read_text(encoding="utf-8"))["chunks"]

    # --- 1. 形状与数值: 真实中英混批 ---
    sample = [c["context_text"] for c in zh[:8]] + [c["context_text"] for c in en[:8]]
    t0 = time.time()
    vecs = encode(sample)
    dt = time.time() - t0
    assert len(vecs) == 16, f"批量条数 {len(vecs)} != 16"
    for v in vecs:
        assert len(v) == 1024, f"维度 {len(v)} != 1024"
        assert all(math.isfinite(x) for x in v), "向量含非有限值"
        assert math.sqrt(sum(x * x for x in v)) > 0.1, "向量范数异常"
    print(f"[1/4] 真实中英混批 16 条: 全部 1024 维、数值健康({dt:.1f}s 含首次模型加载)")

    # --- 2. 批次一致性 ---
    probe = zh[0]["context_text"]
    solo = encode_one(probe)
    batched = encode([probe, en[0]["context_text"]])[0]
    consistency = _cos(solo, batched)
    assert consistency > 0.995, f"批次一致性 {consistency:.4f} <= 0.995"
    print(f"[2/4] 批次一致性: encode_one vs 并批 余弦 {consistency:.4f} > 0.995")

    # --- 3. 中文语义区分: 真实查询在 62 个真实块上排序 ---
    query = "区块链节点的信用值是如何评价和更新的"
    q_vec = encode_one(query)
    chunk_vecs = encode([c["context_text"] for c in zh])
    sims = [_cos(q_vec, v) for v in chunk_vecs]
    ranked_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    top3 = [(sims[i], zh[i]["section"]) for i in ranked_idx[:3]]
    mean_sim = sum(sims) / len(sims)
    print(f"[3/4] 中文检索排序: top-3 = {[(f'{s:.3f}', sec) for s, sec in top3]}")
    print(f"      全体均值 {mean_sim:.3f}")
    assert top3[0][0] > mean_sim + 0.05, "top-1 相对均值无区分度"
    assert any("信用" in sec for _, sec in top3), f"top-3 未命中信用评价相关章节: {top3}"

    # --- 4. 中英同空间 ---
    pairs = encode(
        [
            "检索增强生成通过引入外部知识提升回答质量",
            "Retrieval-augmented generation improves answers with external knowledge",
            "今天食堂的午饭是番茄炒蛋",
        ]
    )
    related = _cos(pairs[0], pairs[1])
    unrelated = _cos(pairs[0], pairs[2])
    print(f"[4/4] 中英同空间: 同义句对 {related:.3f} vs 不相关句对 {unrelated:.3f}")
    assert related > unrelated + 0.15, "中英同义句未显著高于不相关句"
    assert related > 0.6, f"中英同义句相似度过低: {related:.3f}"

    # --- 5. 向量落盘供人工查看 ---
    payload = {
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "note": "dense 向量已 L2 归一(范数≈1), 余弦相似度=点积",
        "query": {"text": query, "vector": [round(x, 6) for x in q_vec]},
        "top3_chunks": [
            {
                "rank": r + 1,
                "similarity": round(sims[i], 6),
                "chunk_id": zh[i]["chunk_id"],
                "section": zh[i]["section"],
                "page": zh[i]["page"],
                "text_preview": zh[i]["context_text"][:80],
                "vector": [round(x, 6) for x in chunk_vecs[i]],
            }
            for r, i in enumerate(ranked_idx[:3])
        ],
    }
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"      向量已保存: {OUT_FILE.relative_to(REPO_ROOT)}(含完整 1024 维)")

    print()
    print("BGE-M3 真实验收通过: 维度/数值/批次一致性/中文排序/中英同空间全部成立。")


if __name__ == "__main__":
    main()
