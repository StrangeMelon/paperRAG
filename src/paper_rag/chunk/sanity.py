"""章节完整性打分器: 按节名判断一次解析是否覆盖论文四大区域。

启发式与基准一致: 只看节名(小写后子串匹配), 不看 chunk 数量(噪声太大)。
输出标签供 `Paper.parsed_with` 增补(`{parser}+{quality}`), 用于日后过滤
坏解析而无需全量重跑:
  - "complete": 四区(intro/method/experiment/conclusion)全部命中
  - "partial": intro+method 命中且 experiment/conclusion 至少一个
  - "minimal": 仅 intro 或 method 命中
  - "broken": 四区全空

与基准的差异(2026-08-01 已确认): 基准关键词表全英文, 完美解析的中文论文
会被判 broken 而遭误过滤。重建版为四区各配中文关键词表, 并按既有全链约定
路由: `language="zh"` 查中文表、`"en"` 查英文表(基准同款行为)、`None`
查双表并集(未知语言不猜)。输出四值标签不变(`parsed_with` 契约不能动)。
"""

from __future__ import annotations

_AREAS_EN = {
    "intro": ["abstract", "introduction", "intro"],
    "method": [
        "method",
        "approach",
        "methodology",
        "model",
        "framework",
        "architecture",
        "retrieval",
        "generation",
        "augmentation",
        "training",
        "implementation",
        "problem formulation",
    ],
    "experiment": [
        "experiment",
        "experimental",
        "evaluation",
        "metric",
        "result",
        "ablation",
        "analysis",
        "dataset",
        "task",
    ],
    "conclusion": ["conclusion", "discussion", "summary", "limitation", "future work"],
}

_AREAS_ZH = {
    "intro": ["摘要", "引言", "绪论", "概述", "前言", "背景"],
    "method": ["方法", "模型", "架构", "框架", "算法", "设计", "机制", "实现", "构建", "策略"],
    "experiment": [
        "实验",
        "评估",
        "评测",
        "测试",
        "结果",
        "分析",
        "数据集",
        "仿真",
        "算例",
        "案例",
    ],
    "conclusion": ["结论", "总结", "结语", "展望", "讨论", "局限"],
}


def grade_sections(section_names: list[str], *, language: str | None = None) -> str:
    """在 ingest 流程中的作用是给论文增加解析质量标签, 供 `Paper.parsed_with` 增补, 用于日后过滤坏解析而无需全量重跑"""

    lows = [n.lower() for n in section_names]
    if language == "zh":
        tables = (_AREAS_ZH,)
    elif language == "en":
        tables = (_AREAS_EN,)
    else:
        tables = (_AREAS_EN, _AREAS_ZH)

    def _has(area: str) -> bool:
        return any(k in name for table in tables for k in table[area] for name in lows)

    intro = _has("intro")
    method = _has("method")
    exp = _has("experiment")
    concl = _has("conclusion")

    if intro and method and exp and concl:
        return "complete"
    if intro and method and (exp or concl):
        return "partial"
    if intro or method:
        return "minimal"
    return "broken"
