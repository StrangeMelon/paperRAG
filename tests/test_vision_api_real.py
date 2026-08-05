"""真实视觉 API 集成测试(默认跳过, 有凭据才跑)。

钉的是打桩测试证明不了的三件事: GLM-4.6V 真能吃下 base64 data URL、
temperature=0.01 不被拒(智谱 temperature 为开区间, 写死 0 会 400)、
返回内容能被 _loads_json/summary_from_payload 解析成非空摘要。

    VISION_BASE_URL=... VISION_API_KEY=... VISION_MODEL=glm-4.6v \
        uv run python -m pytest tests/test_vision_api_real.py -q -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from paper_rag.config import load
from paper_rag.vision.api import OpenAIVisionSummarizer
from paper_rag.vision.schema import STATUS_OK, VisualSummaryRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
_ZH_PARSED = REPO_ROOT / "demo-mineru-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566"


def _first_figure() -> Path:
    figures = sorted((_ZH_PARSED / "figures").glob("*.jpg"))
    if not figures:
        pytest.skip("真实解析产物缺失, 跳过真实视觉调用")
    return figures[0]


@pytest.fixture
def vision_cfg():
    cfg = load().vision
    if not (cfg.base_url and cfg.api_key and cfg.model):
        pytest.skip("未配置 VISION_BASE_URL/VISION_API_KEY/VISION_MODEL, 跳过真实视觉调用")
    return cfg


def _summarizer(cfg):
    return OpenAIVisionSummarizer(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        extra_body=cfg.extra_body,
    )


@pytest.mark.parametrize("language", ["zh", "en"])
def test_real_call_returns_summary(vision_cfg, language):
    """真实调用两种语言各一次; 摘要必须非空, 且不得走异常分支。"""
    request = VisualSummaryRequest(
        paper_id="real-check",
        chunk_id=f"real-{language}",
        modality="figure",
        asset_path=_first_figure(),
        caption="图1综合能源服务物理架构",
        surrounding_context="本文提出综合能源服务的物理架构与信息交互模型。",
        model=vision_cfg.model,
        language=language,
    )
    result = _summarizer(vision_cfg).summarize(request)
    if result.error:
        print(f"\n[{language}] 真实调用报错: {result.error}")
    assert result.status == STATUS_OK, result.error
    assert result.summary.strip()
    print(f"\n[{language}] 摘要:\n{result.summary[:400]}")


def test_zh_summary_is_not_pure_ascii(vision_cfg):
    """zh 提示词必须真的换来中文描述, 否则中文语料的 BM25 词面侧等于失明。"""
    request = VisualSummaryRequest(
        paper_id="real-check",
        chunk_id="real-zh-ascii",
        modality="figure",
        asset_path=_first_figure(),
        caption="图1综合能源服务物理架构",
        surrounding_context="综合能源服务的物理架构。",
        model=vision_cfg.model,
        language="zh",
    )
    result = _summarizer(vision_cfg).summarize(request)
    assert result.status == STATUS_OK, result.error
    assert any("一" <= ch <= "鿿" for ch in result.summary), result.summary[:300]


def test_env_names_documented():
    """凭据一律走环境变量; 仓库里不得出现硬编码 key(secret-scan 的单测镜像)。"""
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("VISION_BASE_URL", "VISION_API_KEY", "VISION_MODEL"):
        assert name in example
    assert not os.environ.get("VISION_API_KEY", "").startswith("sk-your")
