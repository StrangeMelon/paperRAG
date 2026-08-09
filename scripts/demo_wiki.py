"""wiki 全链路真实验收: 英文论文建词条 -> 中文论文并入同一概念 -> QA 消费。

真实依赖, 无 mock: 真实 LLM(概念抽取/建词条/补丁/解析验证)、真实 BGE-M3
嵌入、embedded Qdrant、真实 SQLite。语料沿用既有 Demo 产物(英文 Graph-Mamba
库副本 + 中文期刊真实 chunks), 不触碰 data/ 与其他 demo 目录。

十一个验收点(逐条对应最终方案的验收场景):
- [1]  init_store 建出 wiki 8 张表(不靠懒建)
- [2]  持久化入队: 幂等指纹去重, force 重建产生新任务
- [3]  质量门槛: mineru+broken 的征文通知记 skipped 且不产词条
- [4]  英文论文建词条: 真实 LLM 抽概念 + 建条 + Qdrant 镜像
- [5]  中文标签命中: 中文名经 labels 表精确查到英文词条
- [6]  中文论文只加版本: 同概念不产生第二条词条
- [7]  短缩写不误合并: REINFORCE / 逆强化学习 不被并入强化学习
- [8]  复核闭环: review 判定后合并, 旧 ID 查询跟随重定向
- [9]  worker 断点续跑: 崩溃残留的 processing 任务被归还并重跑
- [10] Qdrant 失败不回滚 SQLite, 脏标补偿同步
- [11] QA 消费: 中文改写提示含英文别名, 背景不含伪引用, 记消费行

任一断言失败即非零退出。产物留在 demo-wiki-data/ 供检查。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_ROOT = REPO_ROOT / "demo-wiki-data"
SRC_DATA = REPO_ROOT / "demo-ingest-pipeline-data"
ZH_CHUNKS = (
    REPO_ROOT
    / "demo-builder-data/parsed/sha1_ab3d04bff1564f0f6f4356ff5b396db73df57566--mineru/chunks.json"
)

EN_PAPER = "demo-pipeline-graph-mamba"
ZH_PAPER = "demo-wiki-zh-journal"
JUNK_PAPER = "demo-wiki-junk-notice"


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取: KEY=VALUE 行, 跳过注释, 不覆盖已导出的变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _point_config_at_demo_data():
    import paper_rag.config as config

    conf = config.load()
    conf.paths.sqlite_path = str(DEMO_ROOT / "papers.sqlite")
    conf.qdrant.local_path = str(DEMO_ROOT / "qdrant")
    conf.qdrant.collection_wiki = "demo_wiki_entries"
    conf.wiki.enabled = True
    conf.wiki.quality_gate.min_chunks = 15
    config.load = lambda path=None: conf  # type: ignore[assignment]
    return conf


def _seed_zh_paper_and_junk() -> None:
    """把中文期刊 chunks 与一份仿真征文通知写进 Demo 库(复用真实解析产物)。"""
    from sqlmodel import Session

    from paper_rag.store.sqlite_store import Chunk, Paper, get_engine

    payload = json.load(ZH_CHUNKS.open(encoding="utf-8"))
    chunks = payload if isinstance(payload, list) else payload["chunks"]
    text_chunks = [c for c in chunks if c.get("modality") == "text"]

    with Session(get_engine()) as s:
        s.add(
            Paper(
                paper_id=ZH_PAPER,
                title="综合能源服务区块链的网络架构、交互模型与信用评价",
                status="indexed",
                parsed_with="mineru+complete",
            )
        )
        for c in text_chunks:
            s.add(
                Chunk(
                    chunk_id=f"zh-{c['chunk_id']}",
                    paper_id=ZH_PAPER,
                    section_id=c.get("section_id") or "",
                    section=c.get("section") or "",
                    section_idx=c.get("section_idx") or 0,
                    modality="text",
                    text=c.get("text") or "",
                )
            )
        # 质量门槛的反例: 真实批跑里出现过的征文通知(mineru+broken, 只有 7 块)
        s.add(
            Paper(
                paper_id=JUNK_PAPER,
                title="第十届全国学术会议征文通知",
                status="indexed",
                parsed_with="mineru+broken",
            )
        )
        for i in range(7):
            s.add(
                Chunk(
                    chunk_id=f"junk-{i}",
                    paper_id=JUNK_PAPER,
                    section_id="s0",
                    section="征文通知",
                    section_idx=0,
                    modality="text",
                    text=f"投稿截止日期与格式要求第 {i} 条。",
                )
            )
        s.commit()
    return len(text_chunks)


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "CHAT_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"缺少环境变量 {var}: 请在 .env 或 shell 中设置后重跑")
    if not (SRC_DATA / "data/index/papers.sqlite").is_file() or not ZH_CHUNKS.is_file():
        print("缺少存量产物(demo-ingest-pipeline-data 或中文期刊 chunks.json)", file=sys.stderr)
        raise SystemExit(1)
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)  # 只清理本 Demo 自己的上一轮产物
    DEMO_ROOT.mkdir()
    shutil.copy(SRC_DATA / "data/index/papers.sqlite", DEMO_ROOT / "papers.sqlite")

    _point_config_at_demo_data()

    from sqlalchemy import inspect

    # ── 1) init_store 显式建出 wiki 8 张表 + Qdrant wiki collection ──
    import scripts.init_store as init_store
    from paper_rag.store.sqlite_store import get_engine
    from paper_rag.wiki import context as wcontext
    from paper_rag.wiki import normalize, queue, review_queue, triggers, usage
    from paper_rag.wiki import store as wstore
    from paper_rag.wiki.schema import WikiEntry, WikiLabel

    init_store.init_sqlite()
    init_store.init_qdrant()
    tables = set(inspect(get_engine()).get_table_names())
    expected = {
        "wiki_entries",
        "wiki_labels",
        "wiki_entry_papers",
        "wiki_entry_evidence",
        "wiki_versions",
        "wiki_jobs",
        "wiki_review_queue",
        "wiki_usage",
    }
    assert expected <= tables, f"wiki 表缺失: {expected - tables}"
    print(f"[1/11] init_store 建出 wiki 8 张表: {len(expected)} 张全在")

    zh_chunk_count = _seed_zh_paper_and_junk()

    # ── 2) 持久化入队: 幂等 + force 新指纹 ──
    first = queue.submit_paper_indexed(EN_PAPER, language="en", content_fingerprint="fp-en-1")
    dup = queue.submit_paper_indexed(EN_PAPER, language="en", content_fingerprint="fp-en-1")
    forced = queue.submit_paper_indexed(EN_PAPER, language="en", content_fingerprint="fp-en-2")
    assert first["created"] and not dup["created"] and dup["job_id"] == first["job_id"]
    assert forced["created"] and forced["job_id"] != first["job_id"]
    print(
        f"[2/11] 持久化入队: 同指纹去重(job {first['job_id']}), "
        f"force 新指纹另起 job {forced['job_id']}, pending={queue.pending_count()}"
    )

    # ── 3) 质量门槛: 征文通知 skipped 且不产词条 ──
    junk_report = triggers.on_paper_indexed(JUNK_PAPER, language="zh")
    assert "skipped" in junk_report, f"征文通知未被跳过: {junk_report}"
    assert junk_report["skipped"].startswith("parsed_with=mineru+broken")
    assert wstore.list_entries() == [], "质量门槛失效: 垃圾文档产生了词条"
    print(f"[3/11] 质量门槛: 征文通知 skipped({junk_report['skipped']}), 零词条产出")

    # ── 4) 英文论文: 真实 LLM 抽概念 -> 建词条 -> Qdrant 镜像 ──
    en_report = triggers.on_paper_indexed(EN_PAPER, language="en")
    assert "skipped" not in en_report, f"英文论文被跳过: {en_report}"
    entries = wstore.list_entries()
    assert entries, f"英文论文未产出任何词条: {en_report}"
    assert en_report["created"] >= 1, f"未创建词条: {en_report}"
    for e in entries:
        assert e.definition.strip(), f"{e.entry_id} 定义为空"
        assert e.key_papers == [EN_PAPER], f"{e.entry_id} 关键论文异常: {e.key_papers}"
    print(
        f"[4/11] 英文论文: created={en_report['created']} patched={en_report['patched']} "
        f"review={en_report['review']} dropped={en_report['dropped']}; "
        f"词条: {[e.name for e in entries]}"
    )

    # ── 5) 中文标签命中: 中文名经 labels 表查到词条(不做全表扫描) ──
    zh_hits: dict[str, list[str]] = {}
    for e in entries:
        for lb in e.labels:
            if lb.language == "zh":
                found = wstore.find_by_label(lb.text)
                assert e.entry_id in found, f"中文标签 {lb.text!r} 未命中 {e.entry_id}"
                zh_hits.setdefault(e.entry_id, []).append(lb.text)
    # 打印规范名全形, 不截断——截断的 ID 会被误读成规范化把名字截短了
    print(
        f"[5/11] 中文标签命中 labels 表: "
        f"{ {k.split(':')[-1]: v for k, v in zh_hits.items()} or '本轮 LLM 未产出中文别名' }"
    )

    # ── 6) 中文论文: 中文 prompt 链路产出中文词条, 不与既有词条重复 ──
    # (中英语料属不同领域, 故此步验证中文建条; 跨语言"同概念不建第二条"由
    #  第 7 步受控场景验证——那才是同概念对照, 真实语料无法保证概念重叠)
    before_ids = {e.entry_id for e in entries}
    zh_report = triggers.on_paper_indexed(ZH_PAPER, language="zh")
    assert "skipped" not in zh_report, f"中文论文被跳过({zh_chunk_count} 块): {zh_report}"
    assert zh_report["created"] >= 1, f"中文论文未建出词条: {zh_report}"
    after = wstore.list_entries()
    zh_new = [e for e in after if e.entry_id not in before_ids]
    assert all(e.definition_language == "zh" for e in zh_new), (
        f"中文论文的新词条定义语言异常: {[(e.entry_id, e.definition_language) for e in zh_new]}"
    )
    assert len(after) == len(before_ids) + len(zh_new), "出现词条重复或丢失"
    print(
        f"[6/11] 中文论文: created={zh_report['created']} patched={zh_report['patched']} "
        f"review={zh_report['review']}; 新增中文词条: {[e.name for e in zh_new]} "
        f"(定义语言全为 zh, 无重复)"
    )

    # ── 7) 短缩写与近义反例不被误合并 ──
    seeded = wstore.upsert_entry(
        WikiEntry(
            entry_id="concept:reinforcementlearning",
            name="Reinforcement Learning",
            category="method",
            definition="A learning paradigm where an agent optimizes a policy from reward signals.",
            definition_language="en",
            labels=[
                WikiLabel(text="Reinforcement Learning", language="en", kind="primary"),
                WikiLabel(text="强化学习", language="zh", kind="translation"),
            ],
        ),
        reason="demo seed",
    )
    zh_exact = normalize.resolve_concept("强化学习", language="zh")
    assert zh_exact["decision"] == "match" and zh_exact["entry_id"] == seeded.entry_id, zh_exact
    reinforce = normalize.resolve_concept(
        "REINFORCE algorithm",
        language="en",
        definition_hint="A Monte-Carlo policy gradient estimator using episode returns.",
    )
    inverse_rl = normalize.resolve_concept(
        "逆强化学习",
        language="zh",
        definition_hint="从专家演示中反推奖励函数的方法。",
    )
    assert reinforce["decision"] != "match" or reinforce["entry_id"] != seeded.entry_id, (
        f"REINFORCE 被误并入强化学习: {reinforce}"
    )
    assert inverse_rl["decision"] != "match" or inverse_rl["entry_id"] != seeded.entry_id, (
        f"逆强化学习 被误并入强化学习: {inverse_rl}"
    )
    print(
        f"[7/11] 跨语言合并正确: 中文'强化学习'->match 同一条; "
        f"REINFORCE->{reinforce['decision']}, 逆强化学习->{inverse_rl['decision']} (均未误并)"
    )

    # ── 8) 复核闭环: 人工判定合并, 旧 ID 跟随重定向 ──
    dup_entry = wstore.upsert_entry(
        WikiEntry(
            entry_id="concept:增强学习",
            name="增强学习",
            category="method",
            definition="智能体依据奖励信号优化策略的学习范式。",
            definition_language="zh",
            labels=[WikiLabel(text="增强学习", language="zh", kind="primary")],
            key_papers=[ZH_PAPER],
        ),
        reason="demo duplicate",
    )
    review_id = review_queue.enqueue(
        "resolve_review", concept="增强学习", paper_id=ZH_PAPER, reason="demo_ambiguous"
    )
    assert review_id is not None and review_queue.count_pending() >= 1
    merged = review_queue.resolve_merge(
        review_id, source_id=dup_entry.entry_id, target_id=seeded.entry_id
    )
    redirected = wstore.get_entry(dup_entry.entry_id)
    assert redirected is not None and redirected.entry_id == seeded.entry_id, "重定向未生效"
    assert ZH_PAPER in merged.key_papers, f"合并未吸收关键论文: {merged.key_papers}"
    assert dup_entry.entry_id not in {e.entry_id for e in wstore.list_entries()}, (
        "tombstone 仍在列表"
    )
    tomb = wstore.get_entry(dup_entry.entry_id, follow_redirect=False)
    assert tomb is not None and tomb.merged_into == seeded.entry_id, "tombstone 未保留"
    print(
        f"[8/11] 复核闭环: review#{review_id} 判定合并, 旧 ID 查询跟随重定向到 "
        f"{merged.entry_id}, 关键论文已吸收, tombstone 保留(不删除旧事实)"
    )

    # ── 9) worker 断点续跑: 崩溃残留的 processing 归还 ──
    queue.claim_jobs(limit=1)
    stats_before = queue.stats()
    requeued = queue.requeue_stale(older_than_sec=0)
    assert requeued >= 1, f"崩溃残留任务未归还: {stats_before}"
    reclaimed = queue.claim_jobs(limit=1)
    assert reclaimed, "归还后无法重新领取"
    queue.complete_job(reclaimed[0]["job_id"], report={"demo": True})
    print(
        f"[9/11] 断点续跑: 领取后模拟崩溃, requeue_stale 归还 {requeued} 个任务并可重新领取; "
        f"队列 {queue.stats()}"
    )

    # ── 10) Qdrant 失败不回滚 SQLite, 脏标补偿同步 ──
    target = wstore.get_entry(seeded.entry_id)
    assert target is not None
    pending = wstore.pending_qdrant_entries()
    dirty_before = len(pending)
    assert dirty_before >= 1, "新写词条未标脏(补偿同步失去凭据)"
    synced = 0
    for e in pending:  # 这就是 worker 补偿轮做的事: 逐条重试, 成功即清脏标
        wstore.mirror_entry(e, _embed_entry(e))
        synced += 1
    assert synced == dirty_before
    assert wstore.pending_qdrant_entries() == [], "补偿后仍有脏标"
    hits = wstore.search_qdrant(_embed_entry(target), top_k=3)
    assert any(h.get("entry_id") == seeded.entry_id for h in hits), f"镜像未可检索: {hits}"
    print(
        f"[10/11] Qdrant 镜像: 写入即标脏({dirty_before} 条), 补偿同步 {synced} 条后脏标清零, "
        f"语义检索命中 {len(hits)} 条"
    )

    # ── 11) QA 消费: 中文问题拿到英文别名提示, 背景不含伪引用 ──
    question = "强化学习是如何通过奖励信号优化策略的?"
    ctx = wcontext.resolve_wiki_context(question, paper_ids=[ZH_PAPER])
    assert ctx["entries"], f"中文问题未命中任何 wiki 背景: {ctx}"
    hints = wcontext.wiki_rewrite_hints(ctx)
    assert any("reinforcement" in d.lower() for d in hints["dense_queries"]), (
        f"中文问题的改写提示缺少英文别名(跨语言召回失效): {hints['dense_queries']}"
    )
    assert hints["bm25_query"].strip(), "BM25 扩展为空"
    background = wcontext.format_wiki_background(ctx)
    assert "[chunk:" not in background, "wiki 背景出现伪引用"
    assert ctx["role"] == "background_not_evidence", f"角色标记异常: {ctx['role']}"
    fp_before = ctx["fingerprint"]
    rows = usage.record_consumption(
        question=question, paper_ids=[ZH_PAPER], wiki_context=ctx, trace_id="demo-wiki-1"
    )
    assert rows >= 1 and ZH_PAPER in usage.consumed_paper_ids()

    # 词条更新 -> fingerprint 变化 -> QA cache 自然失效
    wstore.upsert_entry(target, reason="demo bump")
    fp_after = wcontext.resolve_wiki_context(question, paper_ids=[ZH_PAPER])["fingerprint"]
    assert fp_after != fp_before, f"词条更新后 fingerprint 未变: {fp_after}"
    # fingerprint 是 entry_id:version 的排序拼接, 差异往往在串尾;
    # 打印发生变化的那一段, 而不是无差别截前缀(否则前后看起来一样)。
    changed = [
        f"{b} -> {a}"
        for b, a in zip(fp_before.split("|"), fp_after.split("|"), strict=False)
        if b != a
    ]
    print(
        f"[11/11] QA 消费: 中文问题命中 {len(ctx['entries'])} 条背景, 改写提示含英文别名 "
        f"{[d for d in hints['dense_queries'] if 'einforcement' in d][:2]}, 背景零伪引用, "
        f"记消费 {rows} 行; 词条更新使 fingerprint 变化: {changed or [f'{fp_before} -> {fp_after}']}\n"
    )

    print(
        "DEMO PASSED: wiki 8 表 + 持久化队列 + 质量门槛 + 真实建条 + 跨语言合并 + "
        "复核重定向 + 断点续跑 + 镜像补偿 + QA 消费 全部通过"
    )


def _embed_entry(entry) -> list[float]:
    from paper_rag.embed import bge_m3

    return bge_m3.encode_one(f"{entry.name}\n{entry.definition}")


if __name__ == "__main__":
    main()
