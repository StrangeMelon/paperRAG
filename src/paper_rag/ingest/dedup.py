"""去重辅助程序。

目前: 按 paper_id 查询 SQLite; 如果存在且状态为 'done', 则返回 True
标题规范化的回退方案暂留为待办事项(需要为 papers.title_norm 列建立索引)
"""

from __future__ import annotations

import re

from ..utils.logger import get_logger

log = get_logger(__name__)

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)

# 全部转为小写, 删除空格、标点、下划线, 只保留字母和数字, 以便进行标题比较
def normalize_title(title: str) -> str:
    """将标题转换为适合比较的规范形式。"""
    return _PUNCT_RE.sub("", title.lower())

# 只有当paper_id在数据库中存在并且状态为 'done' 时, 才返回 True
# 表示不重复下载
def is_done(paper_id: str) -> bool:
    """判断论文是否已经完成入库。"""
    from ..store.sqlite_store import get_paper

    row = get_paper(paper_id)
    return bool(row and row.status == "done")
