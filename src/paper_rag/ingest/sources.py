"""论文采集源的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import FetchResult


class PaperSource(ABC):
    """所有论文采集器的公共基类。"""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, identifier: str) -> FetchResult:
        """根据来源标识符采集论文元数据和 PDF。

        identifier 的含义由具体采集器决定, 例如本地路径、PDF URL、
        arXiv ID 或 DOI。
        """
        raise NotImplementedError
