"""
@Project ：CVTools
@File ：data_define.py
@Author ：Haruka
@Date ：2025/8/22 16:01 
"""

from dataclasses import dataclass, field
from typing import Tuple


# ----------------- 类型定义 -----------------
@dataclass
class Area:
    """
    区域的划分,该类用于表示一个矩形区域，并提供判断点是否在该区域内的方法，
    该构建方法可以确保坐标为左上角与右下角坐标的形式。

    Attributes
    ----------
    area : (start_x, start_y, end_x, end_y)
        区域划分,形如 `(start_x, start_y, end_x, end_y)
        表示一个矩形区域的左上角与右下角坐标。
    """
    area:Tuple = field(repr=False)  # 隐藏原始入参

    # 标准化后的坐标
    start_x: int = field(init=False)
    start_y: int = field(init=False)
    end_x: int = field(init=False)
    end_y: int = field(init=False)

    def __post_init__(self):
        sx, sy, ex, ey= self.area
        # 保证 start 始终 ≤ end
        self.start_x = min(sx, ex)
        self.start_y = min(sy, ey)
        self.end_x = max(sx, ex)
        self.end_y = max(sy, ey)

    def is_in_area(self, x: int, y: int) -> bool:
        """判断点是否位于区域内（含边界）。"""
        return self.start_x <= x <= self.end_x and self.start_y <= y <= self.end_y
