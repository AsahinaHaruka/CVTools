"""
@Project : CVTools
@File : perspective_transformation.py
@Author : Haruka
@Date : 2025/10/27 15:59
"""
from typing import Iterable, List, Sequence, Tuple, Union, Optional
import numpy as np
import cv2

PointLike = Union[Sequence[float], np.ndarray]
Points4 = Union[Sequence[PointLike], np.ndarray]
Size2i = Tuple[int, int]


def _order_points(pts):
    # 计算四边形质心
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    ordered = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    # 按逆时针排序后确保第一点是左上
    ordered = np.array(ordered, dtype=np.float32)

    # 找到 y 最小的两点，再从中选 x 最小的作为左上
    idx_tl = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -idx_tl, axis=0)
    return ordered


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class PerspectiveTransformer:
    """
    4点透视变换器。
    - 可显式传入输出尺寸(`dst_size=(w, h)`)，否则从4点自动估计。
    - 调用实例时对传入的`list[np.ndarray]`进行透视变换并返回新`list`。
    """

    def __init__(
        self,
        points: Points4,
        dst_size: Optional[Size2i] = None,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: Union[int, float, Tuple[int, int, int]] = 0,
    ) -> None:
        """
        :param points: 4个点，可为`[(x, y), ...]`, `np.ndarray((4,2))`或`(4,1,2)`等形态。
        :param dst_size: 目标输出尺寸 `(width, height)`。不传则自动计算。
        :param interpolation: `cv2.INTER_`插值方式。
        :param border_mode: `cv2.BORDER_`边界模式。
        :param border_value: 边界填充值。
        """
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value

        src = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if src.shape[0] != 4 or src.shape[1] != 2:
            raise ValueError("points必须包含4个二维点。")
        self.src_rect = _order_points(src)

        self._update_transform(dst_size)

    def _update_transform(self, dst_size: Optional[Size2i]) -> None:
        tl, tr, br, bl = self.src_rect
        widthA = _euclidean(br, bl)
        widthB = _euclidean(tr, tl)
        heightA = _euclidean(tr, br)
        heightB = _euclidean(tl, bl)

        if dst_size is None:
            dst_w = int(round(max(widthA, widthB)))
            dst_h = int(round(max(heightA, heightB)))
        else:
            dst_w = int(dst_size[0])
            dst_h = int(dst_size[1])

        if dst_w <= 0 or dst_h <= 0:
            raise ValueError("计算得到的目标尺寸非法。请检查输入点与`dst_size`。")

        self.dst_w, self.dst_h = dst_w, dst_h
        self.dst_rect = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )
        self.M = cv2.getPerspectiveTransform(self.src_rect, self.dst_rect)

    def __call__(self, images: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        对images中每一张cv2图片进行透视变换。
        :return: 变换后的图片list。
        """
        if images is None:
            return []
        imgs = list(images)
        return [
            cv2.warpPerspective(
                img,
                self.M,
                (self.dst_w, self.dst_h),
                flags=self.interpolation,
                borderMode=self.border_mode,
                borderValue=self.border_value,
            )
            for img in imgs
        ]

    def update_points(self, points: Points4, dst_size: Optional[Size2i] = None) -> None:
        """
        更新4点并重算透视矩阵与输出尺寸。
        可同时传入新的`dst_size`，不传则沿用自动计算。
        """
        src = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if src.shape[0] != 4 or src.shape[1] != 2:
            raise ValueError("points必须包含4个二维点。")
        self.src_rect = _order_points(src)
        self._update_transform(dst_size)

    def set_output_size(self, dst_size: Size2i) -> None:
        """
        仅更新输出尺寸并重算透视矩阵（不改4点）。
        """
        self._update_transform(dst_size)