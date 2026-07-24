"""
@Project : CVKit
@File : rtsp_video.py
@Author : Haruka
@Date : 2026/1/28 09:06
"""

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

RTSP_STREAM_TEMPLATES = {
    "Dahua": "cam/realmonitor?channel={}&subtype={}",
    "HIKVISION": "Streaming/Channels/{}0{}",
}


def _resolve_stream_path(manufacturer: str, channel: int, subtype: int) -> str:
    """
    根据厂商模板解析 RTSP 路径后缀。

    Args:
        manufacturer (str): 摄像机厂商名称（如 "Dahua", "HIKVISION"）。
        channel (int): 通道号。
        subtype (int): 码流类型（通常 0 为主码流，1 为子码流）。

    Returns:
        str: 解析后的 RTSP 路径后缀字符串。

    Raises:
        ValueError: 如果厂商不支持，或者未提供必要的 `channel` 和 `subtype` 参数。
    """
    if manufacturer not in RTSP_STREAM_TEMPLATES:
        raise ValueError(
            f"不支持的厂商: {manufacturer}. 支持列表: {list(RTSP_STREAM_TEMPLATES.keys())}"
        )

    if channel is None or subtype is None:
        raise ValueError("初始化失败: 必须指定 `channel` 和 `subtype` ")

    return RTSP_STREAM_TEMPLATES[manufacturer].format(channel, subtype)


class Video:
    """
    使用 OpenCV 拉流的摄像头封装。
    缓冲最新的 N 张图片（默认5），可用 [] 操作符获取。
    """

    def __init__(
        self,
        ip: str,
        port: int = 554,
        frame_extraction: int = 3,
        cache_size: int = 5,
        username: str | None = None,
        password: str | None = None,
        manufacturer: str | None = None,
        channel: int | None = None,
        subtype: int | None = None,
        stream_suffix: str | None = None,
        buffer: deque[np.ndarray] | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        初始化 Video 对象，配置 RTSP 连接参数。

        Args:
            ip (str): 摄像机的 IP 地址。
            port (int, optional): RTSP 端口号。默认为 554。
            frame_extraction (int, optional): 抽帧间隔，即每隔多少帧缓存一帧。默认为 3。
            cache_size (int, optional): 帧缓冲区大小。默认为 5。
            username (str | None, optional): RTSP 登录用户名。默认为 None。
            password (str | None, optional): RTSP 登录密码。默认为 None。
            manufacturer (str | None, optional): 摄像机厂商（用于自动生成路径）。默认为 None。
            channel (int | None, optional): 通道号（用于自动生成路径）。默认为 None。
            subtype (int | None, optional): 码流类型（用于自动生成路径）。默认为 None。
            stream_suffix (str | None, optional): 自定义 RTSP 路径后缀。如果提供，将忽略厂商模板。默认为 None。
            buffer (deque[np.ndarray] | None, optional): 外部传入的帧缓冲区。如果为 None，则内部创建。默认为 None。
            logger (logging.Logger | None, optional): 日志记录器。如果为 None，则创建名为 "Video" 的记录器。默认为 None。
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.frame_extraction = frame_extraction
        self.logger = logger or logging.getLogger("Video")

        if stream_suffix is not None:
            stream_path = stream_suffix
        elif manufacturer is not None and channel is not None and subtype is not None:
            stream_path = _resolve_stream_path(manufacturer, channel, subtype)
        else:
            raise ValueError(
                "参数不足：需提供 stream_suffix 或 (manufacturer, channel, subtype)"
            )

        # 组合RTSP地址
        if username and password:
            self.rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/{stream_path}"
        else:
            self.rtsp_url = f"rtsp://{self.ip}:{self.port}/{stream_path}"

        self.logger.debug(f"rtsp_url={self.rtsp_url}")

        # 初始化缓存和消息系统
        self.buffer = buffer if buffer is not None else deque(maxlen=cache_size)
        self.cache_size = self.buffer.maxlen

        # 控制变量
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._frame_count = -1  # 缓存首帧
        self.info = None  # (height, width)

        self.start()

    def _capture_loop(self):
        """后台线程循环读取视频帧"""
        while self._running:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.logger.error(f"无法连接 {self.rtsp_url}，5秒后重试")
                cap.release()
                time.sleep(5)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 获取分辨率
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            with self._lock:
                self.info = (h, w)

            while self._running:
                if not cap.grab():
                    self.logger.warning("流中断 (grab failed)，尝试重连")
                    break

                # 每隔 frame_extraction 帧缓存一次
                self._frame_count += 1
                if self._frame_count % self.frame_extraction != 0:
                    continue

                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    self.logger.warning("解码失败")
                    continue
                with self._lock:
                    self.buffer.append(frame)

            cap.release()
            time.sleep(1)

    def start(self):
        """启动拉流线程"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, name="rtsp_capture_loop", daemon=True
            )
            self._thread.start()

    def stop(self):
        """停止拉流"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def __getitem__(self, item):
        with self._lock:
            if isinstance(item, int):
                return self.buffer[item]
            return list(self.buffer)[item]

    def __len__(self):
        with self._lock:
            return len(self.buffer)

    def __del__(self):
        self.stop()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
