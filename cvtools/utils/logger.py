"""
Author: Haruka
Date: 2026-01-15 15:19:47
LastEditors: Haruka
LastEditTime: 2026-01-16 10:19:40
FilePath: /road/uilt/logger.py
"""

import logging
import os
import sys
import multiprocessing
import multiprocessing.queues
from logging.handlers import (
    RotatingFileHandler,
    TimedRotatingFileHandler,
    QueueHandler,
    QueueListener,
)


class ColoredFormatter(logging.Formatter):
    """
    用于控制台输出的带颜色日志格式化器
    """

    # ANSI 颜色代码
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # 针对不同日志级别的颜色映射
    FORMATS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        """
        初始化 ColoredFormatter。

        Args:
            fmt (str, optional): 日志格式字符串。默认为 None。
            datefmt (str, optional): 日期时间格式字符串。默认为 None。
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录，并根据日志级别添加颜色。

        Args:
            record (logging.LogRecord): 需要格式化的日志记录对象。

        Returns:
            str: 带有颜色编码的格式化日志字符串。
        """
        # 1. 获取当前级别的颜色
        log_fmt = self.FORMATS.get(record.levelno)

        original_fmt = self._fmt

        formatter = logging.Formatter(log_fmt + original_fmt + self.RESET)

        # 3. 使用修改后的格式化器
        return formatter.format(record)


class LoggerBuilder:
    """
    基础日志构建器
    负责具体的 Handler 创建和 Formatter 配置
    """

    DEBUG_FORMAT = "%(asctime)s | %(levelname)-8s | %(processName)s:%(process)d | %(filename)s:%(lineno)d | %(message)s"
    DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(processName)s:%(process)d | %(message)s"

    @staticmethod
    def _create_handlers(
            log_dir: str,
            log_filename: str | None,
            console_output: bool,
            file_rotation: str,
            max_bytes: int,
            backup_count: int,
            when: str,
            interval: int,
            delay: bool,
            format: str = DEFAULT_FORMAT,
    ) -> list[logging.Handler]:
        """内部方法：创建 Handler 列表"""
        handlers = []
        formatter = logging.Formatter(format)

        # 1. 控制台 Handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                ColoredFormatter(fmt=format)
            )
            handlers.append(console_handler)

        # 2. 文件 Handler
        if log_filename:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_path = os.path.join(log_dir, log_filename)

            file_handler = None
            if file_rotation == "size":
                file_handler = RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                    delay=delay,
                )
            elif file_rotation == "time":
                file_handler = TimedRotatingFileHandler(
                    file_path,
                    when=when,
                    interval=interval,
                    backupCount=backup_count,
                    encoding="utf-8",
                    delay=delay,
                )
            elif file_rotation == "none":
                file_handler = logging.FileHandler(
                    file_path, delay=delay, encoding="utf-8"
                )

            if file_handler:
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)

        return handlers

    @staticmethod
    def get_logger(
            name: str = "root",
            level: int = logging.INFO,
            log_dir: str = "logs",
            log_filename: str | None = None,
            console_output: bool = True,
            file_rotation: str = "size",  # 选项: 'size', 'time', 'none'
            max_bytes: int = 10 * 1024 * 1024,  # 10MB
            backup_count: int = 5,
            when: str = "midnight",
            interval: int = 1,
            delay: bool = False,
    ) -> logging.Logger:
        """
        获取一个配置好的 `logging.Logger` 实例，用于单进程日志记录。

        此方法根据提供的参数配置日志的输出目标（控制台、文件）和行为（文件轮转策略）。
        它会检查是否已存在同名 Logger 且已配置 Handler，以避免重复添加。

        Args:
            name (str): Logger 的名称。默认为 "root"。
            level (int): 日志记录的最低级别。例如 `logging.INFO`。默认为 `logging.INFO`。
            log_dir (str): 日志文件存储的目录。默认为 "logs"。
            log_filename (str | None): 日志文件的名称。如果为 `None`，则不输出到文件。默认为 `None`。
            console_output (bool): 是否将日志输出到控制台。默认为 `True`。
            file_rotation (str): 文件轮转策略。可选值包括 "size" (按大小轮转), "time" (按时间轮转), "none" (不轮转)。默认为 "size"。
            max_bytes (int): 当 `file_rotation` 为 "size" 时，单个日志文件的最大字节数。默认为 10MB。
            backup_count (int): 当文件轮转时，保留的旧日志文件数量。默认为 5。
            when (str): 当 `file_rotation` 为 "time" 时，时间轮转的间隔单位。例如 "midnight", "h", "d"。默认为 "midnight"。
            interval (int): 当 `file_rotation` 为 "time" 时，时间轮转的间隔数量。默认为 1。
            delay(bool):是否延迟打开日志文件，直到第一次写入时才创建。

        Returns:
            logging.Logger: 配置好的 Logger 实例。

        Raises:
            OSError: 如果无法创建日志目录或写入日志文件。
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if level == logging.DEBUG:
            format = LoggerBuilder.DEBUG_FORMAT
        else:
            format = LoggerBuilder.DEFAULT_FORMAT

        # 防止重复添加 Handler
        if not logger.handlers:
            handlers = LoggerBuilder._create_handlers(
                log_dir=log_dir,
                log_filename=log_filename,
                console_output=console_output,
                file_rotation=file_rotation,
                max_bytes=max_bytes,
                backup_count=backup_count,
                when=when,
                interval=interval,
                delay=delay,
                format=format
            )
            for h in handlers:
                logger.addHandler(h)
        return logger


_GLOBAL_LOG_QUEUE = None
_WORKER_LOG_QUEUE = None


class MultiProcessLogManager:
    """
    [多进程模式] 日志管理器
    """

    @staticmethod
    def init_main_listener(
            level: int = logging.INFO,
            log_dir: str = "logs",
            log_filename: str = None,
            console_output: bool = True,
            file_rotation: str = "size",  # 选项: 'size', 'time', 'none'
            max_bytes: int = 10 * 1024 * 1024,
            backup_count: int = 5,
            when: str = "midnight",
            interval: int = 1,
            delay: bool = False,
    ) -> tuple[multiprocessing.Queue, QueueListener]:
        """
        【主进程调用】初始化队列监听器。

        此方法在主进程中被调用，用于设置一个多进程安全的日志系统。
        它创建一个 `multiprocessing.Queue` 来收集来自子进程的日志消息，
        并启动一个 `QueueListener` 来将这些消息写入到配置好的日志目标（控制台、文件）。
        如果主进程有Handler, 会自动将主进程已有的 Handler 清除，避免文件冲突。

        Args:
            level (int): 日志记录的最低级别。例如 `logging.INFO`。默认为 `logging.INFO`。
            log_dir (str): 日志文件存储的目录。默认为 "logs"。
            log_filename (str | None): 日志文件的名称。如果为 `None`，则不输出到文件。默认为 "app.log"。
            console_output (bool): 是否将日志输出到控制台。默认为 `True`。
            file_rotation (str): 文件轮转策略。可选值包括 "size" (按大小轮转), "time" (按时间轮转), "none" (不轮转)。默认为 "size"。
            max_bytes (int): 当 `file_rotation` 为 "size" 时，单个日志文件的最大字节数。默认为 10MB。
            backup_count (int): 当文件轮转时，保留的旧日志文件数量。默认为 5。
            when (str): 当 `file_rotation` 为 "time" 时，时间轮转的间隔单位。例如 "midnight", "h", "d"。默认为 "midnight"。
            interval (int): 当 `file_rotation` 为 "time" 时，时间轮转的间隔数量。默认为 1。
            delay(bool):是否延迟打开日志文件，直到第一次写入时才创建。

        Returns:
            tuple[multiprocessing.Queue, QueueListener]: 包含日志队列和队列监听器的元组。
                - `multiprocessing.Queue`: 子进程用于发送日志消息的队列。
                - `QueueListener`: 负责从队列中取出日志并写入到实际处理器的监听器。

        Raises:
            OSError: 如果无法创建日志目录或写入日志文件。

        Examples:
            >>> log_queue, listener = MultiProcessLogManager.init_main_listener(
            ...     log_filename="multiprocess_app.log", file_rotation="size", max_bytes=5*1024*1024, console_output=True
            ... )
            >>> multiprocessing.Process(target=run_task, args=(log_queue,))
            >>> listener.stop()
            # 子进程：
            >>> MultiProcessLogManager.configure_worker(queue)
            >>> logger = logging.getLogger(__name__)
            >>> logger.info(f"Task 开始运行...")

        """
        global _GLOBAL_LOG_QUEUE
        queue = multiprocessing.Queue(-1)
        _GLOBAL_LOG_QUEUE = queue

        root = logging.getLogger()
        root.setLevel(level)
        if level == logging.DEBUG:
            _format = LoggerBuilder.DEBUG_FORMAT
        else:
            _format = LoggerBuilder.DEFAULT_FORMAT
        handlers = []

        # 检查并清理 Root Logger 的现有 Handler
        if root.handlers:
            print(f"[LogManager] 警告: 检测到主进程已有 {len(root.handlers)} 个 Handler。")
            print(f"[LogManager] 现有 Handler 为: {root.handlers}")
            print("[LogManager] 为防止日志重复或格式冲突，正在清理旧 Handler，完全使用 LogManager 配置...")

            # 直接清空，而不是复制
            root.handlers = []

        handlers = LoggerBuilder._create_handlers(
            log_dir=log_dir,
            log_filename=log_filename,
            console_output=console_output,
            file_rotation=file_rotation,
            max_bytes=max_bytes,
            backup_count=backup_count,
            when=when,
            interval=interval,
            delay=delay,
            format=_format
        )

        # 启动监听器
        listener = QueueListener(queue, *handlers)

        # 配置主进程
        q_handler = QueueHandler(queue)
        root.addHandler(q_handler)

        return queue, listener

    @staticmethod
    def configure_worker(
            queue: multiprocessing.queues.Queue | None = None, level: int = logging.INFO
    ) -> None:
        """
        【子进程调用】配置当前进程的日志发送端。

        此方法应在每个子进程启动后立即调用，用于将子进程的日志消息重定向到
        主进程的日志队列。它会清空当前进程的根日志器的所有处理器，
        并添加一个 `QueueHandler`，使其将所有日志记录发送到指定的队列。

        Args:
            queue (multiprocessing.Queue): 由主进程创建并传递给子进程的
                `multiprocessing.Queue` 实例。子进程将通过此队列发送日志消息。
            level (int): 子进程中日志记录的最低级别。例如 `logging.INFO`。
                默认为 `logging.INFO`。

        Returns:
            None: 此方法不返回任何值。

        Note:
            调用此方法后，子进程中所有通过 `logging` 模块发出的日志都将
            通过队列发送到主进程进行处理，而不是直接输出到控制台或文件。
        """
        global _GLOBAL_LOG_QUEUE
        target_queue = queue or _GLOBAL_LOG_QUEUE
        if target_queue is None:
            raise ValueError(
                "未找到日志队列！\n"
                "你没有传入 queue 参数，且全局变量为空。\n"
                "1. (Linux) 确保在 fork 子进程前已调用 init_main_listener。\n"
                "2. (Windows/macOS) 由于使用 spawn 模式，无法共享全局变量，你必须显式传入 queue 参数。"
            )
        global _WORKER_LOG_QUEUE
        _WORKER_LOG_QUEUE = target_queue

        h = QueueHandler(target_queue)
        root = logging.getLogger()
        root.handlers = []
        root.addHandler(h)
        root.setLevel(level)

    @staticmethod
    def get_log_queue() -> multiprocessing.Queue:
        """
        【子进程调用】获取当前进程持有的日志队列。

        此方法允许子进程获取其配置的 `multiprocessing.Queue` 实例，
        以便直接与主进程的日志监听器进行通信（尽管通常通过 `logging` 模块的 API 间接完成）。

        Returns:
            multiprocessing.Queue: 当前子进程用于发送日志消息的队列。

        Raises:
            RuntimeError: 如果在调用 `configure_worker` 之前尝试获取日志队列。
        """
        global _WORKER_LOG_QUEUE
        if _WORKER_LOG_QUEUE is None:
            raise RuntimeError("日志队列未初始化！请先调用 configure_worker。")
        return _WORKER_LOG_QUEUE
