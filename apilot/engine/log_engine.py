"""
日志引擎模块

管理系统日志事件处理和文件记录功能
"""

import logging
from datetime import datetime
from pathlib import Path

from apilot.core.utility import get_folder_path
from apilot.core import (
    BaseEngine,
    Event,
    EventEngine,
    MainEngine,
    SETTINGS,
    EVENT_LOG,
)


class LogEngine(BaseEngine):
    """日志引擎 - 处理系统日志事件"""
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super(LogEngine, self).__init__(main_engine, event_engine, "log")

        if not SETTINGS["log.active"]:
            return

        # 创建和配置主logger
        self.logger = self._setup_logger()

        # 注册事件处理
        self.event_engine.register(EVENT_LOG, self.process_log_event)

    def _setup_logger(self) -> logging.Logger:
        """设置logger，整合处理器创建逻辑"""
        logger = logging.getLogger("apilot")
        logger.setLevel(SETTINGS["log.level"])

        # 清除现有处理器避免重复
        logger.handlers.clear()

        # 添加NullHandler防止警告
        logger.addHandler(logging.NullHandler())

        # 创建格式化器
        formatter = logging.Formatter("%(asctime)s  %(levelname)s: %(message)s")

        # 添加控制台处理器
        if SETTINGS["log.console"]:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 添加文件处理器
        if SETTINGS["log.file"]:
            today_date = datetime.now().strftime("%Y%m%d")
            filename = f"apilot_{today_date}.log"
            log_path = get_folder_path("log")
            file_path = log_path.joinpath(filename)

            file_handler = logging.FileHandler(
                file_path, mode="a", encoding="utf8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def process_log_event(self, event: Event) -> None:
        """处理日志事件"""
        if not hasattr(self, "logger"):
            return

        log = event.data

        # 构建格式化消息，包含来源
        if hasattr(log, "source") and log.source:
            formatted_msg = f"[{log.source}] {log.msg}"
        else:
            formatted_msg = log.msg

        # 记录日志
        self.logger.log(log.level, formatted_msg)
