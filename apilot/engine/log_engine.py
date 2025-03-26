"""
日志引擎模块

负责事件驱动日志系统的处理和输出
"""

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from colorama import Fore, Style, init

from apilot.core.engine import BaseEngine
from apilot.core.event import Event, EventEngine
from apilot.core.object import LogData
from apilot.core.constant import EngineType


# 初始化colorama，支持跨平台颜色
init(autoreset=True)


class LogEngine(BaseEngine):
    """日志引擎 - 基于事件驱动处理系统日志"""

    # 日志级别颜色映射
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def __init__(self, main_engine, event_engine: EventEngine) -> None:
        """初始化日志引擎"""
        super().__init__(main_engine, event_engine, "log")

        # 自动检测当前环境
        self.engine_type = self.detect_engine_type()
        self.mode = "backtest" if self.engine_type == EngineType.BACKTESTING else "live"

        # 基础配置
        self.log_level = logging.INFO
        self.console_enabled = True
        self.file_enabled = True
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_count = 7

        # 回测模式特有配置
        self.buffer_logs = False
        self.log_buffer: List[LogData] = []

        # 根据模式配置日志行为
        self.configure_by_mode(self.mode)

        # 设置日志处理器
        self.handlers = []
        self.setup_handlers()

        # 日志过滤器
        self.filters: Dict[str, Callable[[LogData], bool]] = {
            "backtest": lambda log: log.level >= logging.WARNING or (hasattr(log, 'extra') and log.extra and log.extra.get('important', False)),
            "live": lambda log: True  # 实盘记录所有日志
        }

        # 注册事件处理
        self.event_engine.register(EVENT_LOG, self.process_log_event)

        # 记录引擎启动日志
        self.emit_log(logging.INFO, f"日志引擎初始化完成 (模式: {self.mode})")

    def detect_engine_type(self) -> EngineType:
        """检测当前引擎类型"""
        if hasattr(self.main_engine, "engine_type"):
            return self.main_engine.engine_type
        return EngineType.LIVE  # 默认为实盘环境

    def configure_by_mode(self, mode: str) -> None:
        """根据模式配置日志行为"""
        if mode == "backtest":
            # 回测模式配置
            self.log_level = logging.WARNING  # 回测只记录警告和错误
            self.file_enabled = False  # 回测默认不写入文件
            self.buffer_logs = True  # 使用内存缓冲
        else:  # live模式
            # 实盘模式配置
            self.log_level = logging.INFO  # 实盘记录更多信息
            self.file_enabled = True  # 实盘写入文件
            self.buffer_logs = False  # 不使用内存缓冲

    def setup_handlers(self) -> None:
        """设置日志处理器"""
        # 控制台处理器
        if self.console_enabled:
            console_handler = self.create_console_handler()
            self.handlers.append(console_handler)

        # 文件处理器
        if self.file_enabled:
            file_handler = self.create_file_handler()
            self.handlers.append(file_handler)

    def create_console_handler(self) -> logging.Handler:
        """创建控制台日志处理器"""
        return logging.StreamHandler(sys.stdout)

    def create_file_handler(self) -> logging.Handler:
        """创建文件日志处理器"""
        today = datetime.now().strftime("%Y%m%d")
        file_name = f"apilot_{today}.log"
        log_file = self.log_dir / file_name

        handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            backupCount=self.backup_count
        )

        # 设置格式化器
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        handler.setFormatter(formatter)

        return handler

    def process_log_event(self, event: Event) -> None:
        """处理日志事件"""
        log: LogData = event.data

        # 应用过滤器
        filter_func = self.filters.get(self.mode, lambda _: True)
        if not filter_func(log):
            return

        # 检查日志级别
        if log.level < self.log_level:
            return

        # 如果是回测模式且启用了缓冲，添加到缓冲区
        if self.buffer_logs:
            self.log_buffer.append(log)

            # 在控制台显示重要的日志
            if log.level >= logging.WARNING:
                formatted_msg = self.format_log_message(log)
                color = self.LEVEL_COLORS.get(log.level, "")
                print(f"{color}{formatted_msg}{Style.RESET_ALL}")

            return

        # 格式化消息
        formatted_msg = self.format_log_message(log)

        # 控制台输出（带颜色）
        if self.console_enabled:
            color = self.LEVEL_COLORS.get(log.level, "")
            print(f"{color}{formatted_msg}{Style.RESET_ALL}")

        # 文件输出
        if self.file_enabled:
            for handler in self.handlers:
                if isinstance(handler, TimedRotatingFileHandler):
                    record = self.create_log_record(log, formatted_msg)
                    handler.emit(record)

    def format_log_message(self, log: LogData) -> str:
        """格式化日志消息"""
        time_str = log.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_name = log.level_name
        source_prefix = f"[{log.source}] " if log.source else ""

        # 添加回测时间戳（如果存在）
        backtesting_time = ""
        if hasattr(log, 'extra') and log.extra and 'backtesting_time' in log.extra:
            backtesting_time = f"[BT: {log.extra['backtesting_time']}] "

        return f"{time_str} [{level_name}] - {backtesting_time}{source_prefix}{log.msg}"

    def create_log_record(self, log: LogData, formatted_msg: str) -> logging.LogRecord:
        """创建日志记录对象"""
        return logging.LogRecord(
            name="apilot",
            level=log.level,
            pathname="",
            lineno=0,
            msg=formatted_msg,
            args=(),
            exc_info=None
        )

    def emit_log(self, level: int, msg: str, source: str = "", **kwargs) -> None:
        """发送日志到事件系统"""
        extra = kwargs.pop("extra", {})
        extra.update(kwargs)  # 将所有关键字参数添加到extra

        log = LogData(
            level=level,
            msg=msg,
            source=source or self.engine_name,
            extra=extra
        )
        event = Event(EVENT_LOG, log)
        self.event_engine.put(event)

    def flush_buffer(self) -> List[str]:
        """刷新日志缓冲区并返回所有日志消息"""
        messages = []

        for log in self.log_buffer:
            formatted_msg = self.format_log_message(log)
            messages.append(formatted_msg)

            # 写入文件（如果启用）
            if self.file_enabled:
                for handler in self.handlers:
                    if isinstance(handler, TimedRotatingFileHandler):
                        record = self.create_log_record(log, formatted_msg)
                        handler.emit(record)

        # 清空缓冲区
        buffer_copy = self.log_buffer.copy()
        self.log_buffer = []

        return messages

    def log_debug(self, msg: str, source: str = "", **kwargs) -> None:
        """调试级别日志"""
        self.emit_log(logging.DEBUG, msg, source, **kwargs)

    def log_info(self, msg: str, source: str = "", **kwargs) -> None:
        """信息级别日志"""
        self.emit_log(logging.INFO, msg, source, **kwargs)

    def log_warning(self, msg: str, source: str = "", **kwargs) -> None:
        """警告级别日志"""
        self.emit_log(logging.WARNING, msg, source, **kwargs)

    def log_error(self, msg: str, source: str = "", **kwargs) -> None:
        """错误级别日志"""
        self.emit_log(logging.ERROR, msg, source, **kwargs)

    def log_critical(self, msg: str, source: str = "", **kwargs) -> None:
        """严重错误级别日志"""
        self.emit_log(logging.CRITICAL, msg, source, **kwargs)

    # def write_log(self, msg: str, source: str = "", level: int = logging.INFO, **kwargs) -> None:
    #     """
    #     写入日志的通用方法，与BacktestingEngine兼容
    #     """
    #     self.emit_log(level, msg, source, **kwargs)

    def close(self) -> None:
        """关闭引擎时清理资源"""
        # 刷新缓冲区
        if self.buffer_logs and self.log_buffer:
            self.flush_buffer()

        # 关闭所有日志处理器
        for handler in self.handlers:
            handler.close()
