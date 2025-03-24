"""
日志引擎模块

负责事件驱动日志系统的处理和输出
"""

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from colorama import Fore, Style, init

from apilot.core.engine import BaseEngine
from apilot.core.event import Event, EventEngine
from apilot.core.object import LogData


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
        
        # 默认配置
        self.log_level = logging.INFO
        self.console_enabled = True
        self.file_enabled = True
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_count = 7
        
        # 设置日志处理器
        self.handlers = []
        self.setup_handlers()
        
        # 注册事件处理
        self.event_engine.register(EVENT_LOG, self.process_log_event)
        
        # 记录引擎启动日志
        self.emit_log(logging.INFO, "日志引擎初始化完成")
    
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
        
        # 检查日志级别
        if log.level < self.log_level:
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
        
        return f"{time_str} [{level_name}] - {source_prefix}{log.msg}"
    
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
    
    def emit_log(self, level: int, msg: str) -> None:
        """发送日志到事件系统"""
        log = LogData(level=level, msg=msg, source=self.engine_name)
        event = Event(EVENT_LOG, log)
        self.event_engine.put(event)
    
    def log_debug(self, msg: str) -> None:
        """调试级别日志"""
        self.emit_log(logging.DEBUG, msg)
    
    def log_info(self, msg: str) -> None:
        """信息级别日志"""
        self.emit_log(logging.INFO, msg)
    
    def log_warning(self, msg: str) -> None:
        """警告级别日志"""
        self.emit_log(logging.WARNING, msg)
    
    def log_error(self, msg: str) -> None:
        """错误级别日志"""
        self.emit_log(logging.ERROR, msg)
    
    def log_critical(self, msg: str) -> None:
        """严重错误级别日志"""
        self.emit_log(logging.CRITICAL, msg)
        
    def close(self) -> None:
        """关闭引擎时清理资源"""
        # 关闭所有日志处理器
        for handler in self.handlers:
            handler.close()
