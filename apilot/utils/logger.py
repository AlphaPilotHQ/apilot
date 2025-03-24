"""
日志工具模块

提供简单易用的日志功能，可以独立使用或与apilot事件系统集成。
支持控制台彩色输出和文件日志。
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union

from colorama import Fore, Style, init

# 初始化colorama，确保终端颜色能正确显示
init(autoreset=True)


def get_file_path(folder: str, filename: str) -> str:
    """
    获取文件路径，自动创建目录
    """
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent.parent
    target_dir = root_dir / folder
    
    # 确保目录存在
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        
    return str(target_dir / filename)


class CustomFormatter(logging.Formatter):
    """
    自定义日志格式，支持不同级别的日志颜色
    """
    FORMATS = {
        logging.DEBUG: ('', " "),
        logging.INFO: (Fore.GREEN, " "),
        logging.WARNING: (Fore.YELLOW, " "),
        logging.ERROR: (Fore.RED, " "),
        logging.CRITICAL: (Fore.RED + Style.BRIGHT, " "),
    }

    def format(self, record):
        """
        格式化日志记录，添加颜色和前缀
        """
        color, prefix = self.FORMATS.get(record.levelno, (Fore.WHITE, ""))
        
        # 添加源标记，与LogEngine保持一致
        if hasattr(record, 'source') and record.source:
            record.msg = f"[{record.source}] {record.msg}"
            
        original_message = super().format(record)
        return f"{color}{prefix}{original_message}{Style.RESET_ALL}"


class CustomConsoleHandler(logging.StreamHandler):
    """
    自定义控制台日志处理器，支持DEBUG级别直接输出
    """
    def emit(self, record):
        formatted_msg = self.format(record)
        print(formatted_msg, flush=True) if record.levelno == logging.DEBUG else super().emit(record)


class StrategyLogger:
    """
    策略日志类，提供单例模式下的日志功能
    """
    _instances = {}  # 存储已创建的logger实例（单例模式）

    def __new__(cls, name='alphapilot', strategy_name=None):
        key = f"{name}_{strategy_name}" if strategy_name else name
        
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialize_logger(name, strategy_name)
            cls._instances[key] = instance
        return cls._instances[key]

    def _initialize_logger(self, name, strategy_name):
        """
        初始化logger，添加文件和控制台处理器
        """
        self.name = name
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 文件日志处理器，按天滚动存储
        file_handler = TimedRotatingFileHandler(
            get_file_path('logs', f"{name.lower()}.log"), 
            when='midnight',
            interval=1, 
            backupCount=7
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))
        self.logger.addHandler(file_handler)

        # 控制台日志处理器，带颜色格式化
        console_handler = CustomConsoleHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter("%(message)s"))
        self.logger.addHandler(console_handler)
        
    def debug(self, msg, *args, **kwargs):
        """输出调试日志"""
        self._log(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        """输出信息日志"""
        self._log(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        """输出警告日志"""
        self._log(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """输出错误日志"""
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        """输出严重错误日志"""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
        
    def _log(self, level, msg, *args, **kwargs):
        """内部日志方法，支持事件系统和源标记"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
            
        # 如果有策略名称，添加为源
        if self.strategy_name and 'source' not in kwargs['extra']:
            kwargs['extra']['source'] = self.strategy_name
            
        # 尝试发送事件，如果可能
        try:
            from apilot.core import MainEngine, EVENT_LOG, LogData
            main_engine = kwargs.pop('main_engine', None)
            if main_engine and isinstance(main_engine, MainEngine):
                if level == logging.DEBUG:
                    main_engine.log_debug(msg, source=kwargs['extra'].get('source', ''))
                elif level == logging.INFO:
                    main_engine.log_info(msg, source=kwargs['extra'].get('source', ''))
                elif level == logging.WARNING:
                    main_engine.log_warning(msg, source=kwargs['extra'].get('source', ''))
                elif level == logging.ERROR:
                    main_engine.log_error(msg, source=kwargs['extra'].get('source', ''))
                elif level == logging.CRITICAL:
                    main_engine.log_critical(msg, source=kwargs['extra'].get('source', ''))
                return
        except ImportError:
            pass  # 如果没有事件系统，直接使用logger
            
        self.logger.log(level, msg, *args, **kwargs)


def get_logger(name=None, strategy_name=None) -> StrategyLogger:
    """
    获取日志对象
    
    参数:
        name: 日志名称，默认为'alphapilot'
        strategy_name: 策略名称，会作为日志源标记
        
    返回:
        配置好的策略日志器
    """
    return StrategyLogger(name or 'alphapilot', strategy_name)


def set_log_level(level: Union[int, str]) -> None:
    """
    设置全局日志级别
    
    参数:
        level: 日志级别，可以是logging模块的常量或字符串('DEBUG','INFO'等)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)


def divider(name='', sep='=', logger=None) -> None:
    """
    画一个带时间的横线
    
    参数:
        name: 分割线中的文本
        sep: 分隔符
        logger: 指定的logger（可选）
    """
    seperator_len = 72
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    middle = f' {name} {now} '

    # 计算分隔符数量
    decoration_count = max(4, (seperator_len - len(middle)) // 2)
    line = f"{sep * decoration_count}{middle}{sep * decoration_count}"

    # 确保总长度合适
    if len(line) < seperator_len:
        line += sep

    log_target = logger if logger else default_logger
    log_target.debug(line)


# 默认全局日志器
default_logger = get_logger()


if __name__ == '__main__':
    # 基本用法演示
    default_logger.debug("调试信息")
    default_logger.info("提示信息，绿色")
    default_logger.warning("警告信息，黄色")
    default_logger.error("错误信息，红色")
    default_logger.critical("重要提示，深红色")
    
    # 带策略名称的日志
    strategy_logger = get_logger(strategy_name="MomentumStrategy")
    strategy_logger.info("这是策略日志，前面会显示策略名称")
    
    divider('日志测试完成')
    print('日志文件保存在项目根目录下的logs文件夹中')
