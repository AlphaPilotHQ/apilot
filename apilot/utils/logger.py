import logging
import os
import sys
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from typing import ClassVar

from colorama import Fore, Style, init

# 初始化 colorama
init(autoreset=True)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class CustomFormatter(logging.Formatter):
    FORMATS: ClassVar[dict[int, tuple[str, str]]] = {
        logging.DEBUG: (Fore.CYAN, " "),
        logging.INFO: (Fore.GREEN, " "),
        logging.WARNING: (Fore.YELLOW, " "),
        logging.ERROR: (Fore.RED, " "),
        logging.CRITICAL: (Fore.RED + Style.BRIGHT, " "),
    }

    def format(self, record):
        color, prefix = self.FORMATS.get(record.levelno, (Fore.WHITE, ""))
        original_message = super().format(record)
        return f"{color}{prefix}{original_message}{Style.RESET_ALL}"


class CustomLogger:
    _instances: ClassVar[dict[str, "CustomLogger"]] = {}

    def __new__(cls, name="apilot"):
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialize_logger(name)
            cls._instances[name] = instance
        return cls._instances[name]

    def _initialize_logger(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 防止重复handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 日志输出到文件(每天自动创建新文件)
        file_handler = TimedRotatingFileHandler(
            os.path.join(LOG_DIR, f"{name}.log"), when="midnight", backupCount=7
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        )

        # 控制台日志处理器,带颜色格式化
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            CustomFormatter("%(asctime)s [%(levelname)s] - %(message)s")
        )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


def get_logger(name=None) -> logging.Logger:
    return CustomLogger(name or "apilot").logger


logger = get_logger()


def set_level(level_name, name=None):
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = level_map.get(level_name.lower(), logging.INFO)
    get_logger(name).setLevel(level)


def log_exceptions(logger_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(logger_name)
                logger.exception(f"函数 {func.__name__} 出错: {e}")
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    logger.debug("提示信息,青色")
    logger.info("提示信息,绿色")
    logger.warning("警告信息,黄色")
    logger.error("错误信息,红色")
    logger.critical("重要提示,深红色")
