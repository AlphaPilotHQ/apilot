import logging
import sys
import time
import unicodedata
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from colorama import Fore, Style, init

from engine.utils.path_kit import get_file_path

# 初始化 colorama，确保终端颜色能正确显示
init(autoreset=True)

def get_display_width(text: str) -> int:
    """
    获取文本的显示宽度，中文字符算作1.685个宽度单位，以尽量保持显示居中
    """
    return int(sum(1.685 if unicodedata.east_asian_width(c) in ('F', 'W', 'A') else 1 for c in text))


class CustomFormatter(logging.Formatter):
    """
    自定义日志格式，支持不同级别的日志颜色。
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
        格式化日志记录，添加颜色和前缀。
        """
        color, prefix = self.FORMATS.get(record.levelno, (Fore.WHITE, ""))
        original_message = super().format(record)  
        return f"{color}{prefix}{original_message}{Style.RESET_ALL}"


class CustomConsoleHandler(logging.StreamHandler):
    """
    自定义控制台日志处理器，支持 DEBUG 级别直接输出。
    """
    def emit(self, record):
        formatted_msg = self.format(record)
        print(formatted_msg, flush=True) if record.levelno == logging.DEBUG else super().emit(record)


class CustomLogger:
    """
    单例模式日志工具，确保同一名称的 logger 只被初始化一次。
    """
    _instances = {}  # 存储已创建的 logger 实例（单例模式）

    def __new__(cls, name='alphapilot'):
        if name not in cls._instances:
            instance = super().__new__(cls)  # 创建新实例
            instance._initialize_logger(name)  # 初始化日志系统
            cls._instances[name] = instance  # 存入字典，保证单例
        return cls._instances[name]  # 返回已有实例

    def _initialize_logger(self, name):
        """
        初始化 logger，添加文件和控制台处理器。
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别

        if self.logger.hasHandlers():
            self.logger.handlers.clear()  # 防止重复添加处理器

        # 文件日志处理器，按天滚动存储
        file_handler = TimedRotatingFileHandler(
            get_file_path('logs', f"{name.lower()}.log"), when='midnight',
            interval=1, backupCount=7
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))
        self.logger.addHandler(file_handler)

        # 控制台日志处理器，带颜色格式化
        console_handler = CustomConsoleHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter("%(message)s"))
        self.logger.addHandler(console_handler)


def get_logger(name=None) -> logging.Logger:
    """
    获取日志对象。
    """
    return CustomLogger(name or 'alphapilot').logger

def divider(name='', sep='=', _logger=None) -> None:
    """
    画一个带时间的横线
    :param name: 分割线中的文本
    :param sep: 分隔符
    :param _logger: 指定的 logger（可选）
    """
    seperator_len = 72
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    middle = f' {name} {now} '
    middle_width = get_display_width(middle)
    decoration_count = max(4, (seperator_len - middle_width) // 2)
    line = f"{sep * decoration_count}{middle}{sep * decoration_count}"

    if get_display_width(line) < seperator_len:
        line += sep

    log_target = _logger if _logger else logger
    log_target.debug(line)

logger = get_logger()

if __name__ == '__main__':
    logger.debug("调试信息，等同于print")
    logger.info("提示信息，绿色")
    logger.warning("警告信息，黄色")
    logger.error("错误信息，红色")
    logger.critical("重要提示，深红色")
    divider('分割线')
    print('日志都会在 `logs -> alphapilot.log` 文件中，不会丢失')
