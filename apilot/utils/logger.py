import logging
import os
import sys
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from typing import ClassVar

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


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

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler with daily rotation
        file_handler = TimedRotatingFileHandler(
            os.path.join(LOG_DIR, f"{name}.log"), when="midnight", backupCount=7
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        )

        # Console handler with standard formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
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
                logger.exception(f"Error in function {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    logger.debug("Debug message (cyan)")
    logger.info("Info message (green)")
    logger.warning("Warning message (yellow)")
    logger.error("Error message (red)")
    logger.critical("Critical message (bright red)")
