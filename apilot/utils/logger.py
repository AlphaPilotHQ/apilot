import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


def setup_logging(
    logger_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    json_format: bool = False,
):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    log_file = os.path.join(log_dir, f"{logger_name}.log")
    formatter = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
        if json_format
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    handlers = [TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)]
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
