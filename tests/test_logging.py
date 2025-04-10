"""
使用unittest测试事件驱动日志系统

测试内容:
1. 日志器单例模式
2. 不同名称的日志器
3. 不同日志级别处理
4. CustomLogger初始化
"""

import logging
import unittest
from unittest.mock import patch

from apilot.core.event import EventEngine
from apilot.utils.logger import CustomLogger, get_logger


class TestLogging(unittest.TestCase):
    """测试日志系统功能"""

    def setUp(self):
        """设置测试环境"""
        # 创建事件引擎
        self.event_engine = EventEngine()

        # 使用新的日志系统
        self.logger = get_logger("test")

        # 记录测试消息
        self.test_messages = []

    def tearDown(self):
        """清理测试资源"""
        # 关闭事件引擎
        self.event_engine.stop()

    def test_logger_singleton(self):
        """测试日志器是否是单例模式"""
        logger1 = get_logger("test_singleton")
        logger2 = get_logger("test_singleton")
        assert id(logger1) == id(logger2), "日志器应该是单例"

    def test_logger_different_names(self):
        """测试不同名称的日志器是否不同"""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        assert id(logger1) != id(logger2), "不同名称的日志器应该不同"

    def test_log_levels(self):
        """测试不同日志级别功能"""
        with (
            patch.object(logging.Logger, "debug") as mock_debug,
            patch.object(logging.Logger, "info") as mock_info,
            patch.object(logging.Logger, "warning") as mock_warning,
            patch.object(logging.Logger, "error") as mock_error,
            patch.object(logging.Logger, "critical") as mock_critical,
        ):
            logger = get_logger("test_levels")

            logger.debug("调试消息")
            logger.info("信息消息")
            logger.warning("警告消息")
            logger.error("错误消息")
            logger.critical("严重错误消息")

            mock_debug.assert_called_once()
            mock_info.assert_called_once()
            mock_warning.assert_called_once()
            mock_error.assert_called_once()
            mock_critical.assert_called_once()

    def test_custom_logger_initialization(self):
        """测试CustomLogger初始化"""
        custom_logger = CustomLogger("test_custom")
        assert custom_logger.logger is not None
        assert custom_logger.logger.name == "test_custom"


if __name__ == "__main__":
    unittest.main()
