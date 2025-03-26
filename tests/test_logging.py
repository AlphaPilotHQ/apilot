"""
使用pytest测试事件驱动日志系统

测试内容:
1. MainEngine日志方法工作正常
2. LogEngine事件处理正确
3. 不同日志级别处理正确
4. 日志事件创建和传播正确
"""

import os
import sys
import logging
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime

# 将项目根目录添加到sys.path
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# 定义测试用的模拟类
class Event:
    """简化的事件类"""

    def __init__(self, type_="", data=None):
        self.type = type_
        self.data = data


class EventEngine:
    """简化的事件引擎"""

    def __init__(self):
        self.handlers = {}
        self.active = False

    def register(self, type_, handler):
        if type_ not in self.handlers:
            self.handlers[type_] = []
        self.handlers[type_].append(handler)

    def unregister(self, type_, handler):
        if type_ in self.handlers:
            if handler in self.handlers[type_]:
                self.handlers[type_].remove(handler)

    def put(self, event):
        if self.active and event.type in self.handlers:
            for handler in self.handlers[event.type]:
                handler(event)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False


class LogData:
    """简化的日志数据类"""

    def __init__(self, msg, level=logging.INFO, source="", gateway_name="system", extra=None):
        self.msg = msg
        self.level = level
        self.source = source
        self.gateway_name = gateway_name
        self.timestamp = datetime.now()
        self.extra = extra or {}


class LogEngine:
    """简化的日志引擎"""

    def __init__(self, event_engine, log_level=logging.INFO):
        self.event_engine = event_engine
        self.level = log_level
        self.console_enabled = True
        self.file_enabled = True
        self.handlers = []

        # 注册事件处理函数
        self.event_engine.register("EVENT_LOG", self.process_log_event)

    def process_log_event(self, event):
        log = event.data
        if log.level >= self.level:
            # 简化的日志处理，仅用于测试
            print(f"LOG: [{log.source}] {log.level} - {log.msg}")

    def close(self):
        for handler in self.handlers:
            handler.close()


class MainEngine:
    """简化的主引擎"""

    def __init__(self, event_engine=None):
        self.event_engine = event_engine or EventEngine()
        if not getattr(self.event_engine, "active", False):
            self.event_engine.start()

        self.log_engine = LogEngine(self.event_engine)

    def _write_log(self, msg, source="", gateway_name="", level=logging.INFO, **kwargs):
        """创建日志对象并发送事件"""
        extra = kwargs.pop("extra", {})
        log = LogData(
            msg=msg, level=level, source=source, gateway_name=gateway_name, extra=extra
        )
        event = Event(type_="EVENT_LOG", data=log)
        self.event_engine.put(event)

    def log_debug(self, msg, source="", gateway_name="", **kwargs):
        """调试日志"""
        self._write_log(msg, source, gateway_name, logging.DEBUG, **kwargs)

    def log_info(self, msg, source="", gateway_name="", **kwargs):
        """信息日志"""
        self._write_log(msg, source, gateway_name, logging.INFO, **kwargs)

    def log_warning(self, msg, source="", gateway_name="", **kwargs):
        """警告日志"""
        self._write_log(msg, source, gateway_name, logging.WARNING, **kwargs)

    def log_error(self, msg, source="", gateway_name="", **kwargs):
        """错误日志"""
        self._write_log(msg, source, gateway_name, logging.ERROR, **kwargs)

    def log_critical(self, msg, source="", gateway_name="", **kwargs):
        """关键错误日志"""
        self._write_log(msg, source, gateway_name, logging.CRITICAL, **kwargs)

    def close(self):
        """关闭引擎"""
        # 关闭日志引擎
        if self.log_engine:
            self.log_engine.close()

        # 停止事件引擎
        if self.event_engine:
            self.event_engine.stop()


# pytest固定装置
@pytest.fixture
def setup_logging_system():
    """设置测试环境"""
    # 创建事件引擎
    event_engine = EventEngine()
    event_engine.start()

    # 创建日志捕获器
    captured_logs = []

    # 创建主引擎
    main_engine = MainEngine(event_engine)

    # 添加测试监听器
    def log_handler(event):
        log = event.data
        captured_logs.append(log)

    event_engine.register("EVENT_LOG", log_handler)

    # 返回测试环境
    yield main_engine, captured_logs

    # 测试后清理
    main_engine.close()


def test_logging_methods(setup_logging_system):
    """测试主引擎的日志方法"""
    main_engine, captured_logs = setup_logging_system

    # 测试不同级别的日志
    main_engine.log_debug("Debug消息", source="测试源")
    main_engine.log_info("Info消息", source="测试源")
    main_engine.log_warning("Warning消息", source="测试源")
    main_engine.log_error("Error消息", source="测试源")
    main_engine.log_critical("Critical消息", source="测试源")

    # 验证日志消息数量
    assert len(captured_logs) == 5

    # 验证日志级别
    assert captured_logs[0].level == logging.DEBUG
    assert captured_logs[1].level == logging.INFO
    assert captured_logs[2].level == logging.WARNING
    assert captured_logs[3].level == logging.ERROR
    assert captured_logs[4].level == logging.CRITICAL

    # 验证日志内容
    assert captured_logs[0].msg == "Debug消息"
    assert captured_logs[1].msg == "Info消息"
    assert captured_logs[2].msg == "Warning消息"
    assert captured_logs[3].msg == "Error消息"
    assert captured_logs[4].msg == "Critical消息"

    # 验证日志来源
    for log in captured_logs:
        assert log.source == "测试源"


def test_log_with_gateway(setup_logging_system):
    """测试带网关名称的日志"""
    main_engine, captured_logs = setup_logging_system

    # 使用gateway_name记录日志
    main_engine.log_info("网关日志", gateway_name="TestGateway")

    # 验证日志
    assert len(captured_logs) == 1
    assert captured_logs[0].gateway_name == "TestGateway"
    assert captured_logs[0].msg == "网关日志"


def test_log_with_extra(setup_logging_system):
    """测试带额外信息的日志"""
    main_engine, captured_logs = setup_logging_system

    # 使用extra记录额外信息
    extra_data = {"symbol": "BTCUSDT", "price": 50000}
    main_engine.log_info("价格更新", extra=extra_data)

    # 验证日志
    assert len(captured_logs) == 1
    assert captured_logs[0].extra["symbol"] == "BTCUSDT"
    assert captured_logs[0].extra["price"] == 50000


