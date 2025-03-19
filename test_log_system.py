#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试日志系统脚本
"""

import sys
import os
import warnings

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.cta_strategy.engine import CtaEngine
from vnpy.cta_strategy.base import APP_NAME


def test_main_engine_log():
    """测试主引擎日志方法"""
    print("\n=== 测试主引擎日志 ===")
    
    # 创建事件引擎和主引擎
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # 测试各个日志级别
    main_engine.log_debug("这是一条调试日志")
    main_engine.log_info("这是一条信息日志")
    main_engine.log_warning("这是一条警告日志")
    main_engine.log_error("这是一条错误日志")
    main_engine.log_critical("这是一条严重错误日志")
    
    # 测试带source参数的日志
    main_engine.log_info("这是一条来自测试模块的日志", source="TEST")
    
    # 测试已弃用的write_log方法
    with warnings.catch_warnings(record=True) as w:
        # 启用警告记录
        warnings.simplefilter("always")
        
        main_engine.write_log("这是使用弃用方法记录的日志")
        
        # 验证是否有弃用警告
        if len(w) > 0 and issubclass(w[0].category, DeprecationWarning):
            print("成功捕获到弃用警告")
    
    main_engine.close()
    print("主引擎日志测试完成")


def test_cta_engine_log():
    """测试CTA引擎日志方法"""
    print("\n=== 测试CTA引擎日志 ===")
    
    # 创建引擎
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    cta_engine = CtaEngine(main_engine, event_engine)
    
    # 测试弃用的write_log方法
    with warnings.catch_warnings(record=True) as w:
        # 启用警告记录
        warnings.simplefilter("always")
        
        cta_engine.write_log("这是使用CTA引擎弃用方法记录的日志")
        
        # 验证是否有弃用警告
        if len(w) > 0 and issubclass(w[0].category, DeprecationWarning):
            print("成功捕获到CTA引擎弃用警告")
    
    # 检查日志是否通过main_engine记录
    print(f"CTA引擎应该使用了主引擎的日志系统，检查上面是否有一条source为'{APP_NAME}'的日志")
    
    main_engine.close()
    print("CTA引擎日志测试完成")


if __name__ == "__main__":
    test_main_engine_log()
    test_cta_engine_log()
    
    print("\n所有测试完成，请查看上面的日志输出是否符合预期")
    print("请特别注意是否显示了弃用警告，以及CTA引擎的日志是否正确使用了主引擎的日志系统")
