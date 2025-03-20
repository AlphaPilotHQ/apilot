#!/usr/bin/env python3
"""
测试导入是否正确
"""

# 添加项目根目录到Python路径
import os
import sys
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    # 导入主要组件
    import apilot
    print(f"✓ 成功导入 apilot，版本: {apilot.__version__}")
    
    from apilot.event import EventEngine
    print("✓ 成功导入 EventEngine")
    
    from apilot.trader.engine import MainEngine
    print("✓ 成功导入 MainEngine")
    
    from apilot.trader.object import OrderRequest
    print("✓ 成功导入 OrderRequest")
    
    from apilot.cta_strategy import CtaTemplate
    print("✓ 成功导入 CtaTemplate")
    
    from apilot.cta_strategy.backtesting import BacktestingEngine
    print("✓ 成功导入 BacktestingEngine")
    
    # 在此处添加其他需要测试的导入
    
    print("\n所有导入测试通过！")
except Exception as e:
    print(f"导入失败: {e}")
