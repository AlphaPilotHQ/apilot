"""
简单测试脚本，检查新的binance_gateway.py是否能正确导入和初始化
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 尝试导入新的BinanceGateway
    from vnpy.gateway.binance_gateway import BinanceGateway
    print("✓ 成功导入 BinanceGateway 类")
    
    # 尝试导入必要的依赖
    from vnpy.event import EventEngine
    print("✓ 成功导入 EventEngine")
    
    # 尝试初始化网关（不连接）
    event_engine = EventEngine()
    gateway = BinanceGateway(event_engine)
    print("✓ 成功初始化 BinanceGateway 实例")
    
    # 检查网关属性
    print(f"✓ 网关名称: {gateway.gateway_name}")
    print(f"✓ 默认设置数量: {len(gateway.default_setting)}")
    print(f"✓ 支持的交易所: {gateway.exchanges}")
    
    print("\n测试完成: binance_gateway.py 结构正确!")

except Exception as e:
    print(f"❌ 测试失败: {e}")
