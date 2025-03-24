#!/usr/bin/env python3
"""
测试所有重要模块的导入是否正确工作
"""
import os
import sys
import unittest

# 添加项目根目录到Python路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestImports(unittest.TestCase):
    """测试关键模块的导入"""
    
    def test_core_import(self):
        """测试核心包的导入"""
        import apilot
        self.assertTrue(hasattr(apilot, "__version__"))
        self.assertIsInstance(apilot.__version__, str)
    
    def test_event_import(self):
        """测试事件系统的导入"""
        try:
            from apilot.core.event import EventEngine
            self.assertTrue(True)
        except ImportError:
            self.fail("无法导入EventEngine")
    
    def test_trader_imports(self):
        """测试交易相关模块的导入"""
        from apilot.trader.engine import MainEngine
        from apilot.trader.object import OrderRequest
        self.assertTrue(MainEngine)
        self.assertTrue(OrderRequest)
    
    def test_cta_strategy_imports(self):
        """测试CTA策略模块的导入"""
        from apilot.cta_strategy import CtaTemplate
        from apilot.cta_strategy.backtesting import BacktestingEngine
        self.assertTrue(CtaTemplate)
        self.assertTrue(BacktestingEngine)
    
    def test_spread_strategy_imports(self):
        """测试价差交易策略模块的导入"""
        from apilot.spread_strategy import SpreadStrategyTemplate
        from apilot.spread_strategy.backtesting import BacktestingEngine as SpreadBacktestingEngine
        self.assertTrue(SpreadStrategyTemplate)
        self.assertTrue(SpreadBacktestingEngine)
    
    def test_portfolio_strategy_imports(self):
        """测试组合策略模块的导入"""
        from apilot.portfolio_strategy import PortfolioStrategyTemplate
        from apilot.portfolio_strategy.backtesting import BacktestingEngine as PortfolioBacktestingEngine
        self.assertTrue(PortfolioStrategyTemplate)
        self.assertTrue(PortfolioBacktestingEngine)


if __name__ == "__main__":
    unittest.main()
