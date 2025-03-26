"""
风控模块

提供风险控制功能，包括交易限制和风控规则。

主要组件:
- RiskEngine: 风险控制引擎，用于实施风控规则和监控交易风险

推荐用法:
    from apilot.risk import RiskEngine
    risk_engine = RiskEngine(main_engine)
"""

# 定义公共API
__all__ = [
    "RiskEngine",
    "ENGINE_NAME"
]

from .engine import RiskEngine, ENGINE_NAME
